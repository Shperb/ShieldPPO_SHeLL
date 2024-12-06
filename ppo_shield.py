import datetime

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import threading
from utils import constants
import encoders
from encoders import ObservationType

################################## set device ##################################
# set device to cpu or cuda
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.costs = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.costs[:]


class ActorCriticCNN(nn.Module):
    def __init__(self, num_actions):
        super(ActorCriticCNN, self).__init__()
        self.channels = 3
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU()
        )

        def conv2d_size_out(size, kernel_size=3, stride=1, pool=1):
            conv_size = (size - (kernel_size - 1) - 1) // stride + 1
            return conv_size // pool

        # self.w = constants.HW_IMAGE_WIDTH
        # self.h = constants.HW_IMAGE_HEIGHT
        self.w = constants.CP_IMAGE_WIDTH
        self.h = constants.CP_IMAGE_HEIGHT
        # self.w = constants.CR_IMAGE_WIDTH
        # self.h = constants.CR_IMAGE_HEIGHT
        kernel = 5
        stride = 2
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w, kernel, stride), kernel, stride), kernel, stride)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h, kernel, stride), kernel, stride), kernel, stride)
        linear_input_size = convw * convh * 32

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 256),
            nn.ReLU(),
        )
        self.w = constants.CP_IMAGE_WIDTH
        self.h = constants.CP_IMAGE_HEIGHT
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = x.to(device)
        if len(x.shape) > 1:
            batch_size = x.size(0)
            x = x.view(batch_size, self.channels, self.w, self.h)
            x = self.conv_layers(x)
            x = self.fc(x)
            x = x.view(batch_size, -1)
        else:
            x = x.view(1, self.channels, self.w, self.h)
            x = self.conv_layers(x)
            x = self.fc(x)
            x = x.view(-1)

        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values

    def act(self, state):
        action_probs, states_values = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), states_values.detach()

    def evaluate(self, state, action):
        action_probs, state_values = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, obs_type, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.obs_type = obs_type
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        if self.obs_type == ObservationType.Camera:
            self.channels = 1
            out_channels = 8
            last_out_channels = 16
            kernel1 = 5
            kernel2 = 3
            kernel3 = 3
            stride1 = 2
            stride2 = 1
            stride3 = 1
            # CNN
            self.conv = nn.Sequential(
                nn.Conv2d(self.channels, out_channels, kernel_size=kernel1, stride=stride1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(out_channels, last_out_channels, kernel_size=kernel2, stride=stride2),
                nn.BatchNorm2d(last_out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(last_out_channels, last_out_channels, kernel_size=kernel3, stride=stride3),
                nn.BatchNorm2d(last_out_channels),
                nn.ReLU()
            )

            # Calculate the size of the output from the last conv layer to be used in the first linear layer
            def conv2d_size_out(size, kernel_size=3, stride=1, pool=2):
                conv_size = (size - (kernel_size - 1) - 1) // stride + 1
                return conv_size // pool

            self.w = constants.HW_IMAGE_WIDTH
            self.h = constants.HW_IMAGE_HEIGHT
            # self.w = constants.CP_IMAGE_WIDTH
            # self.h = constants.CP_IMAGE_HEIGHT
            # self.w = constants.CR_IMAGE_WIDTH
            # self.h = constants.CR_IMAGE_HEIGHT
            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.w, kernel1, stride1), kernel2, stride2), kernel3, stride3, pool=1)
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.h, kernel1, stride1), kernel2, stride2), kernel3, stride3, pool=1)
            linear_input_size = convw * convh * last_out_channels

            if has_continuous_action_space:
                self.actor = nn.Sequential(
                    nn.Linear(linear_input_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim),
                    nn.Tanh()
                )
            else:
                self.actor = nn.Sequential(
                    nn.Linear(linear_input_size, 512),
                    nn.ReLU(),
                    # nn.Linear(512, 256),
                    # nn.ReLU(),
                    nn.Linear(512, action_dim),
                    nn.Softmax(dim=-1)
                )
                linear = nn.Linear(linear_input_size, 512)
                # print(linear.weight.requires_grad)

            self.critic = nn.Sequential(
                nn.Linear(linear_input_size, 512),
                nn.ReLU(),
                # nn.Linear(512, 256),
                # nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            if has_continuous_action_space:
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.Linear(256, action_dim),
                    nn.Tanh()
                )
            else:
                # actor
                self.actor = nn.Sequential(
                    nn.Linear(state_dim, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.Linear(256, action_dim),
                    nn.Softmax(dim=-1)
                )
            # critic
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def conv_forward(self, x):
        if self.obs_type != ObservationType.Camera:
            return x
        x = x.to(device)
        if len(x.shape) > 1:
            batch_size = x.size(0)
            x = x.view(batch_size, self.channels, self.w, self.h)
            x = self.conv(x)
            x = x.view(batch_size, -1)
        else:
            x = x.view(1, self.channels, self.w, self.h)
            x = self.conv(x)
            x = x.view(-1)
        return x

    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            state = self.conv_forward(state)
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            state = self.conv_forward(state)
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action_probs, action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            obs_state = self.conv_forward(state)
            action_mean = self.actor(obs_state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            obs_state = self.conv_forward(state)
            action_probs = self.actor(obs_state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs_state)
        return action_logprobs, state_values, dist_entropy

        # action_probs, log_probs, value_output = self.forward(state)
        # dist = Categorical(action_probs)
        # dist_entropy = dist.entropy()
        # return log_probs, value_output, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip, has_continuous_action_space, obs_type, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs_ppo = k_epochs_ppo

        self.buffer = RolloutBuffer()

        # self.policy = ActorCriticCNN(action_dim).to(device)
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, obs_type, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # self.policy_old = ActorCriticCNN(action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, obs_type, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # action, action_logprob, state_val = self.policy_old.forward(state)
                _, action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # action, action_logprob, state_val = self.policy_old.forward(state)
                _, action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs_ppo):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.1 * dist_entropy  # TODO: changed ent coef to 0.1 from 0.01

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class ShieldBuffer:
    def __init__(self, n_max=1000000):
        self.n_max = n_max
        self.pos_rb = []
        self.neg_rb = []

    def __len__(self):
        return len(self.pos_rb) + len(self.neg_rb)

    def move_last_pos_to_neg(self):
        # Error Diffusion - "no safe action according to shield"
        if len(self.pos_rb) > 1:
            self.neg_rb.append(self.pos_rb[-1])
            self.pos_rb = self.pos_rb[:-1]

    def add(self, s, a, label, obs_type):
        if label > 0.5:  # safe
            self.pos_rb.append((s, a, obs_type))
        else:
            self.neg_rb.append((s, a, obs_type))

    def sample(self, n):
        n_neg = min(max(n // 2, n - len(self.pos_rb)), len(self.neg_rb))
        n_pos = min(n - n_neg, len(self.pos_rb))
        pos_batch = random.sample(self.pos_rb, n_pos)
        s_pos, a_pos = map(np.array, zip(*[(item[0], item[1]) for item in pos_batch]))
        neg_batch = random.sample(self.neg_rb, n_neg)
        s_neg, a_neg = map(np.array, zip(*[(item[0], item[1]) for item in neg_batch]))
        return torch.FloatTensor(s_pos).to(device), torch.LongTensor(a_pos).to(device), \
            torch.FloatTensor(s_neg).to(device), torch.LongTensor(a_neg).to(device)


class Shield(nn.Module):
    _instance = None
    _lock = threading.Lock()

    def __init__(self, action_dim, has_continuous_action_space, env_type):
        super().__init__()
        # self.shield_buffer = ShieldBuffer()
        hidden_dim = 256
        feature_dim = encoders.feature_dim  # TODO: check what is the shield input dimension
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim).to(device)
            self.action_embedding.weight.data = torch.eye(action_dim).to(device)

        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()).to(device)

        self.action_dim = action_dim
        self.loss_fn = nn.BCELoss()
        self.encoders_dict = self.build_encoders(env_type)
        self.encoder1 = self.encoders_dict[ObservationType.Kinematics]
        # self.encoder2 = self.encoders_dict[ObservationType.Camera]
        self.encoder3 = self.encoders_dict[ObservationType.OccupancyGrid]
        self.param = self.named_parameters()

    def encode_action(self, a):
        if self.has_continuous_action_space:
            return a
        else:
            return self.action_embedding(a.to(device))

    def forward(self, s, a, obs_type, encode_state=True):
        a = self.encode_action(a)
        self.param = self.parameters()
        if encode_state:
            s = self.encode_state(s, obs_type)
        x = torch.cat([s, a], -1)
        return self.net(x)

    def loss(self, s_pos, a_pos, s_neg, a_neg, obs_type):
        y_pos = self.forward(s_pos, a_pos, obs_type)
        y_neg = self.forward(s_neg, a_neg, obs_type)
        loss = self.loss_fn(y_pos, torch.ones_like(y_pos)) + self.loss_fn(y_neg, torch.zeros_like(y_neg))
        return loss

    def encode_state(self, state, obs_type):
        encoder = self.encoders_dict[obs_type]
        encoded_states = encoder.encode(state)
        return encoded_states

    @staticmethod
    def get_shield(action_dim, has_continuous_action_space, env_type):
        with Shield._lock:
            if Shield._instance is None:
                Shield._instance = Shield(action_dim, has_continuous_action_space, env_type)
        return Shield._instance

    @staticmethod
    def build_encoders(env_type):
        if 'Highway' in env_type:
            # Highway
            encoders_dict = {
                ObservationType.Camera: encoders.CameraEncoder(constants.HW_IMAGE_WIDTH, constants.HW_IMAGE_HEIGHT, 1),
                ObservationType.Kinematics: encoders.KinematicsEncoder(
                    constants.VEHICLE_COUNT * constants.ENV_FEATURES_SIZE),
                ObservationType.OccupancyGrid: encoders.OccupancyGridEncoder(constants.OCCUPANCY_INPUT_SIZE)
            }
        elif 'CartPole' in env_type:
            # Cart Pole
            encoders_dict = {
                ObservationType.Camera: encoders.CameraEncoder(constants.CP_IMAGE_WIDTH, constants.CP_IMAGE_HEIGHT, 1),
                ObservationType.Kinematics: encoders.KinematicsEncoder(constants.CP_OBS_SPACE)
            }
        elif 'CarRacing' in env_type:
            # Car Racing
            encoders_dict = {
                ObservationType.Camera: encoders.CameraEncoder(constants.CR_IMAGE_WIDTH, constants.CR_IMAGE_HEIGHT, 3),
                ObservationType.Kinematics: encoders.KinematicsEncoder(constants.CP_OBS_SPACE)
            }

        return encoders_dict


class ShieldPPO(PPO):  # currently only discrete action
    _shield_lock = threading.Lock()

    def __init__(self, shield, obs_type, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, k_epochs_shield, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, masking_threshold=0, safety_threshold=0.5):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
                         has_continuous_action_space, obs_type, action_std_init)
        self.shield = shield
        self.obs_type = obs_type
        self.action_dim = action_dim
        self.s_params = self.shield.parameters()
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr=5e-4)
        self.shield_buffer = ShieldBuffer()
        self.safety_threshold = safety_threshold
        self.masking_threshold = masking_threshold
        self.state_dim = state_dim
        self.k_epochs_shield = k_epochs_shield

    def move_last_pos_to_neg(self):
        self.shield_buffer.move_last_pos_to_neg()

    def add_to_shield(self, s, a, label, obs_type):
        with torch.no_grad():
            self.shield_buffer.add(s, a, label, obs_type)

    def update_shield(self, batch_size):
        with self._shield_lock:
            if len(self.shield_buffer.neg_rb) == 0:
                return 0.
            if len(self.shield_buffer) <= batch_size:
                batch_size = len(self.shield_buffer)
            loss_ = 0.
            # print(f"Positive samples: {len(self.shield.shield_buffer.pos_rb)}     Negative samples: {len(self.shield.shield_buffer.neg_rb)}")
            for i in range(self.k_epochs_shield):
                s_pos, a_pos, s_neg, a_neg = self.shield_buffer.sample(batch_size)
                pre_update_params = {name: param.clone() for name, param in self.shield.named_parameters()}
                self.shield_opt.zero_grad()
                loss = self.shield.loss(s_pos, a_pos, s_neg, a_neg, self.obs_type)
                loss.backward()
                # TODO: lock object in step
                self.shield_opt.step()
                loss_ += loss.item()

                # self.shield.self.encoders_dict[ObservationType.Kinematics].update_encoder()

                # for name, param in self.shield.named_parameters():
                #     if not torch.equal(pre_update_params[name], param):
                #         print(f"Parameter '{name}' has been updated.")
                #     else:
                #         print(f"Parameter '{name}' has NOT been updated.")
            return loss_ / self.k_epochs_shield

    def select_action(self, state, valid_actions, timestep):
        valid_mask = torch.zeros(self.action_dim).to(device)
        no_safe_action = False
        for a in valid_actions:
            valid_mask[a] = 1.0
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # obs_state = self.policy_old.conv_forward(state)
            action_probs, _, _, state_val = self.policy_old.act(state)
            # action_probs = self.policy_old.actor(state)

            actions = torch.arange(self.action_dim).to(device)  # (n_action,)
            # the input for the shield is the state that returns 5 times
            state_ = state.view(1, -1).repeat(self.action_dim, 1)  # (n_action, state_dim)
            if timestep >= self.masking_threshold:
                # returns a safety score per each action from the state
                safety_scores = self.shield(state_, actions, self.obs_type)
                mask = safety_scores.gt(self.safety_threshold).float().view(-1)
                mask = valid_mask * mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    action_probs_[mask == 0] = 0
                else:
                    no_safe_action = True
                    action_probs_[valid_mask == 0] = 0
            else:
                mask = valid_mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    # at least one of the actions is safe according to shield or valid
                    action_probs_[mask == 0] = 0
                else:
                    print("NO VALID ACTIONS AT ALL - SHOULDN'T HAPPEN ")
                    action_probs_[valid_mask == 0] = 0
            action_probs_ = action_probs_ / sum(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist2.log_prob(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item(), no_safe_action

    def save(self, checkpoint_path_ac, checkpoint_path_shield):
        # save actor critic networks
        torch.save(self.policy_old.state_dict(), checkpoint_path_ac)
        # save shield network
        torch.save(self.shield.state_dict(), checkpoint_path_shield)

    def load(self, checkpoint_path_ac, checkpoint_path_shield):
        self.policy_old.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.shield.load_state_dict(torch.load(checkpoint_path_shield, map_location=lambda storage, loc: storage))
