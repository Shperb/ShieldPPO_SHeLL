import datetime

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import threading
import constants
import encoders
from encoders import ObservationType
from ppo_shield import PPO, device

# set device to cpu or cuda
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")


class ShieldBuffer:
    def __init__(self, n_max=1000000):
        self.n_max = n_max
        self.epoch_trajectories = []

    def __len__(self):
        return len(self.epoch_trajectories)

    def add(self, epoch_trajectory):
        self.epoch_trajectories.append(epoch_trajectory)

    def sample(self, n):
        # n - amount of epochs to sample
        # Exclude last epoch because we don't want a partial one (can be partial because we sample after certain amount of timesteps)
        trajectories_to_sample = self.epoch_trajectories[:-1]
        epochs_batch = random.sample(trajectories_to_sample, n - 1)
        return epochs_batch


class Shield(nn.Module):
    _instance = None
    _lock = threading.Lock()

    def __init__(self, action_dim, has_continuous_action_space, env_type):
        super().__init__()
        hidden_dim = 256
        feature_dim = encoders.feature_dim  # TODO: determine what is the shield input dimension
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim).to(device)
            self.action_embedding.weight.data = torch.eye(action_dim).to(device)

        # self.net = nn.Sequential(
        #     nn.Linear(feature_dim + action_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),
        #     nn.Sigmoid()
        # ).to(device)

        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        self.action_dim = action_dim
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.BCELoss()
        self.encoders_dict = self.build_encoders(env_type)
        self.encoder1 = self.encoders_dict[ObservationType.Kinematics]
        # self.encoder2 = self.encoders_dict[ObservationType.Camera]
        self.encoder3 = self.encoders_dict[ObservationType.OccupancyGrid]
        self.param = self.named_parameters()

    def encode_action(self, a):
        if self.has_continuous_action_space:
            return a
        else:
            a = self.action_embedding(a.to(device))
            if not torch.cuda.is_available() and a.dim() == 1:
                a = a.unsqueeze(0)
            return a

    def forward(self, s, a, obs_type, encode_state=True):
        a = self.encode_action(a)
        self.param = self.named_parameters()
        if encode_state:
            s = self.encode_state(s, obs_type)
        x = torch.cat([s, a], -1)
        return self.net(x)

    def loss(self, s, a, obs_type, cost):
        cost = cost.view(-1, 1)
        y = self.forward(s, a, obs_type)
        loss = self.loss_fn(y, cost)
        # loss = self.weighted_mse_loss(y, cost)

        l2_reg = 0.0
        # shield parameters
        for param in self.net.parameters():
            l2_reg += torch.norm(param, 2)

        # relevant encoder parameters
        for param in self.encoders_dict[obs_type].parameters():
            l2_reg += torch.norm(param, 2)

        # Add regularization term to the loss
        lambda_reg = 0.01  # 0.0001
        loss += lambda_reg * l2_reg

        return loss

    def weighted_mse_loss(self, pred, target):
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.0]).to(device)
        return (weights * (pred - target) ** 2).mean()

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
                ObservationType.Kinematics: encoders.KinematicsEncoder(constants.VEHICLE_COUNT * constants.ENV_FEATURES_SIZE),
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
                 has_continuous_action_space, action_std_init=0.6, masking_threshold=0, safety_threshold=0.5, shield_gamma=0.6):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
                         has_continuous_action_space, obs_type, action_std_init)
        self.shield = shield
        self.obs_type = obs_type
        self.action_dim = action_dim
        self.s_params = self.shield.parameters()
        self.shield_buffer = ShieldBuffer()
        self.safety_threshold = safety_threshold
        self.masking_threshold = masking_threshold
        self.state_dim = state_dim
        self.k_epochs_shield = k_epochs_shield
        self.shield_gamma = shield_gamma

    def add_to_shield(self, epoch_trajectory):
        with torch.no_grad():
            self.shield_buffer.add(epoch_trajectory)

    def update_shield(self, batch_size):
        start = datetime.datetime.now()

        shield_buffer_size = len(self.shield_buffer.epoch_trajectories)
        if shield_buffer_size == 0:
            return 0.
        if shield_buffer_size <= batch_size:
            batch_size = shield_buffer_size
        # Sampling batch_size episodes from Shield buffer
        episodes_batch = self.shield_buffer.sample(batch_size)
        # Save average loss across all the episodes in batch (length will be shield_episodes_batch_size)
        episodes_batch_loss = []

        for episode_traj in episodes_batch:
            costs = self.calc_episode_costs(episode_traj)

            states_ = [step[0] for step in reversed(episode_traj)]
            actions_ = [step[1] for step in reversed(episode_traj)]

            states = torch.squeeze(torch.stack(states_, dim=0)).detach().to(device)
            actions = torch.squeeze(torch.stack(actions_, dim=0)).detach().to(device)

            episode_loss = 0.
            for i in range(self.k_epochs_shield):
                # pre_update_params = {name: param.clone() for name, param in self.shield.named_parameters()}
                with self._shield_lock:
                    self.shield.optimizer.zero_grad()
                    loss = self.shield.loss(states, actions, self.obs_type, torch.tensor(costs).to(device))
                    loss.backward()
                    self.shield.optimizer.step()
                episode_loss += loss.item()

                # self.shield.encoders_dict[ObservationType.Kinematics].update_encoder()

                # for name, param in self.shield.named_parameters():
                #     print(f'Layer: {name} | Gradients: {param.grad}')

                # print("")
            average_episode_loss = episode_loss / self.k_epochs_shield
            episodes_batch_loss.append(average_episode_loss)

        end = datetime.datetime.now() - start
        print(f"Batch size is: {batch_size} and it took {end} to update the shield")
        # return loss average across all the instances in the batch
        return sum(episodes_batch_loss) / batch_size

    def calc_episode_costs(self, episode_traj):
        """
        This function calculates the cost for the state-action pairs that led to the termination of the episode using bellman update.
        """
        discounted_cost = 0
        costs = []
        # reverse iterating through the episode - from end to beginning
        for step in zip(reversed(episode_traj)):
            _, _, cost, is_terminal = step[0]
            if is_terminal:
                discounted_cost = 0
            discounted_cost = cost + (self.shield_gamma * discounted_cost)
            costs.insert(0, discounted_cost)
        return costs

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
                mask = safety_scores.lt(self.safety_threshold).float().view(-1)
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