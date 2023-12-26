import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
import threading
import constants
import encoders
from agent_types import ObservationType

################################## set device ##################################

# set device to cpu or cuda
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")


def print_tensor(t, msg, four=False):
    num_of_states0 = t.size(0)
    num_of_states1 = t.size(1)
    num_of_states2 = t.size(2)
    suffix = ''
    if four:
        num_of_states3 = t.size(3)
        suffix = ', ' + str(num_of_states3)
    print(f"the {msg} is ({num_of_states0}, {num_of_states1}, {num_of_states2}{suffix})")


# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.costs = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.costs[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, k_last_states):
        super(ActorCritic, self).__init__()
        self.k_last_states = k_last_states
        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init).to(device)
        self.lstm = nn.LSTM(state_dim, 64, batch_first=True)

        # actor
        # for a given states returns the safety score per each action
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        # for a given state - "how good it is"
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    # Actor architecture is changed to LSTM, extracting the layer of the last state (with the needed information)
    def actor_forward(self, states, last_informative_layers):
        # adding unsqueeze(0) for batch_size = 1
        lstm_output, _ = self.lstm(states)
        # last time steps as output - representation of the sequence
        first_dim_len = len(lstm_output) - 1
        lstm_output_last = lstm_output[first_dim_len, last_informative_layers, :]
        # Pass the LSTM output through the actor network
        actor_output = self.actor(lstm_output_last)
        return actor_output.squeeze(0)  # dim [action_dim,] safety score per action

    # Critic architecture is changed to LSTM, extracting the layer of the last state (with the needed information)
    def critic_forward(self, states, last_informative_layers):
        lstm_output, _ = self.lstm(states)
        first_dim_len = len(lstm_output) - 1
        lstm_output_last = lstm_output[first_dim_len, last_informative_layers, :]
        critic_output = self.critic(lstm_output_last)
        return critic_output.squeeze(0)

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, states, last_informative_layers, action):
        if self.has_continuous_action_space:
            action_mean = self.actor_forward(states)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            if self.k_last_states == 1:
                action_probs = self.actor_forward(states, last_informative_layers)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic_forward(states, last_informative_layers)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, k_last_states=1):
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,
                                  k_last_states).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init,
                                      k_last_states).to(device)
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
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, valid_actions, timestep):
        # initalize tensor size action_dim (amount of actions to choose) with zeros
        valid_mask = torch.zeros(self.action_dim).to(device)
        # fill it with 1 in the indices of the valid-actions -> filter unwanted actions. 1 = valid action, 0 = not valid
        for a in valid_actions:
            valid_mask[a] = 1.0
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # predict the action_probs based on the last state using the actor network
                action_probs = self.policy_old.actor(state)
                # copy the actions probs with clone()
                action_probs_ = action_probs.clone()
                # in the indices that are not valid - put -inf so that it will not choose it
                action_probs_[valid_mask == 0] = -1e10
                # apply softmax (probabilities)
                action_probs_ = F.softmax(action_probs_)
                # categorial distribution - based on the original distribution (before softmax) and the soft-max probabilities
                dist = Categorical(action_probs_)
                dist2 = Categorical(action_probs)
                # sample based on score (an action)
                action = dist.sample()
                # compute it's log prob
                action_logprob = dist2.log_prob(action)
            # apply the buffer - (state, action)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            # return the chosen action - flattened
            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action_probs = self.policy_old.actor(state)
                action_probs_ = action_probs.clone()
                action_probs_[valid_mask == 0] = -1e10
                action_probs_ = F.softmax(action_probs_)
                dist = Categorical(action_probs_)
                dist2 = Categorical(action_probs)
                action = dist.sample()
                action_logprob = dist2.log_prob(action)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
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
        old_states_from_buffer, old_last_informative_layers = [state[0] for state in self.buffer.states], [state[1] for state in self.buffer.states]
        old_states = torch.squeeze(torch.stack(old_states_from_buffer, dim=0), 0).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_last_informative_layers, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            print("loss from update", loss.mean())

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


class PPOCostAsReward(PPO):

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal, cost in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals),
                                             reversed(self.buffer.costs)):
            if is_terminal:
                discounted_reward = 0
            if cost > 0:
                reward = -1000000
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


class ShieldBuffer:
    def __init__(self, n_max=1000000):
        # pos_rb - a buffer that saves tuple, each tuple is (s, last_informative_layer,a) - last_informative_layer before padding. for positive actions.
        # neg_rb - a buffer that saves tuple, each tuple is (s, last_informative_layer,a) - last_informative_layer before padding. for negative actions.
        # positive_action - no collision, negative_actions - result collision
        self.n_max = n_max
        self.pos_rb = []
        self.neg_rb = []

    def move_last_pos_to_neg(self):
        # Error Diffusion - "no safe action according to shield"
        if len(self.pos_rb) > 1:
            self.neg_rb.append(self.pos_rb[-1])
            self.pos_rb = self.pos_rb[:-1]

    def __len__(self):
        return len(self.pos_rb) + len(self.neg_rb)

    def add(self, s, last_informative_layer, a, label, obs_type):
        if label > 0.5:  # SAFE
            self.pos_rb.append((s, last_informative_layer, a, obs_type))
        else:  # COLLISION
            self.neg_rb.append((s, last_informative_layer, a, obs_type))

    def sample(self, n):
        # n_neg = min(n // 2, len(self.neg_rb))
        n_neg = min(max(n // 2, n - len(self.pos_rb)), len(self.neg_rb))
        # original - n_pos = n - n_neg
        n_pos = min(n - n_neg, len(self.pos_rb))
        pos_batch = random.sample(self.pos_rb, n_pos)
        # s_pos, a_pos = map(np.stack, zip(*pos_batch))
        s_pos = np.array([item[0] for item in pos_batch])
        last_informative_layers_pos = np.array([item[1] for item in pos_batch])
        a_pos = np.array([item[2] for item in pos_batch])
        neg_batch = random.sample(self.neg_rb, n_neg)
        # s_neg, a_neg = map(np.stack, zip(*neg_batch))
        s_neg = np.array([item[0] for item in neg_batch])
        last_informative_layers_neg = np.array([item[1] for item in neg_batch])
        a_neg = np.array([item[2] for item in neg_batch])
        # TODO - the following line raises an error when there is no padding..
        return torch.FloatTensor(s_pos).to(device), torch.FloatTensor(last_informative_layers_pos).to(device), torch.LongTensor(a_pos).to(device), \
            torch.FloatTensor(s_neg).to(device), torch.FloatTensor(last_informative_layers_neg).to(device), torch.LongTensor(a_neg).to(device)


class Shield(nn.Module):
    _instance = None
    _lock = threading.Lock()

    def __init__(self, action_dim, has_continuous_action_space, env_type):
        super().__init__()
        hidden_dim = 256
        feature_dim = 128  # TODO: check what is the shield input dimension
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim).to(device)
            self.action_embedding.weight.data = torch.eye(action_dim).to(device)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=1, batch_first=True).to(device)
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, 1).to(device)
        self.action_dim = action_dim
        self.loss_fn = nn.BCELoss()
        self.encoders_dict = self.build_encoders(env_type)

    def forward(self, s, last_informative_layers, a, obs_type):
        """
        tensor(8,3,70) - batch_size=8, k=3, state_dim=70
        Receives a batch of states and actions. (the batch can be size 1)
        s - last k_states (or less) for each sample in the batch (list of last k states)
        last_informative_layers - index of the last informative layer - for each sample in the batch
        a - tensor of actions - for each sample in the batch
        """
        a = self.encode_action(a)
        encoder = self.encoders_dict[obs_type]
        encoded_states = encoder.encode(s)
        lstm_output, _ = self.lstm(encoded_states)
        last_informative_layers_casted = [int(ind.item()) for ind in last_informative_layers]
        first_dim_len = len(lstm_output) - 1
        # select the last layer of the LSTM Shield network according to the index of the last layer (without padding)
        lstm_output_last = lstm_output[first_dim_len, last_informative_layers_casted, :]
        # lstm_output_last = torch.stack([t[int(last_informative_layers[i].item())] for i, t in enumerate(lstm_output)])
        x = torch.cat([lstm_output_last, a], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        safety_scores = torch.sigmoid(self.fc3(x)).squeeze(-1)
        return safety_scores.squeeze(-1)

    def encode_action(self, a):
        if self.has_continuous_action_space:
            return a
        else:
            return self.action_embedding(a.to(device))

    def loss(self, s_pos, last_informative_layers_pos, a_pos, s_neg, last_informative_layers_neg, a_neg, obs_type):
        print_tensor(s_pos, "s-pos")
        print_tensor(s_neg, "s-neg")
        y_pos = self.forward(s_pos, last_informative_layers_pos, a_pos, obs_type)
        y_neg = self.forward(s_neg, last_informative_layers_neg, a_neg, obs_type)
        loss = self.loss_fn(y_pos, torch.ones_like(y_pos)) + self.loss_fn(y_neg, torch.zeros_like(y_neg))
        return loss

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
                ObservationType.Kinematics: encoders.KinematicsEncoder(constants.VEHICLE_COUNT * constants.ENV_FEATURES_SIZE)
            }
        else:
            # Cart Pole
            encoders_dict = {
                ObservationType.Camera: encoders.CameraEncoder(constants.CP_IMAGE_WIDTH, constants.CP_IMAGE_HEIGHT, 3),
                ObservationType.Kinematics: encoders.KinematicsEncoder(constants.CP_OBS_SPACE)
            }

        return encoders_dict


class ShieldPPO(PPO):  # currently only discrete action
    _shield_lock = threading.Lock()

    def __init__(self, shield, obs_type, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, masking_threshold=0, k_last_states=1, safety_threshold=0.5):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                         has_continuous_action_space, action_std_init, k_last_states=1)
        self.shield = shield
        self.obs_type = obs_type
        self.action_dim = action_dim
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr=5e-4)
        self.shield_buffer = ShieldBuffer()
        self.masking_threshold = masking_threshold
        self.k_last_states = k_last_states
        self.state_dim = state_dim
        self.safety_treshold = safety_threshold

    def move_last_pos_to_neg(self):
        self.shield_buffer.move_last_pos_to_neg()

    def add_to_shield(self, s, last_informative_layer, a, label, obs_type):
        self.shield_buffer.add(s, last_informative_layer, a, label, obs_type)

    def update_shield(self, batch_size):
        with self._shield_lock:
            if len(self.shield_buffer.neg_rb) == 0:
                return 0.
            if len(self.shield_buffer) <= batch_size:
                batch_size = len(self.shield_buffer)
            loss_ = 0.
            # k steps to update the shield network - each epoch (step) is one forward and backward pass
            for i in range(self.K_epochs):
                # in each iteration - it samples batch_size positive and negative samples
                #  samples list of k_last_states states. for each sample.
                s_pos, last_informative_layers_pos, a_pos, s_neg, last_informative_layers_neg, a_neg = self.shield_buffer.sample(batch_size)
                self.shield_opt.zero_grad()
                # compute loss - binary cross entropy
                loss = self.shield.loss(s_pos, last_informative_layers_pos, a_pos, s_neg, last_informative_layers_neg, a_neg, self.obs_type)
                # back propagation
                loss.backward()
                # updating the shield parameters using Adam optimizer
                self.shield_opt.step()
                loss_ += loss.item()
            return loss_ / self.K_epochs

    def select_action(self, states, valid_actions, timestep):
        valid_mask = torch.zeros(self.action_dim).to(device)
        no_safe_action = False
        for a in valid_actions:
            valid_mask[a] = 1.0
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            # the last state is the states[-1] - because there is no padding
            last_state = states[-1]
            state = torch.FloatTensor(last_state).to(device)
            action_probs = self.policy_old.actor_forward(states_tensor.unsqueeze(0), len(states) - 1)
            # it will have self.k_last_states rows, each with the same action indices - for the batch prediction
            actions = torch.arange(self.action_dim).to(device)  # (n_action,)
            actions_ = [tensor.unsqueeze(0) for tensor in actions]
            # states_ - a batch of self.action_dim repeated states_ tensor, if self.action_dim is 5 so the batch_size is equal to 5.
            states_ = states_tensor.unsqueeze(0).repeat(self.action_dim, 1, 1)
            if timestep >= self.masking_threshold:
                # print("USING SAFETY MASKING")
                # BATCH_SIZE = self.action_dim,  EACH SEQUENCE LENGTH IS - K_LAST_STATES
                # Send the Shield network a batch of size self.action_dim
                last_informative_layer_repeated = torch.Tensor([len(states) - 1] * self.action_dim).to(torch.int)
                safety_scores = self.shield(states_, last_informative_layer_repeated, actions, self.obs_type)
                # safety_scores = [self.shield(states_, torch.tensor([len(states) - 1]), a, self.obs_type) for a in actions_]
                mask = safety_scores.clone().detach().gt(self.safety_treshold).float().to(device)
                mask = valid_mask * mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    # AT LEAST ONE ACTION IS CONSIDERED SAFE
                    action_probs_[mask == 0] = -1e10
                else:
                    # No safe action according to shield
                    # If it happened - it means that the 'problem' is in the action selected by previous state. The state is not safe so none of the actions is safe according to shield.
                    # print("No safe action according to shield")
                    no_safe_action = True
                    # Solution - Error Diffusion - teach  the network that the prev state prev state is not safe (from highway.py)
                    action_probs_[valid_mask == 0] = -1e10
            else:
                #   print("NOT USING SAFETY MASKING")
                mask = valid_mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    # at least one of the actions is safe according to shield or valid
                    action_probs_[mask == 0] = -1e10
                else:
                    print("NO VALID ACTIONS AT ALL - SHOULDN'T HAPPEN ")
                    action_probs_[valid_mask == 0] = -1e10
            action_probs_ = F.softmax(action_probs_, dim=0)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)
        # action - [k_last_states, action_dim] -
        action = dist.sample()
        action_logprob = dist2.log_prob(action)

        # convert list to tensor
        padding_rows = max(0, self.k_last_states - states_tensor.shape[0])
        if padding_rows > 0:
            # adding padded_states to the ActorCritic buffer.
            zeros = torch.zeros((padding_rows,) + states_tensor.shape[1:]).to(device)
            padded_states = torch.cat((states_tensor, zeros), dim=0)
        else:
            # If no padding is needed, keep the original tensor
            padded_states = states_tensor
        self.buffer.states.append((padded_states, len(states_tensor) - 1))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        if timestep >= self.masking_threshold:
            return action.item(), safety_scores, no_safe_action
        else:
            return action.item(), "no shield yet", "no shield yet"

    def save(self, checkpoint_path_ac, checkpoint_path_shield):
        # save actor critic networks
        torch.save(self.policy_old.state_dict(), checkpoint_path_ac)
        # save shield network
        torch.save(self.shield.state_dict(), checkpoint_path_shield)

    def load(self, checkpoint_path_ac, checkpoint_path_shield):
        self.policy_old.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.shield.load_state_dict(torch.load(checkpoint_path_shield, map_location=lambda storage, loc: storage))


class RuleBasedShieldPPO(PPO):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, safety_rules, action_std_init=0.6):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                         has_continuous_action_space, action_std_init)
        self.action_dim = action_dim
        self.safety_rules = safety_rules

    def add_to_shield(self, s, a, label):
        if not label:
            # print(s,a)
            # print('A mistake was detected')
            input("A mistake was detected, update rules and press Enter to continue...")
            self.safety_rules.parse_rules()

    def select_action(self, state, valid_actions):
        valid_mask = torch.zeros(self.action_dim).to(device)
        for a in valid_actions:
            valid_mask[a] = 1.0

        with torch.no_grad():
            original_state_representation = state
            state = torch.FloatTensor(state).to(device)
            action_probs = self.policy_old.actor(state)
            mask = [
                (1 if self.safety_rules.check_state_action_if_safe(original_state_representation, action_as_int) else 0)
                for action_as_int in range(self.action_dim)]
            mask = torch.FloatTensor(mask).to(device)
            mask = valid_mask * mask

            action_probs_ = action_probs.clone()
            if mask.sum().item() > 0:
                action_probs_[mask == 0] = -1e10
            else:
                action_probs_[valid_mask == 0] = -1e10
            action_probs_ = F.softmax(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist2.log_prob(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()
