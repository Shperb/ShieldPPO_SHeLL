import copy
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
from utils.priority_queue import PER_Buffer
from collections import OrderedDict

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
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, cost):
        self.buffer.append((s, a, cost))

    def sample(self, n):
        # n - amount of epochs to sample
        # Exclude last episode because we don't want a partial one (can be partial because we sample after certain amount of timesteps)
        # trajectories_to_sample = self.epoch_trajectories[:-1]
        epochs_batch = random.sample(self.buffer, n)
        states, actions, costs = zip(*epochs_batch)
        return torch.stack(states).to(device), torch.stack(actions).to(device), torch.tensor(costs).to(device)


class Shield(nn.Module):
    _instance = None
    _lock = threading.Lock()

    def __init__(self, action_dim, has_continuous_action_space, env_type, obs_types):
        super().__init__()
        hidden_dim = 256
        feature_dim = encoders.feature_dim  # TODO: determine what is the shield input dimension
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim).to(device)
            self.action_embedding.weight.data = torch.eye(action_dim).to(device)

        dropout_rate = 0.3

        self.linear1 = nn.Linear(feature_dim + action_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.linear2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.linear3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout_rate)

        self.linear4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        self.shield_layers = [self.linear1, self.linear2, self.linear3, self.linear4]
        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        self.action_dim = action_dim
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.BCELoss()

        self.encoders_dict = self.build_encoders(env_type)
        for obs_type in obs_types:
            setattr(self, f'encoder{obs_type.value}', self.encoders_dict[obs_type])

        self.param = self.named_parameters()

        self.global_itr = 0

    def encode_action(self, a):
        if self.has_continuous_action_space:
            return a
        else:
            a = self.action_embedding(a.to(device))
            if a.dim() == 1:
                a = a.unsqueeze(0)
            elif a.dim() == 3:
                a = a.squeeze(1)
            return a

    def forward(self, s, a, obs_type, encode_state=True):
        a = self.encode_action(a)
        self.param = self.named_parameters()
        if encode_state:
            s = self.encode_state(s, obs_type)
        x = torch.cat([s, a], -1)
        return self.net(x)

    def net(self, x):
        x = self.linear1(x)
        if x.size(0) > 1:  # Apply batch normalization only if batch size is greater than 1
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        if x.size(0) > 1:  # Apply batch normalization only if batch size is greater than 1
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        if x.size(0) > 1:  # Apply batch normalization only if batch size is greater than 1
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.linear4(x)
        x = self.sigmoid(x)

        return x

    def loss(self, s, a, obs_type, cost):
        cost = cost.view(-1, 1)
        if a.dim() == 2:
            a = a.squeeze(1)
        y = self.forward(s, a, obs_type)
        loss = self.loss_fn(y, cost)
        # loss = self.weighted_mse_loss(y, cost)

        l2_reg = 0.0
        # shield parameters
        for layer in self.shield_layers:
            for param in layer.parameters():
                l2_reg += torch.norm(param, 2)

        # relevant encoder parameters
        for param in self.encoders_dict[obs_type].parameters():
            l2_reg += torch.norm(param, 2)

        # Add regularization term to the loss
        lambda_reg = 0.005
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
    def get_shield(action_dim, has_continuous_action_space, env_type, obs_types):
        with Shield._lock:
            if Shield._instance is None:
                Shield._instance = Shield(action_dim, has_continuous_action_space, env_type, obs_types)
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
                 has_continuous_action_space, folder_name, action_std_init=0.6, masking_threshold=0, safety_threshold=0.5, shield_gamma=0.6):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
                         has_continuous_action_space, obs_type, action_std_init)
        self.global_shield = shield
        self.local_shield = Shield(action_dim, has_continuous_action_space, folder_name, [obs_type])
        self.local_optimizer = torch.optim.Adam(self.local_shield.parameters(), lr=5e-4)
        self.obs_type = obs_type
        self.action_dim = action_dim
        self.s_params = self.global_shield.parameters()
        self.shield_buffer = PER_Buffer(constants.SHIELD_BUFFER_SIZE)
        self.safety_threshold = safety_threshold
        self.masking_threshold = masking_threshold
        self.state_dim = state_dim
        self.k_epochs_shield = k_epochs_shield
        self.shield_gamma = shield_gamma
        self.shield_updates = 1
        self.alpha = 0.9
        self.buffer_error = nn.L1Loss()

    def compute_buffer_error(self, state, action, cost):
        """compute MAE error, for experience replay buffer."""
        # pass (s,a) in the network
        y = self.local_shield.forward(state.to(device), action.to(device), self.obs_type)
        # cost is the real label
        cost = cost.view(-1, 1)
        # compute MAE loss
        loss = self.buffer_error(y.to(device), cost.to(device))
        return loss

    def add_to_shield(self, episode_trajectory):
        with torch.no_grad():
            costs = self.calc_episode_costs(episode_trajectory)
            for tup, cost in zip(episode_trajectory, costs):
                s, a = tup[0], tup[1]
                cost = torch.tensor(cost).to(device)
                error = self.compute_buffer_error(s, a, cost)
                self.shield_buffer.add(error, (s, a, cost))

    def update_shield(self, batch_size):
        start = datetime.datetime.now()

        shield_buffer_size = len(self.shield_buffer)
        if shield_buffer_size == 0:
            return 0.
        if shield_buffer_size <= batch_size:
            batch_size = shield_buffer_size

        batch_samples, idxs, _ = self.shield_buffer.sample(batch_size)
        states, actions, costs = zip(*batch_samples)
        _states, _actions, _costs = torch.stack(states).to(device), torch.stack(actions).to(device), torch.tensor(costs).to(device)

        loss_ = 0.
        for i in range(self.k_epochs_shield):
            self.local_shield.optimizer.zero_grad()
            loss = self.local_shield.loss(_states, _actions, self.obs_type, _costs)
            loss.backward()
            self.local_shield.optimizer.step()
            loss_ += loss.item()

        self.shield_updates += 1

        if self.shield_updates % 4 == 0:
            print(f'Agent {self.obs_type} updating shared shield...')
            self.update_shared_shield()
            self.save(constants.AC_PATH, constants.SHIELD_PATH)

        end = datetime.datetime.now() - start
        print(f"Batch size is: {batch_size} and it took {end} to update the shield")

        # update the buffer with the new priorities
        for i, sample in enumerate(batch_samples):
            sample_idx = idxs[i]
            state, action, cost = sample
            error = self.compute_buffer_error(state.unsqueeze(0), action, cost)
            self.shield_buffer.update(sample_idx, error)

        # return loss average across all the instances in the batch
        return loss_ / self.k_epochs_shield

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
                safety_scores = self.local_shield(state_, actions, self.obs_type)
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

    def update_shared_shield(self):
        with self._shield_lock:  # Lock to ensure atomic operation
            local_dict = self.local_shield.state_dict()
            global_dict = self.global_shield.state_dict()

            # Update global model weights using the new local model weights
            for key in local_dict.keys():
                global_dict[key] = ((1 - self.alpha) * global_dict[key]) + (self.alpha * local_dict[key])
            self.global_shield.load_state_dict(global_dict)

            # Update local model with the new global model weights
            filtered_state_dict = OrderedDict((k, v) for k, v in global_dict.items() if k in local_dict)
            self.local_shield.load_state_dict(filtered_state_dict)

            # Update alpha
            self.alpha = 0.99 * self.alpha

    def update_local_shield(self, states, actions, obs_type, costs):
        self.local_optimizer.zero_grad()
        loss = self.local_shield.loss(states, actions, obs_type, costs)
        loss.backward()
        self.local_optimizer.step()
        return loss

    def sync_local_with_shared(self):
        with self._shield_lock:  # Lock to ensure atomic operation
            for local_param, shared_param in zip(self.local_shield.parameters(), self.global_shield.parameters()):
                local_param.data.copy_(shared_param.data)

    def save(self, checkpoint_path_ac, checkpoint_path_shield):
        # save actor critic networks
        torch.save(self.policy_old.state_dict(), checkpoint_path_ac)
        # save shield network
        torch.save(self.global_shield.state_dict(), checkpoint_path_shield)

    def load(self, checkpoint_path_ac, checkpoint_path_shield):
        self.policy_old.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.global_shield.load_state_dict(torch.load(checkpoint_path_shield, map_location=lambda storage, loc: storage))
