# ppo_shieldLSTM 26.09 before changes (from GPU cluster storage)

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

# each input for 'point in time' is state-action pair, so each sequence len is k_last_states length and each batch_size is 1.
# we have batch_size = 1 which is a disavantages.


################################## set device ##################################

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


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

    def actor_forward(self, states):
       # adding unsqueeze(0) for batch_size=1
        lstm_output, _ = self.lstm(states)
        # last time steps's as oiutput - representation of the sequence
        lstm_output_last = lstm_output[:, -1, :]
        # Pass the LSTM output through the actor network
        actor_output = self.actor(lstm_output_last)
        return actor_output.squeeze(0) # dim [action_dim,] safety score per action

    def critic_forward(self, states):
        lstm_output, _ = self.lstm(states)
        lstm_output_last = lstm_output[:, -1, :]
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

    def evaluate(self, states, action):
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
                action_probs = self.actor_forward(states)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic_forward(states)
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

    def select_action(self, state, valid_actions):
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
        # TODO - good for padding - check this. THE SQUEEZE REMOVES ZEROS
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0), 0).detach().to(device)
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
        self.n_max = n_max
        self.pos_rb = []
        self.neg_rb = []

    def __len__(self):
        return len(self.pos_rb) + len(self.neg_rb)

    def add(self, s, a, label):
        if label > 0.5:  # SAFE
            self.pos_rb.append((s, a))
        else: # COLLISION
            self.neg_rb.append((s, a))

    def sample(self, n):
        n_neg = min(n // 2, len(self.neg_rb))
        n_pos = n - n_neg
        pos_batch = random.sample(self.pos_rb, n_pos)

        s_pos, a_pos = map(np.stack, zip(*pos_batch))
        neg_batch = random.sample(self.neg_rb, n_neg)
        s_neg, a_neg = map(np.stack, zip(*neg_batch))
        return torch.FloatTensor(s_pos).to(device), torch.LongTensor(a_pos).to(device), \
               torch.FloatTensor(s_neg).to(device), torch.LongTensor(a_neg).to(device)


class Shield(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, k_last_states):
        super().__init__()

        hidden_dim = 256
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim)
            self.action_embedding.weight.data = torch.eye(action_dim)
        self.lstm = nn.LSTM(state_dim , hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.action_dim = action_dim
        self.loss_fn = nn.BCELoss()

    def encode_action(self, a):
        if self.has_continuous_action_space:
            return a
        else:
            return self.action_embedding(a)

    def forward(self, s, a):
        # Convert input to the same dtype as LSTM parameters
        #x = x.to(self.lstm.weight_hh_l0.dtype)
        a = self.encode_action(a)
        lstm_output, _ = self.lstm(s)
        # Selecting the last layer - hidden state which captures the relevant information from entire sequence (states)
        lstm_output_last = lstm_output[:, -1, :]  # Shape: [1, hidden_dim]
        x = torch.cat([lstm_output_last, a], -1)  # Shape: [action_dim, hidden_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        safety_scores = torch.sigmoid(self.fc3(x)).squeeze(-1) # Shape: [action_dim, 1]
        # safety_scores - shaped [action_dim] each element represents the safety score for an action
        return safety_scores.squeeze(-1)  # Shape: [action_dim]


    def loss(self, s_pos, a_pos, s_neg, a_neg):
        y_pos = self.forward(s_pos, a_pos)
        y_neg = self.forward(s_neg, a_neg)
        loss = self.loss_fn(y_pos, torch.ones_like(y_pos)) + self.loss_fn(y_neg, torch.zeros_like(y_neg))
        return loss

class ShieldPPO(PPO):  # currently only discrete action
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6, masking_threshold=0, k_last_states=1):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                         has_continuous_action_space, action_std_init, k_last_states=1)
        self.action_dim = action_dim
        self.shield = Shield(state_dim, action_dim, has_continuous_action_space, k_last_states=k_last_states).to(device)
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr=5e-4)
        self.shield_buffer = ShieldBuffer()
        self.masking_threshold = masking_threshold
        self.k_last_states = k_last_states
        self.state_dim = state_dim

    def add_to_shield(self, s, a, label):
        self.shield_buffer.add(s, a, label)

    def update_shield(self, batch_size):
        if len(self.shield_buffer.neg_rb) == 0:
            return 0.
        if len(self.shield_buffer) <= batch_size:
            batch_size = len(self.shield_buffer)
        loss_ = 0.
        # k steps to update the shield network - each epoch (step) is one forward and backward pass
        for i in range(self.K_epochs):
            # in each iteration - it samples batch_size positive and negative samples
            # now it samples list of states
            s_pos, a_pos, s_neg, a_neg = self.shield_buffer.sample(batch_size)
            self.shield_opt.zero_grad()
            # compute loss - binary cross entropy
            loss = self.shield.loss(s_pos, a_pos, s_neg, a_neg)
            # back propogation
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
            states_tensor = torch.FloatTensor(states).to(device)
            last_state = states[-1]
            state = torch.FloatTensor(last_state).to(device)
            action_probs = self.policy_old.actor_forward(states_tensor.unsqueeze(0))
            # it will have self.k_last_states rows, each with the same action indices
            # Prepare the actions tensor based on the number of states
            actions = torch.arange(self.action_dim).to(device)  # (n_action,)
            # the input for the shield is the state that returns 5 times
            states_ = states_tensor.unsqueeze(0).repeat(self.action_dim, 1, 1)  # (n_action, state_dim)
            if timestep >= self.masking_threshold:
             #   print("USING SAFETY MASKING")
                # TODO -  encode_action (?)

        ## BATCH_SIZE = 1 SEQEUENCE = 1, EACH SEQUENCE LENGTH IS - K_LAST_STATES
                safety_scores = self.shield(states_, actions)
                mask = safety_scores.gt(0.5).float()
                mask = valid_mask * mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                     # AT LEAST ONE ACTION IS CONSIDERED SAFE
                    action_probs_[mask == 0] = -1e10
                else:
                    # No safe action according to shield (all zeros - below 0.5)
                    print("No safe action according to shield")
                    no_safe_action = True
                    # TODO - NO SAFE ACTION ACCORDING TO SHIELD - MABYE THE STATE IS UNSAFE FROM THE BEGGINING
                    # TODO -  WE SHOULD LET THE NETWORK KNOW THAT THIS STATE IS UNSAFE WITH THE ACTION THAT LED TO IT
                    # TODO - PRINT - THE AMOUNT OF TIMES IT ARIVES HERE IN THE CODE - AND IF SO WHY?
                    action_probs_[valid_mask == 0] = -1e10
            else:
                #   print("NOT USING SAFETY MASKING")
                mask = valid_mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    # at least one of the actions is safe according to shield or valid
                    action_probs_[mask == 0] = -1e10
                else:
                    # TODO - should NOT arrive here - CHECK THIS! does it happen? no valid actions at all.
                    print("NO VALID ACTIONS AT ALL - SHOULDN'T HAPPEN ")
                    action_probs_[valid_mask == 0] = -1e10
            ## FOR ALL STATES (SEQUENCE)
            action_probs_ = F.softmax(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)
        # action - [k_last_states, action_dim] -
        action = dist.sample()
        # action_logprob = action_probs_[-1, action].log()
        action_logprob = dist2.log_prob(action)

        # convert list to tensor
        padding_rows = max(0, self.k_last_states - states_tensor.shape[0])
        if padding_rows > 0:
            # Create a tensor of zeros with the same shape as the states except for the first dimension
            zeros = torch.zeros((padding_rows,) + states_tensor.shape[1:])
            padded_states = torch.cat((zeros, states_tensor), dim=0)
        else:
            # If no padding is needed, keep the original tensor
            padded_states = states_tensor
        self.buffer.states.append(padded_states)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        if 'safety_scores' not in globals():
          safety_scores = 'No shield masking yet'
        return action.item(), safety_scores, no_safe_action

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



"-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

# highway_shieldLSTM 26.09 before changes (from GPU cluster storage)

import pandas as pd
import argparse
import collections
import json
import gym
import os
import glob
import time
from datetime import datetime
import torch
import torch.nn as nn
from gym.utils import seeding
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from highway_env.envs import HighwayEnvFast, MergeEnv
# import tensorflow as tf
# import safe_rl
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import highway_env
# from SafetyRulesParser import SafetyRulesParser
from ppo_shieldLSTM import PPO, ShieldPPO, RuleBasedShieldPPO, PPOCostAsReward
from gym import spaces, register
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        dim_state = np.prod(self.env.observation_space.shape)
        self.observation_space = spaces.Box(low=low.reshape(-1),
                                            high=high.reshape(-1),
                                            shape=(dim_state,),
                                            dtype=np.float32)

    def observation(self, obs):
        return obs.reshape(-1)

    def reset(self):
        # returns the first observation
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        # returns the next observation from the current state + given action, and reward& info
        next_obs, reward, done, info = self.env.step(action)
        cost = self.env._cost(0)
        info['cost'] = cost
        return self.observation(next_obs), reward, done, info


class EnvsWrapper(gym.Wrapper):
    def __init__(self, envs, has_continuous_action_space=False, seed=46456, no_render=False):
        self.envs = envs  # the first environment is assumed to have the full set of actions
        self.np_random = self.np_random, _ = seeding.np_random(seed)
        self.env_index = self.np_random.randint(0, len(envs))
        super().__init__(self.envs[self.env_index])
        self.state_dim = np.prod(self.env.observation_space.shape) + len(envs)  # low = self.env.observation_space.low
        self.no_render = no_render
        if self.no_render:
            self.env.render_mode = 'no_render'
        # high = self.env.observation_space.high
        # dim_state = np.prod(self.env.observation_space.shape)
        # self.observation_space = spaces.Box(low=low.reshape(-1),
        #                                     high=high.reshape(-1),
        #                                     shape=(dim_state,),
        #                                     dtype=np.float32)
        # dim_state =no_render

        self.actions = envs[0].action_type.actions
        self.has_continuous_action_space = has_continuous_action_space

    def observation(self, obs):
        one_hot_task = np.zeros(len(self.envs))
        one_hot_task[self.env_index] = 1
        return np.append(obs, one_hot_task)

    def reset(self):
        self.env_index = self.np_random.randint(0, len(self.envs))
        self.env = self.envs[self.env_index]
        # low = self.env.observation_space.low
        # high = self.env.observation_space.high
        # dim_state = np.prod(self.env.observation_space.shape)
        # self.observation_space = spaces.Box(low=low.reshape(-1),
        #                                     high=high.reshape(-1),
        #                                     shape=(dim_state,),
        #                                     dtype=np.float32)
        obs = self.env.reset()
        # return obs also because its shape is (number_of_vehicels, number_of_features) -> good for the log
        return self.observation(obs), obs

    def step(self, action):
        try:
            action = action[0]
        except:
            pass
        mapped_action = list(self.env.action_type.actions.keys())[
            list(self.env.action_type.actions.values()).index(self.actions[action])]
        next_obs, reward, done, info = self.env.step(mapped_action)
        cost = self.env._cost(mapped_action)
        info['cost'] = cost
        # return next_obs also because its shape is (number_of_vehicels, number_of_features) -> good for the log
        return self.observation(next_obs), reward, done, info, next_obs

    def action_space_size(self):
        if self.has_continuous_action_space:
            action_dim = self.envs[0].action_space.shape[0]
        else:
            action_dim = self.envs[0].action_space.n
        return action_dim

    def active_action_space_size(self):
        if self.has_continuous_action_space:
            action_dim = self.env.action_space.shape[0]
        else:
            action_dim = self.env.action_space.n
        return action_dim

    def get_valid_actions(self):
        values = self.env.action_type.actions.values()
        valid_actions = []
        for item in self.actions.items():
            if item[1] in values:
                valid_actions.append(item[0])
        return valid_actions

    def get_current_env_name(self):
        return type(self.env).__name__


def register_envs(envs):
    # create end points for the environments
    suffix = '-v0'
    for env in envs:
        entry = env[:-len(suffix)]
        register(
            id=env,
            entry_point='envs:' + entry,
        )


def train(arguments=None):
    # max_training_timestamps - the maximal number of interactions of the agent with the environment
    # max_ep_len - maximum number of steps in each episode
    """
    time step = interaction with the environment (outer loop) = one action
    episodes = sequence of interactions
    each iteration in the inner loop is a new episodes (sequence of actions) and the number of actions is limited due to max_timesteps
    trajectory - sequence of (state,action) pairs that the agent encounters during its interaction with the environment within a single episode.
    """
    global env, rewards

    #        ppo_agent = ShieldPPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    # has_continuous_action_space, action_std)
    global env, rewards
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        help="algorithm to use:  PPO | ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--envs", default=["HighwayEnvFastNoNormalization-v0"], nargs="+",
                        help="names of the environment to train on")
    parser.add_argument("--print_freq", default=1000,
                        help="print avg reward in the interval (in num timesteps)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log_freq", type=int, default=1000,
                        help="log avg reward in the interval (in num timesteps)")
    parser.add_argument("--save_model_freq", type=int, default=int(2.5e4),
                        help="save model frequency (in num timesteps)")
    parser.add_argument("--K_epochs", type=int, default=10,
                        help="update policy for K epochs")
    parser.add_argument("--eps_clip", type=float, default=0.1,
                        help="clip parameter for PPO")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="discount factor")
    parser.add_argument("--lr_actor", type=float, default=0.0005,
                        help="learning rate for actor network")
    parser.add_argument("--lr_critic", type=float, default=0.0001,
                        help="learning rate for critic network")
    parser.add_argument("--max_ep_len", type=int, default=500,
                        help="max timesteps in one episode")
    parser.add_argument("--max_training_timesteps", type=int, default=int(0.5e6),
                        help="break training loop if timeteps > max_training_timesteps")
    parser.add_argument("--record_mistakes", type=bool, default=False,
                        help="record episodes with mistakes")
    parser.add_argument("--render", type=bool, default=True,
                        help="render environment")
    parser.add_argument("--record_trajectory_length", type=int, default=20,
                        help="Record trajectory length")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of cpus")
    # Shira - New Arguments
    parser.add_argument("--masking_threshold", type=int, default=0, help="Time step at which to start using the shield")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering during simulation")
    parser.add_argument("--k_last_states", type=int, default=1, help="K last states to update the policies based on")
    args = parser.parse_args(arguments)
    # nv_name = "highway-three-v0"
    # env_name = "highway-v0"
    # env_name = "highway-fast-v0"
    # env_name = "intersection-v0"
    # env_name = "two-way-v0"
    # env_name = "HighwayEnvFastNoNormalization-v0"
    has_continuous_action_space = False
    k_last_states = args.k_last_states
    max_ep_len = args.max_ep_len  # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps  # break training loop if timeteps > max_training_timesteps
    # print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
    # log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
    print_freq = args.print_freq  # print avg reward in the interval (in num timesteps)
    log_freq = args.log_freq  # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_model_freq  # save model frequency (in num timesteps)
    masking_threshold = args.masking_threshold
    action_std = None
    no_render = args.no_render
    action_std_decay_freq = None
    action_std_decay_rate = None
    min_action_std = None
    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    update_shield_timestep = max_ep_len
    K_epochs = args.K_epochs  # update policy for K epochs [10,50,100]
    eps_clip = args.eps_clip  # clip parameter for PPO  [0.1,0.2]
    gamma = args.gamma  # discount factor [0.9,0.95  0.99]
    lr_actor = args.lr_actor  # learning rate for actor network [1e-4, 5e-4, 1e-3]
    lr_critic = args.lr_critic  # learning rate for critic network [1e-4, 5e-4, 1e-3]
    #####################################################
    agent = args.algo
    register_envs(args.envs)
    env_list = [gym.make(x) for x in args.envs]
    random_seed = args.seed
    # env_list = [gym.make("MergeEnvNoNormalization-v0"),
    #             gym.make("HighwayEnvFastNoNormalization-v0"),
    #             gym.make("IntersectionEnvNoNormalization-v0"),
    #             gym.make("RoundaboutEnvNoNormalization-v0"),
    #             gym.make("TwoWayEnvNoNormalization-v0"),
    #             gym.make("UTurnEnvNoNormalization-v0"),
    #             gym.make("MergeEnvNoNormalization-v0")
    #             ]
    multi_task_env = EnvsWrapper(env_list, has_continuous_action_space, no_render)
    if args.record_mistakes:
        for env in multi_task_env.envs:
            env.metadata["video.frames_per_second"] = 30
            env.metadata["video.output_frames_per_second"] = 30
    # state_dim = multi_task_env.observation_space.shape[0] + len(multi_task_env.envs)
    action_dim = multi_task_env.action_space_size()
    if args.no_render:
        print("Rendering is disabled")
    else:
        print("Rendering is enabled")
    #### create new log file for each run
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    #"/sise/home/wilds/ShieldPPO_SHeLL/models/20.09 - updated logs/piapua/"
    base_path = f"./models/20.09 - updated logs/piapua/{agent}_masking_@{masking_threshold}_{random_seed}_{curr_time}"
    save_model_path = f"./{base_path}/model.pth"
    save_shield_path = f"./{base_path}/shield.pth"
    save_collision_info_path = f"./{base_path}/collision_info_log.log"
    save_stats_path = f"./{base_path}/stats.log"
    save_args_path = f"./{base_path}/commandline_args.txt"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/Videos", exist_ok=True)
    # safe_rl_baselines = ['ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    with open(save_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if agent == "PPO":
        ppo_agent = PPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)
    elif agent == "ShieldPPO":
        ppo_agent = ShieldPPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                              has_continuous_action_space, action_std, masking_threshold=masking_threshold,
                              k_last_states=k_last_states)


    elif agent == 'PPOCAR':
        ppo_agent = PPOCostAsReward(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs,
                                    eps_clip,
                                    has_continuous_action_space, action_std)
    else:
        raise NotImplementedError
    time_step = 0
    i_episode = 0
    time_steps = []
    rewards = []
    tasks = []
    episodes_len = []
    collision_info = {}
    costs = []
    shield_losses = []
    stats = []
    start_time = time.time()
    amount_of_done = 0

    # training loop
    while time_step <= max_training_timesteps:
        # NEW EPOCH / EPISODE (defined by i_episode) - EACH EPISODE STARTS WITH A NEW STATE
        # print("Current time_step is ", time_step)
        state, state_vf = multi_task_env.reset()
        last_states = [state]
        last_states_vf = [state_vf]

        task_name = multi_task_env.get_current_env_name()
        valid_actions = multi_task_env.get_valid_actions()
        trajectory = collections.deque([state])
        recorder_closed = False
        # INITALIZE THE REWARDS & COSTS (CUMULATIVE)
        current_ep_reward = 0
        current_ep_cost = 0
        current_ep_len = 0
        is_mistake = False

        if args.record_mistakes:
            base_video_path = f"./{base_path}/Videos"
            os.makedirs(base_path, exist_ok=True)
            video_path = f"{base_video_path}/episode_" + str(i_episode)
            trajectory_path = f"{base_video_path}/episode_" + str(i_episode) + "_trajectory.txt"
            video_recorder = VideoRecorder(multi_task_env.env, base_path=video_path, enabled=video_path is not None)
        for t in range(1, max_ep_len + 1):
            # select action with policy
            if args.render and not args.no_render:
                multi_task_env.env.render()
            # RELEVANT
            prev_states = last_states.copy()
            prev_states_vf = last_states_vf.copy()
            action, safety_scores, no_safe_action = ppo_agent.select_action(last_states, valid_actions, time_step)

        #    print("the selected action is", action)
            state, reward, done, info, state_vf = multi_task_env.step(action)
            last_states.append(state)
            last_states_vf.append(state_vf)
            if len(last_states) > k_last_states:
                last_states = last_states[1:]
            if len(last_states_vf) > k_last_states:
                last_states_vf = last_states_vf[1:]
            trajectory.appendleft((action, state))
            if len(trajectory) > args.record_trajectory_length:
                trajectory.pop()
            trajectory_actions_after = [t[0] for t in trajectory]
            # TODO - SHOW SHAHAR (PIAPUA)
            if (len(prev_states) > 1) and (time_step >= masking_threshold) and (no_safe_action == True):
                # it means that we want to teach the network that the action that led to the PREVIOUS states is not good
                pre_action =  [t[0] for t in trajectory][1]
                # the states that led to the previous states
                prev_prev_states = prev_states[:-1]
                # padding if needed
                if len(prev_prev_states) < k_last_states:
                    padding = [np.zeros_like(prev_prev_states[0])] * (k_last_states - len(prev_prev_states))
                    padded_prev_prev_states = prev_prev_states + padding
                else: # no padding is needed
                    padded_prev_prev_states = prev_prev_states
                # define this as UNSAFE.
                ppo_agent.add_to_shield(padded_prev_prev_states, pre_action, 0)
            if args.record_mistakes:
                video_recorder.capture_frame()
            ##  adding only one reward to the buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.costs.append(info['cost'])
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward
            #    print(" += info['cost'] is",  info['cost'])
            current_ep_cost += info['cost']
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                ("update succuess")
            # the update of the shield is more frequent
            if time_step % update_shield_timestep == 0 and agent == "ShieldPPO":
                shield_loss = ppo_agent.update_shield(1024)
                shield_losses.append(shield_loss)
            # cost - the real label of the action
            if info["cost"] > 0:
                print("Collision !!!:-(")
                collision_info[time_step] = (i_episode, t, prev_states_vf, safety_scores, no_safe_action, action)
                is_mistake = True
            if agent == "ShieldPPO" or agent == "RuleBasedShieldPPO":
                # TODO - PADDING (negligble)  - ?
                if len(prev_states) < k_last_states:
                    padding = [np.zeros_like(prev_states[0])] * (k_last_states - len(prev_states))
                    padded_prev_states = prev_states + padding
                else:
                    padded_prev_states = prev_states
                if info["cost"] > 0:
                    if args.record_mistakes:
                        video_recorder.close()
                        recorder_closed = True
                        with open(trajectory_path, 'w') as f:
                            for item in reversed(trajectory):
                                f.write(f"{item}\n")
                    ppo_agent.add_to_shield(padded_prev_states, action, 0)
                else:
                    ppo_agent.add_to_shield(padded_prev_states, action, 1)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                torch.save((time_steps, rewards, costs, tasks, time.time() - start_time, episodes_len, amount_of_done), save_stats_path)

            # printing average reward
            if time_step % print_freq == 0:
                recent_reward = np.array(rewards[max(0, len(rewards) - 10):]).mean()
                recent_cost = np.array(costs[max(0, len(costs) - 10):]).mean()
                if agent == "ShieldPPO":
                    recent_shield_loss = np.array(shield_losses[max(0, len(shield_losses) - 10):]).mean()
                    print(
                        f"Episode : {i_episode:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Shield Loss {recent_shield_loss:6.2f} Time {(time.time() - start_time) / 60:.2f} min")
                else:
                    print(
                        f"Episode : {i_episode:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Time {(time.time() - start_time) / 60:.2f} min")

            # save model weights
            if time_step % save_model_freq == 0:
                ppo_agent.save(save_model_path, save_shield_path)
            # THE EPOCH FINISHES IF DONE == TRUE (REACHED TO FINAL STATE) OR REACHED TO MAX_EP_LEN
            if done:
                amount_of_done += 1
                break
        # IN THE END OF EACH EPOCH
        rewards.append(current_ep_reward)
        costs.append(current_ep_cost)
        tasks.append(task_name)
        time_steps.append(time_step)
        episodes_len.append(current_ep_len)

        if args.record_mistakes and not recorder_closed:
            video_recorder.close()
            if not is_mistake:
                os.remove(video_recorder.path)
                os.remove(video_recorder.metadata_path)
            else:
                with open(trajectory_path, 'w') as f:
                    for item in reversed(trajectory):
                        f.write("%s\n" % item)
        # i_episodes = counts the amount of episodes
        i_episode += 1
    end_time = time.time()
    total_training_time = end_time - start_time
    torch.save((time_steps, rewards, costs, tasks, total_training_time, episodes_len, amount_of_done), save_stats_path)
    multi_task_env.close()
    # store collision_dict - log of collisions
    collision_log = [
        {'Time Step': timestep, 'Episode': i_episode + 1, "Step in the Episode": t, 'Prev States': prev_states,
         'Safety Scores (Shield)': safety_scores,
         'No Safe Action': no_safe_action, 'Chosen Action': action}
        for timestep, (i_episode, t, prev_states, safety_scores, no_safe_action, action) in collision_info.items()]

    # Create a DataFrame from the list of dictionaries
    collision_log_df = pd.DataFrame(collision_log)

    # Save the DataFrame to a CSV file
    # collision_log_df.to_csv(save_collision_info_path, index=False)
    torch.save(collision_log_df, save_collision_info_path)


if __name__ == '__main__':
    train()
