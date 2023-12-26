
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical



################################## set device ##################################

# set device to cpu or cuda

if(torch.cuda.is_available()):
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
        self.is_terminals = []
        self.costs = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.costs[:]


import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical



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

        # actor
        ## INPUT FOR THE ACTOR NETWORK - State dim is -> state_dim * k_last_states
        ## the actor network goal is to set the policy - map from states to actions (learns actions that maximize reward)
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        ## critic - learns the expected reward (value) - how good the state is
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
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

    def forward(self):
        raise NotImplementedError

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

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, k_last_states=1):

        self.has_continuous_action_space = has_continuous_action_space

        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, k_last_states).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, k_last_states).to(device)
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
        # Reshape rewards tensor with padding
        # Pad the rewards tensor with zeros to match k_last_states
        # Calculate the required padding size
        padding_value = 1.0
        # Reshape the rewards tensor
        # Number of rows in the desired 2D tensor
        num_rows = rewards.shape[0]

        # Reshape the rewards tensor into a 2D tensor with shape [80, 5]
        rewards = rewards.unsqueeze(1).expand(num_rows, self.k_last_states)

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
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

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

class PPOCostAsReward(PPO):

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal, cost in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), reversed(self.buffer.costs)):
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
        if label > 0.5: # safe
            self.pos_rb.append((s,a))
        else:
            self.neg_rb.append((s,a))

# before changes

    def sample(self, n):
        n_neg = min(n // 2, len(self.neg_rb))
        n_pos = n - n_neg
        pos_batch = random.sample(self.pos_rb, n_pos)
        s_pos, a_pos = map(np.stack, zip(*pos_batch))
        neg_batch = random.sample(self.neg_rb, n_neg)
        s_neg, a_neg = map(np.stack, zip(*neg_batch))
        return torch.FloatTensor(s_pos).to(device), torch.LongTensor(a_pos).to(device),\
            torch.FloatTensor(s_neg).to(device), torch.LongTensor(a_neg).to(device)



class Shield(nn.Module):

    def __init__(self, state_dim, action_dim, has_continuous_action_space, k_last_states):
        super().__init__()

        hidden_dim = 256
        self.has_continuous_action_space = has_continuous_action_space
        if not self.has_continuous_action_space:
            self.action_embedding = nn.Embedding(action_dim, action_dim)
            self.action_embedding.weight.data = torch.eye(action_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        self.loss_fn = nn.BCELoss()

    def encode_action(self, a):
        if self.has_continuous_action_space:
            return a
        else:
            return self.action_embedding(a)

    def forward(self, s, a):
        a = self.encode_action(a)
        x = torch.cat([s, a], -1)
        return self.net(x)

    def loss(self, s_pos, a_pos, s_neg, a_neg):
        # add extra dimension since s_pos is now [14, last_k_states, state_dim] -> it has an extra dimension.
        a_pos = self.encode_action(a_pos)
        a_neg = self.encode_action(a_neg)


        x_pos = torch.cat([s_pos, a_pos], dim=-1)
        x_neg = torch.cat([s_neg, a_neg],  dim=-1)
        y_pos = self.net(x_pos)
        y_neg = self.net(x_neg)
        loss = self.loss_fn(y_pos, torch.ones_like(y_pos)) + self.loss_fn(y_neg, torch.zeros_like(y_neg))
        return loss

class ShieldPPO(PPO): # currently only discrete action
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, masking_threshold = 0, k_last_states=1):
        super().__init__(state_dim, action_dim,
                         lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                         has_continuous_action_space, action_std_init,  k_last_states=1)
        self.action_dim = action_dim
        self.shield = Shield(state_dim, action_dim, has_continuous_action_space, k_last_states = k_last_states).to(device)
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr=5e-4)
        self.shield_buffer = ShieldBuffer()
        self.masking_threshold = masking_threshold
        self.k_last_states = k_last_states
        print("masking_threshold is: ", self.masking_threshold)
        print("k_last_states is", self.k_last_states)

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
            s_pos, a_pos, s_neg, a_neg = self.shield_buffer.sample(batch_size)
            self.shield_opt.zero_grad()
            # compute loss - binary cross entropy
            loss = self.shield.loss(s_pos, a_pos, s_neg, a_neg)
            #back propogation
            loss.backward()
            # updating the shield parameters using Adam optimizer
            self.shield_opt.step()
            loss_ += loss.item()
        return loss_/self.K_epochs

    def select_action(self, states, valid_actions, timestep):
        ## The function gets a list of states - not just the last one
        print("len of states is", len(states))
        valid_mask = torch.zeros(self.k_last_states, self.action_dim).to(device)
        for a in valid_actions:
            valid_mask[:, a] = 1.0

        # Pad the last_states list if its length is less than k_states
        while len(states) < self.k_last_states:
            # TODO (?) - What should be the dummy variable - 0?
            dummy_state = np.zeros_like(states[0])  # Create a dummy state (all zeros)
            states.insert(0, dummy_state)
        # Print the shape of each state in the states list
        with torch.no_grad():
            # Create one tensor of all states tensor
            # TODO LATER - maybe add more sequences, parallel
            states_tensor = torch.FloatTensor(states).to(device)
            action_probs = self.policy_old.actor(states_tensor)
            # it will have self.k_last_states rows, each with the same action indices
            actions = torch.arange(self.action_dim).repeat(self.k_last_states, 1).to(device)
            # Repeating states for each action ( 5 actions)
            state_ = states_tensor.unsqueeze(1).repeat(1, self.action_dim, 1)
            # Pad action_probs tensor to match k_last_states
            while action_probs.size(0) < self.k_last_states:
                dummy_probs = torch.zeros_like(action_probs[0])
                action_probs = torch.cat([dummy_probs.unsqueeze(0), action_probs], dim=0)
            ## TODO - Try more tesholds / according to shield loss improvement
            ## TODO -  do it with 10,000 ... 20,000...

            if timestep >= self.masking_threshold:
                print("USING SAFETY MASKING")
            #    print("state_.shape", state_.shape)
                safety = self.shield(state_, actions)
                mask = safety.gt(0.5).float()  # Shape: [k_last_states, n_actions]
                mask = mask.squeeze(-1)  # Remove the extra singleton dimension
                mask = valid_mask * mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    #masking according to the shield + valid_mask
                    action_probs_[mask == 0] = -1e10
                else:
                    # masking according to valid_mask only

                    action_probs_[valid_mask == 0] = -1e10
            else:
                print("NOT USING SAFETY MASKING")
                mask = valid_mask
                action_probs_ = action_probs.clone()
                if mask.sum().item() > 0:
                    action_probs_[mask == 0] = -1e10
                else:
                    action_probs_[valid_mask == 0] = -1e10
            action_probs_ = F.softmax(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)
        # action is a vector (action per state)
        # [3,5,3]
        # TODO - ONE ACTION STORED IN THE BUFFER
        action = dist.sample()
        action_logprob = dist2.log_prob(action)
        self.buffer.states.append(states_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        # if k_last_states is 2 so action is for example [4,1] - which predicts the next action for each one of the states. we want the last one.
        # last_action = action[-1].item()
        # returns the vector of action (the net action for all states and not just for the last one like the comment before)
        return action


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
            #print(s,a)
            #print('A mistake was detected')
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
            mask = [(1 if self.safety_rules.check_state_action_if_safe(original_state_representation, action_as_int) else 0) for action_as_int in range(self.action_dim)]
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
