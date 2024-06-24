# external

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import random

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        # shira changes - adding costs
        self.costs = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
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
        state_val = self.critic(state)
        # TODO _ shira: return also action_probs.
        return action_probs, action.detach(), action_logprob.detach(), state_val.detach()

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
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
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

    def select_action(self, state, evaluation=False):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            if not evaluation:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
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
        for _ in range(self.K_epochs):
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
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

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

    def add(self, s, a, label):
        if label > 0.5:  # safe
            self.pos_rb.append((s, a))
        else:
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
    def __init__(self, state_dim, action_dim, has_continuous_action_space):
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
        a_pos = self.encode_action(a_pos)
        a_neg = self.encode_action(a_neg)
        x_pos = torch.cat([s_pos, a_pos], -1)
        x_neg = torch.cat([s_neg, a_neg], -1)
        y_pos = self.net(x_pos)
        y_neg = self.net(x_neg)
        loss = self.loss_fn(y_pos, torch.ones_like(y_pos)) + self.loss_fn(y_neg, torch.zeros_like(y_neg))
        return loss


class ShieldPPO(PPO):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, eps_clip, k_epochs_ppo, k_epochs_shield,
                 k_epochs_gen,
                 has_continuous_action_space, lr_shield, lr_gen, latent_dim, shield_gamma, action_std_init,
                 masking_threshold, unsafe_tresh, param_ranges=None):
        super().__init__(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
                         has_continuous_action_space, action_std_init)
        self.action_dim = action_dim
        self.shield = Shield(state_dim, action_dim, has_continuous_action_space).to(device)
        # self.gen = Generator(action_dim = self.action_dim, gamma = gamma, latent_dim = latent_dim, param_ranges = param_ranges).to(device)
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr=lr_shield)
        # self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr= lr_gen)
        self.shield_buffer = ShieldBuffer()
        # self.gen_buffer = GeneratorBuffer()
        self.masking_threshold = masking_threshold
        self.state_dim = state_dim
        self.unsafe_treshold = unsafe_tresh
        self.k_epochs_shield = k_epochs_shield
        self.k_epochs_gen = k_epochs_gen
        self.k_epochs_ppo = k_epochs_ppo
        self.shield_gamma = shield_gamma

    def add_to_shield(self, s, a, label):
        self.shield_buffer.add(s, a, label)

    """
    def add_to_gen(self, s, a, label):
    self.gen_buffer.add(s, a, label)
    """

    def update_shield(self, batch_size):
        if len(self.shield_buffer.neg_rb) == 0:
            return 0.
        if len(self.shield_buffer) <= batch_size:
            batch_size = len(self.shield_buffer)
        loss_ = 0.
        for i in range(self.K_epochs):
            s_pos, a_pos, s_neg, a_neg = self.shield_buffer.sample(batch_size)
            self.shield_opt.zero_grad()
            loss = self.shield.loss(s_pos, a_pos, s_neg, a_neg)
            loss.backward()
            self.shield_opt.step()
            loss_ += loss.item()
        return loss_ / self.K_epochs

        """
        def update_gen(self, batch_size):
        if len(self.gen_buffer) == 0:
            return 0

        if len(self.gen_buffer) <= batch_size:
            batch_size = len(self.gen_buffer)
        loss_ = 0.

        # K steps to update the generator network - each epoch (step) is one forward and backward pass
        for i in range(self.k_epochs_gen):
            # Sampling batch_size samples from gen buffer
            batch_states, batch_actions, batch_steps_before_collisions = self.gen_buffer.sample(batch_size)
            self.gen_opt.zero_grad()
            # compute loss - binary cross entropy
            loss = self.gen.loss(self.shield, batch_states, batch_actions, batch_steps_before_collisions)
            # back propagation
            loss.backward()
            # updating the shield parameters using Adam optimizer
            self.gen_opt.step()
            loss_ += loss.item()

        return loss_ / self.k_epochs_gen
    def get_generated_env_config(self):
        return self.gen()
        """

    def ppo_select_action(self, state, evaluation=False):
        # TODO - LATER COMBINE IT WITH THE SELECT_ACTION FUNCTION BECAUSE IT'S WEIRD.
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                _, action, action_logprob, state_val = self.policy_old.act(state)
            if not evaluation:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                _, action, action_logprob, state_val = self.policy_old.act(state)
            if not evaluation:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
            return action.item(),

    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(deviֻce)
            action_probs, _, _, state_value= self.policy_old.act(state)
            actions = torch.arange(self.action_dim).to(device)  # (n_action,)
            state_ = state.view(1, -1).repeat(self.action_dim, 1)  # (n_action, state_dim)
            safety = self.shield(state_, actions)  # (nֻ_action)
            mask = safety.gt(0.5).float().view(-1)
            mask = mask

            action_probs_ = action_probs.clone()
            if mask.sum().item() > 0:
                action_probs_[mask == 0] = -1e10

            action_probs_ = F.softmax(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)

        # found_valid_action = False
        # while not found_valid_action:€ֻ
        #    action = dist2.sample()
        #    if action in valid_actions:
        #        found_valid_action = True

        action = dist.sample()
        action_logprob = dist2.log_prob(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_value)
        return action.item()

    """
    def select_action(self, state, evaluation=False):
        # create valid mask full of zeros
        no_safe_action = False

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            # compute scores per action according to PPO algorithm
            action_probs, ppo_selected_action, ppo_action_logprob, state_val = self.policy_old.act(state)

            actions = torch.arange(self.action_dim).to(device)  # (n_action,)
            state_ = state.view(1, -1).repeat(self.action_dim, 1)  # (n_action, state_dim)
            # Using unsafe Masking
            # compute unsafe scores for each (s,a)
            unsafe_score = self.shield(state_, actions)

            shield_mask = unsafe_score.lt(self.unsafe_treshold).float().view(-1).to(device)
            # multiply valid_mask with mask to get the final map
            action_probs_ = action_probs.clone()

            if shield_mask.sum().item() > 0:
                # AT LEAST ONE ACTION IS CONSIDERED SAFE
                # actions with mask = 0 are considered not safe, so giving it -inf score.
                action_probs_[shield_mask == 0] = -1e10

            action_probs_ = F.softmax(action_probs_)
            dist = Categorical(action_probs_)
            dist2 = Categorical(action_probs)
        # action - [k_last_states, action_dim] -
        action = dist.sample()
        action_logprob = dist2.log_prob(action)

        if not evaluation:
            # TODO - CHECK THIS WITH SHAHAF
            self.buffer.states.append(state)
            self.buffer.actions.append(ppo_selected_action)
            self.buffer.logprobs.append(ppo_action_logprob)
            self.buffer.state_values.append(state_val)

        return action.item(), unsafe_score
         """

    def save(self, checkpoint_path_ac, checkpoint_path_shield, checkpoint_path_gen):
        # save actor critic networks
        torch.save(self.policy_old.state_dict(), checkpoint_path_ac)
        # save shield network
        torch.save(self.shield.state_dict(), checkpoint_path_shield)
        # save gen network
        # torch.save(self.gen.state_dict(), checkpoint_path_gen)

    def load(self, checkpoint_path_ac, checkpoint_path_shield, checkpoint_path_gen):
        # Load the models - Shield, policy, old_policy, Gen
        self.policy_old.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.shield.load_state_dict(torch.load(checkpoint_path_shield, map_location=lambda storage, loc: storage))
        # self.gen.load_state_dict(torch.load(checkpoint_path_gen, map_location=lambda storage, loc: storage))

