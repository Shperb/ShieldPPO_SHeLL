# external
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import random
import numpy as np

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
        # epoch_trajectories -
        self.epoch_trajectories = []

    def add(self, epoch_trajectory):
        self.epoch_trajectories.append(epoch_trajectory)

    def __len__(self):
        return len(self.epoch_trajectories)

    def sample(self, n):
        # n - amount of epochs to sample
        # Exclude last epoch because we don't want a partial one
        if n > len(self.epoch_trajectories[:-1]):
            n = len(self.epoch_trajectories[:-1])
        trajectories_to_sample = self.epoch_trajectories[:-1]
        epochs_batch = random.sample(trajectories_to_sample, n)
        return epochs_batch


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
        self.loss_fn = nn.MSELoss()

    def encode_action(self, a):
        # returns one hot vector with the given action - for example (0,1) for action 2
        if self.has_continuous_action_space:
            return a
        else:
            one_hot = torch.zeros(a.size(0), 2).to(device)
            one_hot.scatter_(1, a.long().unsqueeze(1), 1)
            return one_hot

    def forward(self, s, a):
        # pass (state,action)) in the network. returns the unsafe score.
        a = self.encode_action(a)
        x = torch.cat([s, a], -1)
        return self.net(x)

    def loss(self, state, action, cost):
        # encode the given action
        encoded_action = self.encode_action(action.to(device))
        # pass (s,a) in the network
        x = torch.cat([state, encoded_action], -1)
        x = x.float()
        y = self.net(x)
        # cost is the real label
        cost = cost.view(-1, 1)
        # compute MSE loss
        loss = self.loss_fn(y.to(device), cost.to(device))
        return loss


class GeneratorBuffer:
    def __init__(self, n_max=1000000):
        self.n_max = n_max
        # PFL - LATER: Maybe here we will save the original prediction of the shield in the buffers / use mutual buffer for shield and generator
        # collision_rb - a buffer that saves tuples of (s, last_informative_layer, a) -> in case there was a collision (the label is positive)
        self.buffer = []

    def __len__(self):
        # returns the length of both collision and no collision buffers
        return len(self.buffer)

    def add(self, state,  action , steps_before_collisions):
        """
        The generator buffer saves (s,a, label) each episode:
        s is the initial state generated by the generator
        a is the initial action generated by the generator
        label determined wether there was a collision throughout the epoch
        """
        # cost = if there was a collision or not.
        self.buffer.append((state, action, steps_before_collisions))

    def sample(self, n):
        batch_elements = random.sample(self.buffer, n)
        batch_states = np.array([item[0] for item in batch_elements])
        batch_actions = np.array([item[1] for item in batch_elements])
        batch_steps_before_collisions = np.array([item[2] for item in batch_elements])
        return torch.FloatTensor(batch_states).to(device), torch.LongTensor(batch_actions).to(device), \
               torch.FloatTensor(batch_steps_before_collisions).to(device)


class Generator(nn.Module):
    def __init__(self, action_dim, gamma, latent_dim, param_ranges = None):
        """
        output_dim - set according to the number of parameters values to be predicted
        latent_dim - to sample from
        param_ranges - limitation for the parameters values
        """
        super(Generator, self).__init__()
        # PFL - latent_dim would be controlled by HYPERPARAMETER - make it a paramater
        self.latent_dim = latent_dim
        self.output_dim = len(param_ranges)
        self.params = list(param_ranges.keys())
        self.param_ranges = param_ranges
        self.loss_fn = nn.MSELoss()
        self.action_dim = action_dim
        self.gamma = gamma
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # output shape - output_dim (same as the amount of parameters to generate) + action_dim (safety score per each action)
            nn.Linear(64, self.output_dim + self.action_dim)
        )

    def forward(self):
        noise = torch.randn(1, self.latent_dim).to(device)
        gen_output = self.model(noise)
        # Separate the output into parameters and safety scores
        param_output = gen_output[:, :self.output_dim]
        # Each head of safety_score_output represents a safety score value for a specific action
        safety_score_output = gen_output[:, self.output_dim:]
        normalized_params = []
        # Normalize the output according to given ranges in self.speed_limit
        # Each 'head' of the output dim represents a generated parameter value
        for i in range(param_output.size(1)):
            param_range = self.param_ranges[self.params[i]]['range']
            param_type =  self.param_ranges[self.params[i]]['type']
            param_output_normalized = F.sigmoid(gen_output[:, i]) * (param_range[1] - param_range[0]) + param_range[0]
            if param_type == int:
                param_output_normalized =  param_output_normalized.int()
            normalized_params.append(param_output_normalized.item())
        safety_scores = F.softmax(safety_score_output, dim = 1)
        param_dict = dict(zip(self.params, normalized_params))
        return param_dict, safety_scores.squeeze().tolist() # {'p1': 30, .. }, [s1, s2, .., s5]


    def loss(self, shield, states , actions , steps_before_collisions):
        # TODD -  last_informative layer is len(s_collision) or len(s_no_collision) because there is no padding - only 1 state
        # last_informative_layer_repeated = torch.Tensor([len(states)-1] * self.action_dim).to(torch.int)
        y_shield = shield.forward(states, actions)
        # Loss explaination - if there was a collision the label is 1 (becuase we want it to predict it as safe), and vice versa
        # The gamma is the same value as the shield gamma
        labels = (self.gamma) ** (steps_before_collisions).view(-1,1)
        #
        loss = -1 * self.loss_fn(y_shield, labels)
        return loss


class Generator_v2(nn.Module):
    # TODO - Show Shahaf (22.06) - removed the support in safety_scores and the action return.
    def __init__(self, action_dim, gamma, latent_dim, param_ranges = None):
        """
        output_dim - set according to the number of parameters values to be predicted
        latent_dim - to sample from
        param_ranges - limitation for the parameters values
        """
        super(Generator_v2, self).__init__()
        # PFL - latent_dim would be controlled by HYPERPARAMETER - make it a paramater
        self.latent_dim = latent_dim
        self.output_dim = len(param_ranges)
        self.params = list(param_ranges.keys())
        self.param_ranges = param_ranges
        self.loss_fn = nn.MSELoss()
        self.action_dim = action_dim
        self.gamma = gamma
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # output shape - output_dim (same as the amount of parameters to generate) + action_dim (safety score per each action)
            # TODO - Show Shahaf (22.06)
            nn.Linear(64, self.output_dim)
        )

    def forward(self):
        noise = torch.randn(1, self.latent_dim).to(device)
        gen_output = self.model(noise)
        # Separate the output into parameters and safety scores

        normalized_params = []
        # Normalize the output according to given ranges in self.speed_limit
        # Each 'head' of the output dim represents a generated parameter value
        for i in range(gen_output.size(1)):
            param_range = self.param_ranges[self.params[i]]['range']
            param_type =  self.param_ranges[self.params[i]]['type']
            gen_output_normalized = F.sigmoid(gen_output[:, i]) * (param_range[1] - param_range[0]) + param_range[0]
            if param_type == int:
                gen_output_normalized = gen_output_normalized.int()
            normalized_params.append(gen_output_normalized.item())
        param_dict = dict(zip(self.params, normalized_params))
        return param_dict #{'p1': 30, .. }, [s1, s2, .., s5]


    def loss(self,shield_avg_episode_loss):
        """shield_avg_episodes_loss: the average predictions of the shield across all predictions in the episode"""
        loss_value = -1 * shield_avg_episode_loss
        return torch.tensor(loss_value, requires_grad=True)

class ShieldPPO(PPO):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, eps_clip, k_epochs_ppo, k_epochs_shield,
                 k_epochs_gen,
                 has_continuous_action_space, lr_shield, lr_gen, latent_dim, shield_gamma, action_std_init,
                 masking_threshold, unsafe_tresh,  use_gen_v2, param_ranges=None):
        super().__init__(state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
                         has_continuous_action_space, action_std_init)
        self.use_gen_v2 = use_gen_v2
        self.action_dim = action_dim
        self.shield = Shield(state_dim, action_dim, has_continuous_action_space).to(device)
        if self.use_gen_v2:
            self.gen = Generator_v2(action_dim=self.action_dim, gamma=shield_gamma, latent_dim=latent_dim,
                                 param_ranges=param_ranges).to(device)
        else:
            self.gen = Generator(action_dim=self.action_dim, gamma=shield_gamma, latent_dim=latent_dim,
                                 param_ranges=param_ranges).to(device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr= lr_gen)
        self.shield_opt = torch.optim.Adam(self.shield.parameters(), lr=lr_shield)
        self.shield_buffer = ShieldBuffer()
        self.gen_buffer = GeneratorBuffer()
        self.masking_threshold = masking_threshold
        self.state_dim = state_dim
        self.unsafe_treshold = unsafe_tresh
        self.k_epochs_shield = k_epochs_shield
        self.k_epochs_gen = k_epochs_gen
        self.k_epochs_ppo = k_epochs_ppo
        self.shield_gamma = shield_gamma

    def add_to_shield(self, epoch_trajectory):
        # epoch trajectory is a list that looks like this = [(s1,a1,label1,done1), (s2,a2,label2,done2), ...., sn,an,labeln,donen)] while n is the amount of steps in this episode
        self.shield_buffer.add(epoch_trajectory)


    def add_to_gen(self, s, a, label):
        # relevant only for generator v1
        self.gen_buffer.add(s, a, label)


    def update_shield(self, shield_episodes_batch_size):
        if len(self.shield_buffer) == 0:
            return 0
        # -1 because we can't sample the last episode
        if len(self.shield_buffer) <= shield_episodes_batch_size:
            shield_episodes_batch_size = len(self.shield_buffer)
        # Sampling shield_episodes_batch_size episodes from Shield buffer
        episodes_batch = self.shield_buffer.sample(shield_episodes_batch_size)
        # Save average loss across all the episodes in batch (length will be shield_episodes_batch_size)
        episodes_batch_loss = []
        for episode_traj in episodes_batch:
            # for each sampled episode
            discounted_cost = 0
            costs = []
            # reverse iterating through the episode - from end to beginning
            for step in zip(reversed(episode_traj)):
                state, action, cost, is_terminal = step[0]
                if is_terminal:
                    discounted_cost = 0
                discounted_cost = cost + (self.shield_gamma * discounted_cost)
                costs.insert(0, discounted_cost)
            # no need to normalize
            states_ = [step[0] for step in episode_traj]
            actions_ = [step[1] for step in episode_traj]

            states = torch.squeeze(torch.stack(states_, dim=0)).detach().to(device)
            actions = torch.squeeze(torch.stack(actions_, dim=0)).detach().to(device)

            episode_loss = 0.
            for _ in range(self.k_epochs_shield):
                self.shield_opt.zero_grad()
                loss = self.shield.loss(states, actions, torch.tensor(costs))
                loss.backward()
                self.shield_opt.step()
                episode_loss += loss.item()
            average_episode_loss = episode_loss / self.k_epochs_shield
            episodes_batch_loss.append(average_episode_loss)
        # return loss average across all the instances in the batch
        average_loss_across_all_batch = sum(episodes_batch_loss) / shield_episodes_batch_size
        return average_loss_across_all_batch

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

    def update_gen_v2(self, shield_loss):
        # K steps to update the generator network - each epoch (step) is one forward and backward pass
        k_losses = []
        for i in range(self.k_epochs_gen):
            # compute loss - binary cross entropy
            loss = self.gen.loss(shield_loss)
            # back propagation
            loss.backward()
            # updating the shield parameters using Adam optimizer
            self.gen_opt.step()
            k_losses.append(loss.item())
        return sum(k_losses) / self.k_epochs_gen


    def get_generated_env_config(self):
        return self.gen()


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

    def select_action(self, state, evaluation=False):
        # TODO - LATER COMBINE IT WITH THE SELECT_ACTION FUNCTION BECAUSE IT'S WEIRD.
        if self.has_continuous_action_space:
            # continuous action space - does not support the use of shield.
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
            # discrete action space - supports the use of shield.
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action_probs, _, _, state_val = self.policy_old.act(state)
                actions = torch.arange(self.action_dim).to(device)  # (n_action,)
                state_ = state.view(1, -1).repeat(self.action_dim, 1)  # (n_action, state_dim)
                # compute unsafe scores for each (s,a)
                unsafe_score = self.shield(state_, actions)
                shield_mask = unsafe_score.lt(self.unsafe_treshold).float().view(-1).to(device)

                action_probs_ = action_probs.clone()
                amount_of_safe_actions = shield_mask.sum().item()

                if amount_of_safe_actions > 0:
                    # AT LEAST ONE ACTION IS CONSIDERED SAFE
                    action_probs_[shield_mask == 0] = 0
                # normalize action probs: if all / none actions r safe - it will keep the original action probabilities.
                action_probs_ = action_probs_ / sum(action_probs_)
                dist = Categorical(action_probs_)
                # according to PPO action probabilities
                dist2 = Categorical(action_probs)
                action = dist.sample()
                action_logprob = dist2.log_prob(action)
            if not evaluation:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)
            return action.item(), unsafe_score

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

        #    def save(self, checkpoint_path_ac, checkpoint_path_shield, checkpoint_path_gen):
        # save actor critic networks
        torch.save(self.policy_old.state_dict(), checkpoint_path_ac)
        # save shield network
        torch.save(self.shield.state_dict(), checkpoint_path_shield)
        # save gen network
        torch.save(self.gen.state_dict(), checkpoint_path_gen)

    def load(self, checkpoint_path_ac, checkpoint_path_shield, checkpoint_path_gen):
        # Load the models - Shield, policy, old_policy, Gen
        self.policy_old.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path_ac, map_location=lambda storage, loc: storage))
        self.shield.load_state_dict(torch.load(checkpoint_path_shield, map_location=lambda storage, loc: storage))
        self.gen.load_state_dict(torch.load(checkpoint_path_gen, map_location=lambda storage, loc: storage))

