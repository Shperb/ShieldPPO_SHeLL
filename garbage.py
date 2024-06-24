# run on server - before adding GAN


# external
import json
import gym
import argparse
import collections
import os
import time
from datetime import datetime
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import numpy as np
from gym import spaces, register

# internal
from ppo import ShieldPPO, PPO
from modified_envs.NoNormalizationEnvs import CartPoleWithCost




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
        # the function returns an observation
        return obs.reshape(-1)

    def reset(self):
        # returns the first observation of the env
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        # returns the next observation from the current state by taking input action, and additional info
        next_obs, reward, done, placeholder1, info = self.env.step(action)
        cost = self.env._cost(next_obs)
        info['cost'] = cost
        #  observation, reward, terminated (True/False if arrived to terminated state), False. {} -> the last 2 are placeholders for general gym env purposes
        return self.observation(next_obs), reward, done, info



def register_env(env):
    # create end points for the environments
    entry = 'modified_envs.NoNormalizationEnvs:' + env[:-len('-v0')]  # Update the entry_point
    register(
            id=env,
            entry_point= entry,
    )

def get_valid_actions(env):
    return list(range(env.action_space.n))  # returns a vector of values 0/1 which indicate which actions are valid

################################### Training ###################################

def train(arguments=None):
    print("============================================================================================")

    ########################### Argumenets ##########################
    global env, rewards

    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument("--algo", default="PPO",
                        help="algorithm to use:  PPO | ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--env", default="CartPoleWithCost-v0",
                        help="names of the environment to train on")
    parser.add_argument("--print_freq", default=1000,
                        help="print avg reward in the interval (in num timesteps)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log_freq", type=int, default=10000,
                        help="log avg reward in the interval (in num timesteps)")
    parser.add_argument("--save_model_freq", type=int, default=int(10000),
                        help="save model frequency (in num timesteps)")
    parser.add_argument("--max_ep_len", type=int, default=200,
                        help="max timesteps in one episode")
    parser.add_argument("--max_training_timesteps", type=int, default=int(1e6),
                        help="break training loop if timeteps > max_training_timesteps")
    parser.add_argument("--record_mistakes", type=bool, default=False,
                        help="record episodes with mistakes")
    parser.add_argument("--render", type=bool, default=True,
                        help="render environment")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of cpus")
    parser.add_argument("--base_path", type=str, default="models/",
                        help="base path for saving logs")


    # PPO Arguments
    parser.add_argument("--K_epochs", type=int, default=80,
                        help="update policy for K epochs")
    parser.add_argument("--eps_clip", type=float, default=0.2,
                        help="clip parameter for PPO")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--lr_actor", type=float, default=0.0003,
                        help="learning rate for actor network")
    parser.add_argument("--lr_critic", type=float, default=0.001,
                        help="learning rate for critic network")
    parser.add_argument("--record_trajectory_length", type=int, default=20,
                        help="Record trajectory length")

    # Shield Arguments
    parser.add_argument("--K_epochs_shield", type=int, default=30,
                        help="update Shield for K epochs")
    parser.add_argument("--shield_gamma", type=float, default=0.6,
                        help="discount factor for shield, while calculating loss function")
    parser.add_argument("--masking_threshold", type=int, default=0, help="Time step at which to start using the shield")
    parser.add_argument("--lr_shield", type=float, default=5e-5, help="Shield learning rate")
    parser.add_argument("--unsafe_tresh", type=float, default=0.5, help="Unsafe treshold for the Shield network")
    parser.add_argument("--update_shield_timestep", type=float, default=200,
                        help="Update the shield network each update_shield_timestep time steps")
    parser.add_argument("--shield_episodes_batch_size", type=float, default=3,
                        help="The number of episdoes from shield buffer while updating Shield")

    # Gen Arguments
    parser.add_argument("--K_epochs_gen", type=int, default=30,
                        help="update Gen for K epochs")
    parser.add_argument("--lr_gen", type=float, default=5e-5, help="Generator learning rate")
    parser.add_argument("--gen_masking_tresh", type=float, default=0,
                        help="Episode Number at which to start using the Generator, for GAN")
    parser.add_argument("--update_gen_timestep", type=float, default=500,
                        help="Update the generator network each update_gen_timestep time steps")
    parser.add_argument("--gen_batch_size", type=float, default=1024,
                        help="Batch size to sample from buffer while updating generator")
    parser.add_argument("--generator_latent_dim", type=float, default=32,
                        help="The dimension of latent space (Generator)")

    args = parser.parse_args(arguments)


    ########################### Parse Argumenets ##########################

    base_path = args.base_path
    agent = args.algo
    has_continuous_action_space = False
    max_ep_len = args.max_ep_len  # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps  # break training loop if timeteps > max_training_timesteps
    print_freq = args.print_freq  # print avg reward in the interval (in num timesteps)
    log_freq = args.log_freq  # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_model_freq  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    record_trajectory_length = args.record_trajectory_length
    record_mistakes = args.record_mistakes
    ## Note : print/log frequencies should be > than max_ep_len
    update_timestep = args.max_ep_len * 4  # update policy every n timesteps
    K_epochs = args.K_epochs  # update policy for K epochs in one PPO update
    eps_clip = args.eps_clip  # clip parameter for PPO
    gamma = args.gamma  # discount factor
    lr_actor = args.lr_actor  # learning rate for actor network
    lr_critic = args.lr_critic  # learning rate for critic network
    random_seed = 0  # set random seed if required (0 = no random seed)

    # shield
    update_shield_timestep = args.update_shield_timestep
    shield_episodes_batch_size = args.shield_episodes_batch_size
    K_epochs_shield = args.K_epochs_shield
    lr_shield = args.lr_shield
    shield_gamma = args.shield_gamma
    masking_threshold = args.masking_threshold
    unsafe_tresh = args.unsafe_tresh
    # gen
    K_epochs_gen = args.K_epochs_gen
    lr_gen = args.lr_gen
    latent_dim = args.generator_latent_dim

    #####################################################
    # register modified_envs
    register_env(args.env)

    env = gym.make(args.env)

    # state space dimension
    state_dim = np.prod(env.observation_space.shape)

    if record_mistakes:
        for env in env.envs:
            env.metadata["video.frames_per_second"] = 30
            env.metadata["video.output_frames_per_second"] = 30

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    save_model_path = f"./{base_path}/model.pth"
    save_shield_path = f"./{base_path}/shield.pth"
    save_args_path = f"./{base_path}/commandline_args.txt"


    """
    TODO - add after we add the support in shield + gen 
    save_gen_path = f"./{base_path}/gen.pth"
    """
    save_stats_path = f"./{base_path}/stats.log"
    save_shield_loss_stats_path = f"./{base_path}/shield_loss_stats.log"
    save_stats_path_renv = f"./{base_path}/stats_renv.log"


    """
    save_gen_loss_stats_path = f"./{base_path}/gen_loss_stats.log"
    """
    # save the run arguments in a text file
    save_args_path = f"./{base_path}/commandline_args.txt"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/Videos", exist_ok=True)
    param_ranges = {
        'gravity': {'range': (9.0, 10.0), 'type': float},
        'masscart': {'range': (0.5, 2), 'type': float},
        'masspole': {'range': (0.05, 0.5), 'type': float},
        'length': {'range': (0.4, 0.6), 'type': float}
        }

    # Create agent object
    with open(save_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("random seed is set to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    ################# training procedure ################

    if agent == "PPO":
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)

    elif agent == "ShieldPPO":
        ppo_agent = ShieldPPO(state_dim = state_dim, action_dim = action_dim, lr_actor = lr_actor, lr_critic = lr_critic, gamma = gamma,  eps_clip = eps_clip, k_epochs_ppo = K_epochs, k_epochs_shield = K_epochs_shield, k_epochs_gen = K_epochs_gen,
                              has_continuous_action_space = has_continuous_action_space, lr_shield = lr_shield, lr_gen = lr_gen, latent_dim = latent_dim, shield_gamma = shield_gamma, action_std_init = action_std, masking_threshold=masking_threshold,
                              unsafe_tresh = unsafe_tresh, param_ranges=param_ranges)

    else:
        print("Accepting one of the following agents as input - PPO, ShieldPPO, RuleBasedShieldPPO")
        raise NotImplementedError
    # save training start time for logs
    start_time = datetime.now().replace(microsecond=0)

    ###################### logging ######################
    # printing and logging variables
    time_step = 0
    time_step_renv = 0
    i_episode = 0
    time_steps = []
    rewards = []
    tasks = []
    episodes_len = []
    episodes_len_renv = []
    collision_info = {}
    costs = []
    shield_losses = []
    #gen_losses = []
    stats = []
    amount_of_done = 0
    # Save times steps, rewards and costs for random environment , for Training Evaluation.
    amount_of_done_renv = 0
    time_steps_renv = []
    rewards_renv = []
    costs_renv = []
    shield_loss_update_stats = {}
    #gen_loss_update_stats = {}
    steps_before_collision = None
    # training loop
    while time_step <= max_training_timesteps:
        """
                if i_episode >= gen_masking_tresh:
            # using generator to get a generated configuration for env, and the first chosen action
            steps_before_collision = 0
            param_dict, unsafe_scores = ppo_agent.get_generated_env_config()
            state, state_vf = multi_task_env.reset(param_dict)
            gen_chosen_state = state
            # The first action to apply, chosen by the generator (the action with the maximal score)
            gen_chosen_action = unsafe_scores.index(max(unsafe_scores))
        """
        state, _ = env.reset()
        valid_actions = get_valid_actions(env)
        trajectory = collections.deque([state])
        recorder_closed = False

        # INITIALIZE THE REWARDS & COSTS (CUMULATIVE)
        current_ep_reward = 0
        current_ep_cost = 0
        current_ep_len = 0

        # Save episode rewards and costs for random env episodes (renv), for Training Evaluation.
        current_ep_reward_renv = 0
        current_ep_cost_renv = 0
        current_ep_len_renv = 0

        is_mistake = False

        if record_mistakes:
            base_video_path = f"./{base_path}/Videos"
            os.makedirs(base_path, exist_ok=True)
            video_path = f"{base_video_path}/episode_" + str(i_episode)
            trajectory_path = f"{base_video_path}/episode_" + str(i_episode) + "_trajectory.txt"
            video_recorder = VideoRecorder(env, base_path=video_path, enabled=video_path is not None)
        shield_epoch_trajectory = []

        for t in range(1, max_ep_len + 1):
            """
            if i_episode >= gen_masking_tresh:
            # An episode generated by GAN (in case of i_episode >= gen_masking_tresh)
            steps_before_collision += 1
            """
            # select action with policy
            current_ep_len += 1
            if type(ppo_agent) == ShieldPPO:
                if time_step >= masking_threshold:
                    # using shield
                    action, unsafe_scores = ppo_agent.select_action(state)
                else:
                    # not using shield
                    action = ppo_agent.ppo_select_action(state)[0]
            else:
                action = ppo_agent.select_action(state)

            if record_mistakes:
                video_recorder.capture_frame()

            if args.render:
                env.render()
            prev_state = state
            state, reward, done ,info = env.step(action)

            #return observation, reward, done, info

            trajectory.appendleft((action, state))

            if len(trajectory) > record_trajectory_length:
                trajectory.pop()

            cost = info["cost"]

            # Add to ppo buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.costs.append(cost)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            # Update ppo and shield stats for the current step
            current_ep_reward += reward
            current_ep_cost += cost
            shield_epoch_trajectory.append((torch.tensor(prev_state), torch.tensor([action]), cost, done))

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # update Shield
            if time_step % update_shield_timestep == 0 and type(ppo_agent) == ShieldPPO:
                shield_loss = ppo_agent.update_shield(shield_episodes_batch_size)
                shield_losses.append(shield_loss)
                shield_loss_update_stats[time_step] = (i_episode, t, shield_loss)
            """
                        if time_step % update_gen_timestep == 0 and agent == "ShieldPPO" and i_episode >= gen_masking_tresh:
                gen_loss = ppo_agent.update_gen(batch_size)
                gen_loss_update_stats[time_step] = (i_episode, t, gen_loss)
                gen_losses.append(gen_loss)
            """


            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Log in logging file
            if time_step % log_freq == 0:
              torch.save((time_steps, rewards, costs, tasks, datetime.now().replace(microsecond=0) - start_time, episodes_len, amount_of_done, i_episode), save_stats_path)
              torch.save(shield_loss_update_stats, save_shield_loss_stats_path)
              #torch.save(gen_loss_update_stats, save_gen_loss_stats_path)

            # Print average stats
            if time_step % print_freq == 0:
                recent_reward = np.array(rewards[max(0, len(rewards) - 10):]).mean()
                recent_cost = np.array(costs[max(0, len(costs) - 10):]).mean()
                if agent == "ShieldPPO":
                    recent_shield_loss = np.array(shield_losses[max(0, len(shield_losses) - 10):]).mean()
                    #recent_gen_loss = np.array(gen_losses[max(0, len(gen_losses) - 10):]).mean()
                    print("Time Step is", time_step)
                    print(
                        f"Episode : {i_episode:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f} Shield Loss {recent_shield_loss:6.2f}")
                else:
                    print(
                        f"Episode : {i_episode:4d} Reward {recent_reward:6.2f} Cost {recent_cost:6.2f}")

            # Save model weights
            if time_step % save_model_freq == 0:
                # ppo_agent.save(save_model_path, save_shield_path, save_gen_path)
                if type(ppo_agent) == ShieldPPO:
                    ppo_agent.save(save_model_path, save_shield_path)
                    # TODO- add save_gen_path
                else:
                    ppo_agent.save(save_model_path)
            # break; if the episode is over
            if done:
                amount_of_done += 1
                break
        """
                # IN THE END OF EACH EPOCH
        if i_episode >= gen_masking_tresh:
            # In case of using generator - saving gen_chosen_state in list (same structure as k_last_states, as it is sent to the shield in the Gen.loss())
            if steps_before_collision == 1:
                steps_before_collision = 0
            ppo_agent.add_to_gen(gen_chosen_state, gen_chosen_action, steps_before_collision)
        """
        # Save episode stats (rewards, costs, task, time_steps and episodes_length)
        rewards.append(current_ep_reward)
        costs.append(current_ep_cost)
        tasks.append(args.env)
        time_steps.append(time_step)
        episodes_len.append(current_ep_len)

        if type(ppo_agent) == ShieldPPO:
            # Save to shield buffer in the end of each episode a list of [(s1,a1,cost1,done1),.............] for all steps in episode
            ppo_agent.add_to_shield(shield_epoch_trajectory)


        # Initialize a random environment - for Training Evaluation.
        # Another episode - with a random env (not generated by Gan) - for Training Evaluation.
        state_renv, _ = env.reset()
        for t_renv in range(1, max_ep_len + 1):
            if args.render:
                env.render()
            prev_state_renv = state_renv.copy()
            if type(ppo_agent) == ShieldPPO:
                if time_step_renv >= masking_threshold:
                    action_renv, unsafe_scores_renv = ppo_agent.select_action(state_renv, evaluation=True)
                else:
                    action_renv = ppo_agent.ppo_select_action(state_renv, evaluation=True)[0]
            else:
                # Sending Evaluation = True so it will not save the rewards/ etc in the ppo_agent
                action_renv = ppo_agent.select_action(state_renv, evaluation=True)

            state_renv, reward_renv, done_renv, info_renv = env.step(action_renv)
            current_ep_reward_renv += reward_renv
            current_ep_cost_renv += info_renv['cost']
            # Log in logging file
            if time_step_renv % log_freq == 0:
                torch.save((time_steps_renv, rewards_renv, costs_renv, datetime.now().replace(microsecond=0) - start_time, episodes_len_renv,
                            amount_of_done_renv, i_episode), save_stats_path_renv)
            time_step_renv += 1
            if done_renv:
                amount_of_done_renv += 1
                break

        rewards_renv.append(current_ep_reward_renv)
        costs_renv.append(current_ep_cost_renv)
        time_steps_renv.append(time_step_renv)
        episodes_len_renv.append(current_ep_len_renv)

        i_episode += 1
        # save for last episode anyway.
    torch.save((time_steps, rewards, costs, tasks,  datetime.now().replace(microsecond=0) - start_time, episodes_len, amount_of_done, i_episode),
               save_stats_path)
    torch.save(shield_loss_update_stats, save_shield_loss_stats_path)
    #torch.save(gen_loss_update_stats, save_gen_loss_stats_path)

    env.close()

if __name__ == '__main__':
    train()

#PPO


# external
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
        self.loss_fn = nn.BCELoss()

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
        # compute BCE loss
        loss = self.loss_fn(y.to(device), cost.to(device))
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

    def add_to_shield(self, epoch_trajectory):
        # epoch trajectory is a list that looks like this = [(s1,a1,label1,done1), (s2,a2,label2,done2), ...., sn,an,labeln,donen)] while n is the amount of steps in this episode
        self.shield_buffer.add(epoch_trajectory)

    """
    def add_to_gen(self, s, a, label):
    self.gen_buffer.add(s, a, label)
    """

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

    def save(self, checkpoint_path_ac, checkpoint_path_shield):

        #    def save(self, checkpoint_path_ac, checkpoint_path_shield, checkpoint_path_gen):
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


