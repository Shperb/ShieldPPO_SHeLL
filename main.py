# external
import ast
import random
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
from utilities.priority_queue import *

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
        entry_point=entry,
    )


def get_valid_actions(env):
    return list(range(env.action_space.n))  # returns a vector of values 0/1 which indicate which actions are valid



def get_episode_samples(episode, shield_gamma):
    episode_samples = []
    # reverse iterating through the episode - from end to beginning
    d_cost = 0
    for step in zip(reversed(episode)):
        state, action, cost, is_terminal = step[0]
        if is_terminal:
            d_cost = 0
        d_cost = cost + (shield_gamma * d_cost)
        d_cost_tensor = torch.tensor(d_cost)
        episode_samples.insert(0, (state, action, d_cost_tensor))
    return episode_samples


################################### Training ###################################

def train(arguments=None):
    print("============================================================================================")

    ########################### Argumenets ##########################

    global env, rewards

    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument("--algo", default="ShieldPPO",
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
    """
    parser.add_argument("--shield_episodes_batch_size", type=float, default=5,
                        help="The number of episdoes from shield buffer while updating Shield")
    """
    parser.add_argument("--shield_sample_batch_size", type=float, default=1024,
                        help="The number of states to sample from shield buffer while updating Shield")
    # new argument for shield buffer (prioritizied experience replay)
    parser.add_argument("--shield_buffer_size", type=int, default=5000,
                        help="maximum amount of samples in shield buffer (prioritizied experience replay buffer)")




    # Gen Arguments
    parser.add_argument("--use_gen_v2", type=bool, default=False,
                        help="If True, use generator version 2.")
    parser.add_argument("--K_epochs_gen", type=int, default=30,
                        help="update Gen for K epochs")
    parser.add_argument("--lr_gen", type=float, default=5e-5, help="Generator learning rate")
    parser.add_argument("--gen_masking_tresh", type=float, default=0,
                        help="Episode Number at which to start using the Generator, for GAN")
    parser.add_argument("--update_gen_timestep", type=float, default=500,
                        help="Update the generator network each update_gen_timestep time steps")
    parser.add_argument("--gen_batch_size", type=float, default=5,
                        help="Batch size to sample from buffer while updating generator")
    parser.add_argument("--generator_latent_dim", type=float, default=32,
                        help="The dimension of latent space (Generator)")

    # param ranges arguments - an example for range input for gravity will be like this: --gravity 1 20
    parser.add_argument('--gravity_range', type=float, nargs=2, default=[9.0, 10.0], metavar=('min', 'max'),
                        help='Gravity range (min, max)')
    parser.add_argument('--masscart_range', type=float, nargs=2, default=[1.0, 20.0], metavar=('min', 'max'),
                        help='Masscart range (min, max)')
    parser.add_argument('--masspole_range', type=float, nargs=2, default=[0.1, 2.0], metavar=('min', 'max'),
                        help='Masspole range (min, max)')
    parser.add_argument('--length_range', type=float, nargs=2, default=[0.5, 3.0], metavar=('min', 'max'),
                        help='Length range (min, max)')

    parser.add_argument('--force_mag_range', type=float, nargs=2, default=[5, 20], metavar=('min', 'max'),
                        help='force mag range (min, max)')

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
    random_seed = args.seed  # set random seed if required (0 = no random seed)

    # shield
    update_shield_timestep = args.update_shield_timestep
    # shield_episodes_batch_size = args.shield_episodes_batch_size
    K_epochs_shield = args.K_epochs_shield
    lr_shield = args.lr_shield
    shield_gamma = args.shield_gamma
    masking_threshold = args.masking_threshold
    unsafe_tresh = args.unsafe_tresh
    shield_buffer_size = args.shield_buffer_size
    shield_sample_batch_size = args.shield_sample_batch_size
    # gen
    use_gen_v2 = args.use_gen_v2
    K_epochs_gen = args.K_epochs_gen
    lr_gen = args.lr_gen
    gen_masking_tresh = args.gen_masking_tresh
    update_gen_timestep = args.update_gen_timestep
    latent_dim = args.generator_latent_dim
    gen_batch_size = args.gen_batch_size

    #####################################################

    param_ranges = {
        'gravity': {'range': (args.gravity_range[0], args.gravity_range[1]), 'type': float},
        'masscart': {'range': (args.masscart_range[0], args.masscart_range[1]), 'type': float},
        'masspole': {'range': (args.masspole_range[0], args.masspole_range[1]), 'type': float},
        'length': {'range': (args.length_range[0], args.length_range[1]), 'type': float},
        'force_mag': {'range': (args.force_mag_range[0], args.force_mag_range[1]), 'type': float}
    }

    # register modified_envs
    register_env(args.env)
    env = gym.make(args.env, param_ranges=param_ranges)
    evaluation_env =  gym.make(args.env, param_ranges=param_ranges)
    env.action_space.seed(random_seed)
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

    # TODO - add after we add the support in shield + gen
    save_gen_path = f"./{base_path}/gen.pth"

    save_stats_path = f"./{base_path}/stats.log"
    save_shield_loss_stats_path = f"./{base_path}/shield_loss_stats.log"
    save_stats_path_renv = f"./{base_path}/stats_renv.log"
    save_gen_loss_stats_path = f"./{base_path}/gen_loss_stats.log"
    # save the run arguments in a text file
    save_args_path = f"./{base_path}/commandline_args.txt"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/Videos", exist_ok=True)
    # Construct the param_ranges dictionary

    # Create agent object
    with open(save_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("random seed is set to ", random_seed)
        torch.manual_seed(random_seed)
        # CUDA seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    ################# training procedure ################

    if agent == "PPO":
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)

    elif agent == "ShieldPPO":
        ppo_agent = ShieldPPO(state_dim=state_dim, action_dim=action_dim, lr_actor=lr_actor, lr_critic=lr_critic,
                              gamma=gamma, eps_clip=eps_clip, k_epochs_ppo=K_epochs, k_epochs_shield=K_epochs_shield,
                              k_epochs_gen=K_epochs_gen,
                              has_continuous_action_space=has_continuous_action_space, lr_shield=lr_shield,
                              lr_gen=lr_gen, latent_dim=latent_dim, shield_gamma=shield_gamma,
                              action_std_init=action_std, masking_threshold=masking_threshold,
                              unsafe_tresh=unsafe_tresh, use_gen_v2 = use_gen_v2,shield_buffer_size = shield_buffer_size,  param_ranges=param_ranges)

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
    gen_losses = []
    stats = []
    amount_of_done = 0
    # Save times steps, rewards and costs for random environment , for Training Evaluation.
    amount_of_done_renv = 0
    time_steps_renv = []
    rewards_renv = []
    costs_renv = []
    shield_loss_update_stats = {}
    gen_loss_update_stats = {}
    gen_loss_update_stats_v2 = {}
    steps_before_collision = None

    # training loop
    _, _ = env.reset(seed = random_seed, param_ranges=param_ranges)
    _, _ = evaluation_env.reset(seed=random_seed+10, param_ranges=param_ranges)

    # TODO - Show Shahaf (22.6)
    use_gen_v1 = True
    if use_gen_v2:
        use_gen_v1 = False

    while time_step <= max_training_timesteps:
        if i_episode >= gen_masking_tresh:
            # using generator to get a generated configuration for env, and the first chosen action
            steps_before_collision = 0
            # TODO - Show Shahaf (22.6)
            if use_gen_v1:
                gan_output_param_dict, gan_output_unsafe_scores = ppo_agent.get_generated_env_config()
                gen_chosen_action = gan_output_unsafe_scores.index(max(gan_output_unsafe_scores))
            else:
                # gen_v2 -> doesn't choose an action.
                gan_output_param_dict = ppo_agent.get_generated_env_config()
            state, _ = env.reset(param_ranges=param_ranges, gan_output=gan_output_param_dict)
            gen_chosen_state = state
        else:
            state, _ = env.reset(param_ranges=param_ranges)
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
            if i_episode >= gen_masking_tresh:
                # An episode generated by GAN (in case of i_episode >= gen_masking_tresh)
                steps_before_collision += 1

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
            # TODO - Show Shahaf (22.6)
            if use_gen_v1:
                # in version 2 there is action prediction.
                if (i_episode >= gen_masking_tresh) and (t == 1):
                    # For the first time, the agent will choose the action chosen by generator in any case.
                    action = gen_chosen_action
            if record_mistakes:
                video_recorder.capture_frame()

            if args.render:
                env.render()

            prev_state = state
            state, reward, done, info = env.step(action)

            # return observation, reward, done, info

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
                shield_loss = ppo_agent.update_shield(shield_sample_batch_size)
                shield_losses.append(shield_loss)
                shield_loss_update_stats[time_step] = (i_episode, t, shield_loss)
                # TODO - Show Shahaf (22.6)
                if use_gen_v2 and i_episode >= gen_masking_tresh:
                    gen_loss = ppo_agent.update_gen_v2(shield_loss)
                    gen_loss_update_stats[time_step] = (i_episode,t,gen_loss)

            # updated for generator version 1
            if use_gen_v1:
                # relevant only for gen version 1
                # TODO - Show Shahaf (22.6)
                if time_step % update_gen_timestep == 0 and agent == "ShieldPPO" and i_episode >= gen_masking_tresh:
                    gen_loss = ppo_agent.update_gen(gen_batch_size)
                    gen_loss_update_stats[time_step] = (i_episode, t, gen_loss)
                    gen_losses.append(gen_loss)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Log in logging file
            if time_step % log_freq == 0:
                torch.save((time_steps, rewards, costs, tasks, datetime.now().replace(microsecond=0) - start_time,
                            episodes_len, amount_of_done, i_episode), save_stats_path)
                torch.save(shield_loss_update_stats, save_shield_loss_stats_path)
                torch.save(gen_loss_update_stats, save_gen_loss_stats_path)

            # Print average stats
            if time_step % print_freq == 0:
                recent_reward = np.array(rewards[max(0, len(rewards) - 10):]).mean()
                recent_cost = np.array(costs[max(0, len(costs) - 10):]).mean()
                if agent == "ShieldPPO":
                    recent_shield_loss = np.array(shield_losses[max(0, len(shield_losses) - 10):]).mean()
                    # recent_gen_loss = np.array(gen_losses[max(0, len(gen_losses) - 10):]).mean()
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
                    ppo_agent.save(save_model_path, save_shield_path, save_gen_path)
                    # TODO- add save_gen_path
                else:
                    ppo_agent.save(save_model_path)
            # break; if the episode is over
            if done:
                amount_of_done += 1
                break

        # IN THE END OF EACH EPOCH
        if use_gen_v1:
            # if gen version = 2 , there is no generator buffer.
            if i_episode >= gen_masking_tresh:
                # using generator - saving gen_chosen_state in list (same structure as k_last_states, as it is sent to the shield in the Gen.loss())
                if steps_before_collision == 1:
                    steps_before_collision = 0
                ppo_agent.add_to_gen(gen_chosen_state, gen_chosen_action, steps_before_collision)

        # Save episode stats (rewards, costs, task, time_steps and episodes_length)
        rewards.append(current_ep_reward)
        costs.append(current_ep_cost)
        tasks.append(args.env)
        time_steps.append(time_step)
        episodes_len.append(current_ep_len)

        if type(ppo_agent) == ShieldPPO:
            # old one - Save to shield buffer in the end of each episode a list of [(s1,a1,cost1,done1),.............] for all steps in episode
            episode_samples = get_episode_samples(shield_epoch_trajectory, shield_gamma)
            for sample in episode_samples:
                state, action, cost = sample
                error = ppo_agent.shield.loss(state.unsqueeze(0), action, cost)
                ppo_agent.add_to_shield(error, sample)

        # Initialize a random environment - for Training Evaluation.
        # Another episode - with a random env (not generated by Gan) - for Training Evaluation.
        # NOTE - the evaluation stats are for different seed (random_seed+constant)
        state_renv, _ = evaluation_env.reset(param_ranges=param_ranges)
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
                torch.save((time_steps_renv, rewards_renv, costs_renv,
                            datetime.now().replace(microsecond=0) - start_time, episodes_len_renv,
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
    torch.save((time_steps, rewards, costs, tasks, datetime.now().replace(microsecond=0) - start_time, episodes_len,
                amount_of_done, i_episode),
               save_stats_path)
    torch.save((time_steps_renv, rewards_renv, costs_renv, tasks, datetime.now().replace(microsecond=0) - start_time,
                episodes_len_renv, amount_of_done_renv, i_episode),
               save_stats_path_renv)
    torch.save(shield_loss_update_stats, save_shield_loss_stats_path)
    torch.save(gen_loss_update_stats, save_gen_loss_stats_path)
    env.close()


if __name__ == '__main__':
    train()
