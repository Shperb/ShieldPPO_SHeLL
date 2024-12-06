# imports
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
from gym.wrappers import ResizeObservation
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.pixel_observation import PixelObservationWrapper
from highway_env.envs import HighwayEnvFast, MergeEnv
# import tensorflow as tf
# import safe_rl
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import highway_env

from utils import constants
import ppo_original
# from SafetyRulesParser import SafetyRulesParser
from ppo_shield import PPO, device, ShieldPPO, Shield
from ppo_shieldLSTM import RuleBasedShieldPPO
from gym import spaces, register
import sys
import threading
from encoders import ObservationType
import cv2

# from xvfbwrapper import Xvfb

# Start virtual display
# vdisplay = Xvfb()
# vdisplay.start()

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
        super().__init__(envs[0])
        self.envs = envs  # the first environment is assumed to have the full set of actions
        self.np_random = self.np_random, _ = seeding.np_random(seed)
        # self.env_index = self.np_random.randint(0, len(envs))
        self.state_dim = np.prod(self.env.observation_space.shape)  # low = self.env.observation_space.low
        self.no_render = no_render
        # if self.no_render:
        #     self.env.render_mode = 'no_render'
        # high = self.env.observation_space.high
        # dim_state = np.prod(self.env.observation_space.shape)
        # self.observation_space = spaces.Box(low=low.reshape(-1),
        #                                     high=high.reshape(-1),
        #                                     shape=(dim_state,),
        #                                     dtype=np.float32)
        # dim_state =no_render

        if 'Highway' in envs[0].spec.id:
            self.actions = envs[0].action_type.actions
        elif 'CartPole' in envs[0].spec.id:
            self.actions = {0: "Move Left", 1: "Move Right"}  # for cart pole
        else:
            self.actions = {0: "Do Nothing", 1: "Steer Left", 2: "Steer Right", 3: "Gas", 4: "Brake"}  # for Car Racing
            self.envs[0].unwrapped.continous = False
        self.has_continuous_action_space = has_continuous_action_space

    def observation(self, obs):
        one_hot_task = np.zeros(len(self.envs))
        one_hot_task[0] = 1
        n = np.append(obs, one_hot_task)[:-1]
        return n

    def reset(self):
        # self.env_index = self.np_random.randint(0, len(self.envs))
        # self.env = self.envs[self.env_index]
        # low = self.env.observation_space.low
        # high = self.env.observation_space.high
        # dim_state = np.prod(self.env.observation_space.shape)
        # self.observation_space = spaces.Box(low=low.reshape(-1),
        #                                     high=high.reshape(-1),
        #                                     shape=(dim_state,),
        #                                     dtype=np.float32)
        obs = self.env.reset()
        # return obs also because its shape is (number_of_vehicels, number_of_features) -> good for the log
        if type(obs) == tuple:
            obs = obs[0]
        return self.observation(obs)

    def step(self, action):
        try:
            action = action[0]
        except:
            pass
        # mapped_action = list(self.env.action_type.actions.keys())[
        #     list(self.env.action_type.actions.values()).index(self.actions[action])]
        # next_obs, reward, done, truncated, info = self.env.step(action)
        next_obs, reward, done, info = self.env.step(action)
        cost = self.env.cost(next_obs)
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


class PixelEnvWrapper(EnvsWrapper):
    def __init__(self, envs, has_continuous_action_space=False, seed=46456, no_render=False):
        super().__init__(envs, has_continuous_action_space, seed, no_render)
        try:
            self.pixel_env = PixelObservationWrapper(envs[0])
        except Exception as e:
            print(e)
        self.size = (constants.CP_IMAGE_WIDTH, constants.CP_IMAGE_HEIGHT)
        self.pixel_env.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.size[0], self.size[1], 3), dtype=np.uint8)
        self.state_dim = np.prod(self.pixel_env.observation_space.shape) // 3

    def step(self, action):
        try:
            action = action[0]
        except:
            pass
        next_obs, reward, done, info = self.pixel_env.step(action)
        next_state, _, _, _ = self.env.step(action)
        cost = self.env.cost(next_state)
        info['cost'] = cost
        # next_obs = next_obs['pixels']
        # return next_obs also because its shape is (number_of_vehicels, number_of_features) -> good for the log
        return self.observation(next_obs), reward, done, info, next_obs

    def reset(self):
        obs = self.pixel_env.reset()
        return self.observation(obs)

    def observation(self, obs):
        # pixels = obs['pixels']
        # resized_pixels = cv2.resize(pixels, self.size, interpolation=cv2.INTER_AREA)
        # obs['pixels'] = resized_pixels
        obs = obs['pixels']
        grayscale_array = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        obs = super().observation(grayscale_array)
        return obs


def register_envs(envs):
    # create end points for the environments
    suffix = '-v0'
    for env in envs:
        entry = env[:-len(suffix)]
        register(
            id=env,
            entry_point='envs:' + entry,
            max_episode_steps=200,  # CHANGED
        )


def train(args, env_name, env_idx):
    # max_training_timestamps - the maximal number of interactions of the agent with the environment
    # max_ep_len - maximum number of steps in each episode
    """
    time step = interaction with the environment (outer loop) = one action
    episodes = sequence of interactions
    each iteration in the inner loop is a new episodes (sequence of actions) and the number of actions is limited due to max_timesteps
    trajectory - sequence of (state,action) pairs that the agent encounters during its interaction with the environment within a single episode.
    """

    obs_name, obs_type, folder_name = get_env_obs(env_name)

    has_continuous_action_space = False
    safety_threshold = args.safety_threshold
    max_ep_len = args.max_ep_len  # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps  # break training loop if timeteps > max_training_timesteps
    # print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
    # log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
    print_freq = args.print_freq  # print avg reward in the interval (in num timesteps)
    log_freq = args.log_freq  # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_model_freq  # save model frequency (in num timesteps)
    masking_threshold = args.masking_threshold
    batch_size = args.batch_size
    action_std = None
    no_render = args.no_render
    action_std_decay_freq = None
    action_std_decay_rate = None
    min_action_std = None
    ################ PPO hyperparameters ################
    # update_shield_timestep = update_timestep = max_ep_len * 8  # TODO: change back to max_ep_len * 4 - update policy and shield every n timesteps
    update_shield_timestep = update_timestep = max_ep_len * 4  # TODO: change back to max_ep_len * 4 - update policy and shield every n timesteps
    k_epochs_ppo = args.k_epochs_ppo  # update policy for K epochs [10,50,100]
    k_epochs_shield = args.k_epochs_shield  # update Shield for K epochs [10,50,100]
    eps_clip = args.eps_clip  # clip parameter for PPO  [0.1,0.2]
    gamma = args.gamma  # discount factor [0.9,0.95  0.99]
    lr_actor = args.lr_actor  # learning rate for actor network [1e-4, 5e-4, 1e-3]
    lr_critic = args.lr_critic  # learning rate for critic network [1e-4, 5e-4, 1e-3]
    seed = args.seed
    envs = [env_name]
    register_envs(envs)
    env_list = [gym.make(x) for x in envs]
    if 'CartPoleImage' in env_name:
        for e in env_list:
            e.reset()
        multi_task_env = PixelEnvWrapper(env_list, has_continuous_action_space, no_render)
    else:
        # env_list = [env_list[0].unwrapped]
        multi_task_env = EnvsWrapper(env_list, has_continuous_action_space, seed, no_render)

    if args.record_mistakes:
        for env in multi_task_env.envs:
            env.metadata["video.frames_per_second"] = 30
            env.metadata["video.output_frames_per_second"] = 30
    action_dim = multi_task_env.action_space_size()
    if args.no_render:
        print("Rendering is disabled")
    else:
        print("Rendering is enabled")
    #### create new log file for each run
    test_name = args.test_name
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = f"./models/{folder_name}/{test_name}_{seed}_{curr_time}/{obs_name}{env_idx}_CO"
    print(base_path)
    save_model_path = f"./{base_path}/model.pth"
    # Added another path to save the shield network (updated parameters)
    log_file_size_threshold = 10 * 1024 * 1024  # 10MB in bytes
    stats_file_index = 1
    save_shield_path = f"./{base_path}/shield.pth"
    save_collision_info_path = f"./{base_path}/collision_info_log.log"
    save_stats_path = f"./{base_path}/stats{stats_file_index}.json"
    shield_loss_stats_path = f"./{base_path}/shield_loss_stats{stats_file_index}.json"
    save_args_path = f"./{base_path}/commandline_args.txt"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/Videos", exist_ok=True)
    # safe_rl_baselines = ['ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    # Create the object for agent
    with open(save_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    action_std = args.action_std  # 0.15 0.9

    # shield = Shield.get_shield(action_dim, has_continuous_action_space, folder_name)
    # ppo_agent = ShieldPPO(shield, obs_type, multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, k_epochs_shield, eps_clip,
    #                          has_continuous_action_space, action_std, masking_threshold, safety_threshold)
    # ppo_agent = ShieldPPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, k_epochs_shield, eps_clip,
    #                       has_continuous_action_space, action_std, 1000000, safety_threshold)
    ppo_agent = PPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
                    has_continuous_action_space, obs_type, action_std)
    # ppo_agent = ppo_original.ShieldPPO(multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs_ppo, eps_clip,
    #                                    has_continuous_action_space, action_std)

    time_step = 0
    i_episode = 0
    time_steps = []
    shield_loss_timesteps = []
    rewards = []
    tasks = []
    episodes_len = []
    collision_info = {}
    costs = []
    shield_losses = []
    stats = []
    start_time = time.time()
    amount_of_done = 0
    num_of_collisions = 0
    # print_running_reward = 0
    # print_running_episodes = 0
    # log_running_reward = 0
    # log_running_episodes = 0
    # training loop
    while time_step <= max_training_timesteps:
        # NEW EPOCH / EPISODE (defined by i_episode) - EACH EPISODE STARTS WITH A NEW STATE
        # print("Current time_step is ", time_step)
        state = multi_task_env.reset()
        task_name = multi_task_env.get_current_env_name()
        if 'Highway' in env_name:
            valid_actions = multi_task_env.get_valid_actions()
        elif 'CartPole' in env_name:
            valid_actions = [0, 1]  # for CartPole
        # elif 'CarRacing' in task_name:
        else:
            valid_actions = [0, 1, 2, 3, 4]  # for CarRacing
        trajectory = collections.deque([state])
        recorder_closed = False
        # INITIALIZE THE REWARDS & COSTS (CUMULATIVE)
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
            if type(ppo_agent) == ShieldPPO:
                action, no_safe_action = ppo_agent.select_action(state, valid_actions, time_step)
            else:
                action = ppo_agent.select_action(state)
            prev_state = state
            state, reward, done, info, state_vf = multi_task_env.step(action)
            trajectory.appendleft((action, state))
            if len(trajectory) > args.record_trajectory_length:
                trajectory.pop()
            # if type(ppo_agent) == ShieldPPO and t > 1 and time_step >= args.pos_to_neg_threshold and no_safe_action and last_added_to_buffer == 1:
            #     ppo_agent.move_last_pos_to_neg()
            if args.record_mistakes:
                video_recorder.capture_frame()
            # Adding only one reward to the buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.costs.append(info['cost'])
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward
            current_ep_cost += info['cost']

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # The update of the shield is more frequent
            if time_step % update_shield_timestep == 0 and type(ppo_agent) == ShieldPPO:
                shield_loss = ppo_agent.update_shield(batch_size)
                shield_loss_timesteps.append(time_step)
                shield_losses.append(shield_loss)

            if type(ppo_agent) == ShieldPPO or type(ppo_agent) == RuleBasedShieldPPO:
                if info["cost"] > 0:
                    # Collision
                    if args.record_mistakes:
                        video_recorder.close()
                        recorder_closed = True
                        with open(trajectory_path, 'w') as f:
                            for item in reversed(trajectory):
                                f.write(f"{item}\n")
                    last_added_to_buffer = 0
                    ppo_agent.add_to_shield(prev_state, action, 0, obs_type)
                else:
                    last_added_to_buffer = 1
                    ppo_agent.add_to_shield(prev_state, action, 1, obs_type)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # torch.save((time_steps, rewards, costs, time.time() - start_time, episodes_len, amount_of_done), save_stats_path)
                # torch.save((shield_loss_timesteps, shield_losses), shield_loss_stats_path)
                with open(save_stats_path, 'w') as f:
                    json.dump((time_steps, rewards, costs, episodes_len), f)
                with open(shield_loss_stats_path, 'w') as f:
                    json.dump((shield_loss_timesteps, shield_losses), f)

                if os.path.getsize(save_stats_path) >= log_file_size_threshold:
                    stats_file_index += 1
                    time_steps = []
                    shield_loss_timesteps = []
                    rewards = []
                    episodes_len = []
                    costs = []
                    shield_losses = []
                    save_stats_path = f"./{base_path}/stats{stats_file_index}.json"
                    shield_loss_stats_path = f"./{base_path}/shield_loss_stats{stats_file_index}.json"

            # printing average reward
            if time_step % int(print_freq) == 0:
                # print_avg_reward = print_running_reward / print_running_episodes
                # print_avg_reward = round(print_avg_reward, 2)

                recent_reward = np.array(rewards[max(0, len(rewards) - 10):]).mean()
                recent_cost = np.array(costs[max(0, len(costs) - 10):]).mean()
                print(f"Obs: {obs_name}  Episode: {i_episode}  Reward: {recent_reward:0.2f}  Cost: {recent_cost:0.2f}  Timestep: {time_step}, Timestamp: {datetime.now()}")
                # print(np.array(rewards).mean())
                # print(recent_reward)
                # print_running_reward = 0
                # print_running_episodes = 0

            # save model weights
            # if time_step % save_model_freq == 0:
            #     ppo_agent.save(save_model_path, save_shield_path)
            current_ep_len += 1
            if done:
                amount_of_done += 1
                break

        # IN THE END OF EACH EPOCH
        # print_running_reward += current_ep_reward
        # print_running_episodes += 1
        #
        # log_running_reward += current_ep_reward
        # log_running_episodes += 1

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
    # torch.save((time_steps, rewards, costs, tasks, total_training_time, episodes_len, amount_of_done), save_stats_path)
    with open(save_stats_path, 'w') as f:
        json.dump((time_steps, rewards, costs, episodes_len), f)
    with open(shield_loss_stats_path, 'w') as f:
        json.dump((shield_loss_timesteps, shield_losses), f)
    multi_task_env.close()
    # store collision_dict - log of collisions
    collision_log = [
        {'Time Step': timestep, 'Episode': i_episode + 1, "Step in the Episode": t, 'Prev States': prev_states, 'Chosen Action': action}
        for timestep, (i_episode, t, prev_states, action) in collision_info.items()]

    # Collision_log_df - a dataframe with information regarding the collisions
    collision_log_df = pd.DataFrame(collision_log)

    # Save the DataFrame to a CSV file
    # collision_log_df.to_csv(save_collision_info_path, index=False)
    torch.save(collision_log_df, save_collision_info_path)


def get_env_obs(env_name):
    if "Grayscale" in env_name or "Image" in env_name:
        obs_name = "Camera"
        obs_type = ObservationType.Camera
    elif "Occupancy" in env_name:
        obs_name = "Occupancy"
        obs_type = ObservationType.OccupancyGrid
    else:
        obs_name = "Kinematics"
        obs_type = ObservationType.Kinematics

    if 'Highway' in env_name:
        folder_name = "Highway_stats"
    elif 'CartPole' in env_name:
        folder_name = "CartPole_stats"
    else:
        folder_name = "CarRacing_stats"

    # env_dict = {
    #     'Highway': 'Highway_stats',
    #     'CartPole': 'CartPole_stats',
    #     'CarRacing': 'CarRacing_stats',
    # }
    # folder_name = env_dict[env_name]

    return obs_name, obs_type, folder_name


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default=["ShieldPPO", "PPO"], nargs="+",
                        help="algorithm to use:  PPO | ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--envs", default=["HighwayEnvFastNoNormalization-v0"], nargs="+",
                        help="names of the environment to train on")
    parser.add_argument("--print_freq", default=1000,
                        help="print avg reward in the interval (in num timesteps)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log_freq", type=int, default=20000,
                        help="log avg reward in the interval (in num timesteps)")
    parser.add_argument("--save_model_freq", type=int, default=int(5e4),
                        help="save model frequency (in num timesteps)")
    parser.add_argument("--k_epochs_shield", type=int, default=30,
                        help="update Shield for K epochs")
    parser.add_argument("--k_epochs_ppo", type=int, default=30,
                        help="update policy for K epochs")
    parser.add_argument("--eps_clip", type=float, default=0.3,
                        help="clip parameter for PPO")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--lr_actor", type=float, default=5e-5,
                        help="learning rate for actor network")
    parser.add_argument("--lr_critic", type=float, default=5e-5,
                        help="learning rate for critic network")
    parser.add_argument("--max_ep_len", type=int, default=400,
                        help="max timesteps in one episode")
    parser.add_argument("--max_training_timesteps", type=int, default=3000000,  # int(1e6)
                        help="break training loop if timeteps > max_training_timesteps")
    parser.add_argument("--record_mistakes", type=bool, default=False,
                        help="record episodes with mistakes")
    parser.add_argument("--render", type=bool, default=True,
                        help="render environment")
    parser.add_argument("--record_trajectory_length", type=int, default=20,
                        help="Record trajectory length")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of cpus")
    parser.add_argument("--action_std", type=float, default=0.6,
                        help="action std for PPO")
    parser.add_argument("--test_name", type=str, default="",
                        help="name for folder to save the results and model")
    # Shira - New Arguments
    parser.add_argument("--masking_threshold", type=int, default=0, help="Time step at which to start using the shield")
    parser.add_argument("--pos_to_neg_threshold", type=int, default=300000, help="Time step at which to start using the shield")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering during simulation")
    parser.add_argument("--safety_threshold", type=float, default=0.05, help="safety_threshold")  # 0.5
    parser.add_argument("--batch_size", type=float, default=1024, help="Batch size to sample from buffer while updating shield")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    threads = []

    for idx, env in enumerate(args.envs):
        threads.append(threading.Thread(target=train, args=(args, env, idx + 1)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

# vdisplay.stop()
