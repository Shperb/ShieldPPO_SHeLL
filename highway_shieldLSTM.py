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
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.pixel_observation import PixelObservationWrapper
from highway_env.envs import HighwayEnvFast, MergeEnv
# import tensorflow as tf
# import safe_rl
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
import highway_env

import constants
# from SafetyRulesParser import SafetyRulesParser
from ppo_shieldLSTM import PPO, ShieldPPO, RuleBasedShieldPPO, PPOCostAsReward, Shield, device
from gym import spaces, register
import sys
import threading
from agent_types import ObservationType
import cv2

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
        self.state_dim = np.prod(self.env.observation_space.shape)  # low = self.env.observation_space.low
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

        # self.actions = envs[0].action_type.actions
        self.actions = {0: "Move Left", 1: "Move Right"}  # for cart pole
        self.has_continuous_action_space = has_continuous_action_space

    def observation(self, obs):
        one_hot_task = np.zeros(len(self.envs))
        one_hot_task[self.env_index] = 1
        n = np.append(obs, one_hot_task)[:-1]
        return n

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
        # mapped_action = list(self.env.action_type.actions.keys())[
        #     list(self.env.action_type.actions.values()).index(self.actions[action])]
        next_obs, reward, done, info = self.env.step(action)
        cost = self.env._cost(next_obs)
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
        self.pixel_env = PixelObservationWrapper(envs[0], pixels_only=True)
        self.size = (constants.CP_IMAGE_WIDTH, constants.CP_IMAGE_HEIGHT)
        self.pixel_env.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.size[0], self.size[1], 3), dtype=np.uint8)
        self.state_dim = np.prod(self.pixel_env.observation_space.shape)

    def step(self, action):
        try:
            action = action[0]
        except:
            pass
        next_obs, reward, done, info = self.pixel_env.step(action)
        next_state, _, _, _ = self.env.step(action)
        cost = self.env._cost(next_state)
        info['cost'] = cost
        # next_obs = next_obs['pixels']
        # return next_obs also because its shape is (number_of_vehicels, number_of_features) -> good for the log
        return self.observation(next_obs), reward, done, info, next_obs

    def reset(self):
        obs = self.pixel_env.reset()
        # obs = obs['pixels']
        return self.observation(obs), obs

    def observation(self, obs):
        pixels = obs['pixels']
        resized_pixels = cv2.resize(pixels, self.size, interpolation=cv2.INTER_AREA)
        obs['pixels'] = resized_pixels
        obs = obs['pixels']
        obs = super().observation(obs)
        return obs


def register_envs(envs):
    # create end points for the environments
    suffix = '-v0'
    for env in envs:
        entry = env[:-len(suffix)]
        register(
            id=env,
            entry_point='envs:' + entry,
        )


def train(args, env_name):
    # max_training_timestamps - the maximal number of interactions of the agent with the environment
    # max_ep_len - maximum number of steps in each episode
    """
    time step = interaction with the environment (outer loop) = one action
    episodes = sequence of interactions
    each iteration in the inner loop is a new episodes (sequence of actions) and the number of actions is limited due to max_timesteps
    trajectory - sequence of (state,action) pairs that the agent encounters during its interaction with the environment within a single episode.
    """

    if "Grayscale" in env_name or "Image" in env_name:
        obs_name = "Camera"
        obs_type = ObservationType.Camera
    else:
        obs_name = "Kinematics"
        obs_type = ObservationType.Kinematics
    has_continuous_action_space = False
    k_last_states = args.k_last_states
    safety_threshold = args.safety_threshold
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
    update_shield_timestep = update_timestep = max_ep_len * 4  # TODO: change back to max_ep_len * 4 - update policy and shield every n timesteps
    K_epochs = args.K_epochs  # update policy for K epochs [10,50,100]
    eps_clip = args.eps_clip  # clip parameter for PPO  [0.1,0.2]
    gamma = args.gamma  # discount factor [0.9,0.95  0.99]
    lr_actor = args.lr_actor  # learning rate for actor network [1e-4, 5e-4, 1e-3]
    lr_critic = args.lr_critic  # learning rate for critic network [1e-4, 5e-4, 1e-3]
    #####################################################
    envs = [env_name]
    register_envs(envs)
    env_list = [gym.make(x) for x in envs]
    # env = gym.make('CartPole-v1')
    # env = CustomPixelObservationWrapper(env)
    if 'CartPoleImage' in env_name:
        for env in env_list:
            env.reset()
        multi_task_env = PixelEnvWrapper(env_list, has_continuous_action_space, no_render)
    else:
        multi_task_env = EnvsWrapper(env_list, has_continuous_action_space, no_render)
    random_seed = args.seed
    # env_list = [gym.make("MergeEnvNoNormalization-v0"),
    #             gym.make("HighwayEnvFastNoNormalization-v0"),
    #             gym.make("IntersectionEnvNoNormalization-v0"),
    #             gym.make("RoundaboutEnvNoNormalization-v0"),
    #             gym.make("TwoWayEnvNoNormalization-v0"),
    #             gym.make("UTurnEnvNoNormalization-v0"),
    #             gym.make("MergeEnvNoNormalization-v0")
    #             ]
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
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    folder_name = "Highway_stats" if 'Highway' in env_name else "CartPole_stats"
    base_path = f"./models/{folder_name}/{obs_name}_{curr_time}"
    save_model_path = f"./{base_path}/model.pth"
    # Added another path to save the shield network (updated parameters)
    save_shield_path = f"./{base_path}/shield.pth"
    save_collision_info_path = f"./{base_path}/collision_info_log.log"
    save_stats_path = f"./{base_path}/stats.log"
    shield_loss_stats_path = f"./{base_path}/shield_loss_stats.log"
    save_args_path = f"./{base_path}/commandline_args.txt"
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/Videos", exist_ok=True)
    # safe_rl_baselines = ['ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    # Create the object for agent
    with open(save_args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    shield = Shield.get_shield(action_dim, has_continuous_action_space, folder_name)
    ppo_agent = ShieldPPO(shield, obs_type, multi_task_env.state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                          has_continuous_action_space, action_std, masking_threshold=masking_threshold,
                          k_last_states=k_last_states, safety_threshold=safety_threshold)
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
    max_training_timesteps = max_training_timesteps / 2
    # training loop
    while time_step <= max_training_timesteps:
        # NEW EPOCH / EPISODE (defined by i_episode) - EACH EPISODE STARTS WITH A NEW STATE
        # print("Current time_step is ", time_step)
        state, state_vf = multi_task_env.reset()
        last_states = [state]
        prev_states = []
        last_states_vf = [state_vf]
        task_name = multi_task_env.get_current_env_name()
        # valid_actions = multi_task_env.get_valid_actions()
        valid_actions = [0, 1]  # for cartpole
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
            # if args.render and not args.no_render:
            #     multi_task_env.env.render()
            prev_prev_states = prev_states.copy()
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
            # Error Diffusion - "no safe action according to shield"
            # in this case we assume that it happens because there are no safe actions from the given state,
            # so we want to teach the agent that the prev state and the action that led to the current state is a bad example for shield buffer.
            if len(prev_prev_states) > 0 and time_step >= args.pos_to_neg_threshold and no_safe_action and last_added_to_buffer == 1:
                ppo_agent.move_last_pos_to_neg()
            if args.record_mistakes:
                video_recorder.capture_frame()
            # Adding only one reward to the buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.costs.append(info['cost'])
            ppo_agent.buffer.is_terminals.append(done)
            # if not done and info['cost'] > 0:
            #     print("NOW")
            time_step += 1
            current_ep_reward += reward
            current_ep_cost += info['cost']
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                ("update succuess")
            # The update of the shield is more frequent
            if time_step % update_shield_timestep == 0 and type(ppo_agent) == ShieldPPO:
                shield_loss = ppo_agent.update_shield(1024)
                shield_loss_timesteps.append(time_step)
                shield_losses.append(shield_loss)
            # cost - the real label of the action
            if info["cost"] > 0:
                num_of_collisions += 1
                print(f"##### COLLISION {num_of_collisions} in {env_name} !!! :( ##### {time_step}")
                collision_info[time_step] = (i_episode, t, prev_states_vf, safety_scores, no_safe_action, action)
                is_mistake = True
            if type(ppo_agent) == ShieldPPO or type(ppo_agent) == RuleBasedShieldPPO:
                # PADDING - for less than k_last_states states the list of states will be padded with dummy states (zero)
                if len(prev_states) < k_last_states:
                    padding = [np.zeros_like(prev_states[0])] * (k_last_states - len(prev_states))
                    padded_prev_states = prev_states + padding
                else:
                    padded_prev_states = prev_states

                if info["cost"] > 0:
                    # Collision
                    if args.record_mistakes:
                        video_recorder.close()
                        recorder_closed = True
                        with open(trajectory_path, 'w') as f:
                            for item in reversed(trajectory):
                                f.write(f"{item}\n")
                    last_added_to_buffer = 0
                    # The second argument is the last informative layer index - according to the original amount of states, with no padding.
                    ppo_agent.add_to_shield(padded_prev_states, len(prev_states) - 1, action, 0, obs_type)
                else:
                    last_added_to_buffer = 1
                    ppo_agent.add_to_shield(padded_prev_states, len(prev_states) - 1, action, 1, obs_type)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                cumulative_costs = np.cumsum(np.array(costs))
                print("########################## saving stats to log ##########################")
                torch.save((time_steps, rewards, costs, tasks, time.time() - start_time, episodes_len, amount_of_done), save_stats_path)
                torch.save((shield_loss_timesteps, shield_losses), shield_loss_stats_path)

            # printing average reward
            if time_step % print_freq == 0:
                recent_reward = np.array(rewards[max(0, len(rewards) - 10):]).mean()
                recent_cost = np.array(costs[max(0, len(costs) - 10):]).mean()
                if type(ppo_agent) == ShieldPPO:
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

    # Collision_log_df - a dataframe with information regarding the collisions
    collision_log_df = pd.DataFrame(collision_log)

    # Save the DataFrame to a CSV file
    # collision_log_df.to_csv(save_collision_info_path, index=False)
    torch.save(collision_log_df, save_collision_info_path)


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
    parser.add_argument("--log_freq", type=int, default=20,
                        help="log avg reward in the interval (in num timesteps)")
    parser.add_argument("--save_model_freq", type=int, default=int(5e4),
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
    parser.add_argument("--max_ep_len", type=int, default=400,
                        help="max timesteps in one episode")
    parser.add_argument("--max_training_timesteps", type=int, default=int(1e6),
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
    parser.add_argument("--pos_to_neg_threshold", type=int, default=100000, help="Time step at which to start using the shield")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering during simulation")
    parser.add_argument("--k_last_states", type=int, default=1, help="K last states to update the policies based on")
    parser.add_argument("--safety_threshold", type=float, default=0.5, help="safety_threshold")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    threads = []

    for env in args.envs:
        threads.append(threading.Thread(target=train, args=(args, env)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()
