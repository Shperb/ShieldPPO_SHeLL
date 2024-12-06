import pickle
import time
import torch
import torch.nn as nn
import gymnasium as gym
import argparse
import numpy as np
import pandas as pd
import os
import random
from ppo_shield import Shield
from gymnasium import spaces, register
from ppo import PPO

# import highway_env


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

torch.backends.cudnn.benchmark = True


# create end points for the environments
def register_env(env):
    entry = 'envs.modified_envs:' + env[:-len('-v0')]  # Update the entry_point
    register(
        id=env,
        entry_point=entry,
    )


def get_valid_actions(env):
    return list(range(env.action_space.n))  # returns a vector of values 0/1 which indicate which actions are valid


def get_discounted_costs_episode_samples(episode, discount_factor_cost):
    """
    given episode and discount_factor_costs, returns the episode samples with discounted cost
    """
    episode_samples = []
    # reverse iterating through the episode - from end to beginning
    d_cost = 0
    for step in zip(reversed(episode)):
        state, action, cost, is_terminal = step[0]
        if is_terminal:
            d_cost = 0
        d_cost = cost + (discount_factor_cost * d_cost)
        d_cost_tensor = torch.tensor(d_cost)
        episode_samples.insert(0, (state, action, d_cost_tensor))
    return episode_samples


def train(arguments=None):
    # parse arguments
    parser = argparse.ArgumentParser()

    # General training arguments
    # TBD - add GenShieldPPO (however we'll name it)
    parser.add_argument("--select_action_algo", default="PPO",
                        help="algorithm to use when selecting action:  PPO | Random | ShieldPPO | RuleBasedShieldPPO (REQUIRED)")
    parser.add_argument("--env", default="CartPoleWithCost-v0",
                        help="names of the environment to train on")
    parser.add_argument("--max_ep_len", type=int, default=200,
                        help="max timesteps in one episode. In cartpole v0 it's 200.")
    parser.add_argument("--max_training_timesteps", type=int, default=int(1e6),
                        help="break training loop after 'max_training_timesteps' timesteps.")

    # running setups

    parser.add_argument("--render", type=bool, default=False,
                        help="render environment. default is False.")
    parser.add_argument("--cpu", type=int, default=4,
                        help="Number of cpus")
    parser.add_argument("--seed", type=int, default=0,
                        help="defines random seed. default is 0.")
    parser.add_argument("--highway_observation_type", type=str, default="Kinematics",
                        choices=["Kinematics", "OccupancyGrid", "TimeToCollision"],
                        help="Defines the observation type of the highway environment.")

    # logs & frequencies
    parser.add_argument("--log_freq", type=int, default=5,
                        help="save logs every log_freq episodes. default is 5.")
    parser.add_argument("--save_model_freq", type=int, default=int(5000),
                        help="save trained model every save_model_freq timestpes. default is 5000.")
    parser.add_argument("--base_path", type=str, default="models/",
                        help="base path for saving logs and more")
    parser.add_argument("--record_trajectory_length", type=int, default=200,
                        help="Record trajectory length")
    parser.add_argument("--save_buffer_pickle_n", type=float, default=5000,
                        help="save buffer pickle after n steps")

    # PPO arguments
    parser.add_argument("--ppo_K_epochs", type=int, default=80,
                        help="update policy for K epochs")
    parser.add_argument("--ppo_eps_clip", type=float, default=0.2,
                        help="clip parameter for PPO")
    parser.add_argument("--ppo_gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--ppo_lr_actor", type=float, default=0.0003,
                        help="learning rate for actor network")
    parser.add_argument("--ppo_lr_critic", type=float, default=0.001,
                        help="learning rate for critic network")

    # Shield arguments
    parser.add_argument("--shield_K_epochs", type=int, default=30,
                        help="update Shield for K epochs")
    parser.add_argument("--shield_discount_factor_cost", type=float, default=0.6,
                        help="discount factor for shield, while calculating loss function")
    parser.add_argument("--shield_unsafe_tresh", type=float, default=0.5, help="Unsafe treshold for the Shield network")
    parser.add_argument("--shield_update_timestep", type=float, default=50,
                        help="Update the shield network each 'shield_update_episode' time steps")
    parser.add_argument("--shield_batch_size", type=int, default=50000,
                        help="The number of states to sample from shield buffer while updating Shield")
    parser.add_argument("--shield_buffer_size", type=int, default=int(1e6),
                        help="maximum amount of samples in shield buffer (prioritizied experience replay buffer)")
    parser.add_argument("--shield_lr", type=float, default=1e-3,
                        help="shield optimizer learning rate")
    parser.add_argument("--shield_minimum_buffer_samples", type=int, default=50000,
                        help="start training the Shield only when the buffer has minimum 'minimum_buffer_samples' samples")
    parser.add_argument("--shield_sub_batch_size", type=int, default=500,
                        help="Shield sub batch size")

    # Shield Convergence args
    parser.add_argument("--shield_convergence_threshold", type=float, default=0.01,
                        help="Minimum improvement in loss for considering convergence.")
    parser.add_argument("--shield_convergence_episodes_interval", type=int, default=100,
                        help="Number of episodes to check for convergence.")

    # cost arguments - cartpole
    parser.add_argument("--safe_limit_x", type=float, default=0.4,
                        help="safety distance from x-treshold (defining cost)")
    parser.add_argument("--safe_limit_theta", type=float, default=0.03,
                        help="safety distance from theta tresh (defining cost)")

    # parse the arguments

    args = parser.parse_args(arguments)
    print(args)
    # General training arguments

    select_action_algo = args.select_action_algo
    env = args.env
    max_ep_len = args.max_ep_len
    max_training_timesteps = args.max_training_timesteps
    ## running setups

    render = args.render
    cpu = args.cpu
    seed = args.seed

    ## logs & frequencies

    log_freq = args.log_freq
    save_model_freq = args.save_model_freq
    record_trajectory_length = args.record_trajectory_length

    # PPO arguments

    ppo_K_epochs = args.ppo_K_epochs
    ppo_eps_clip = args.ppo_eps_clip
    ppo_gamma = args.ppo_gamma
    ppo_lr_actor = args.ppo_lr_actor
    ppo_lr_critic = args.ppo_lr_critic
    # ppo_update_timestep = args.max_ep_len * 4
    # dont train ppo
    ppo_update_timestep = 10000000000000000000000000000000000000000000000000000000000000000
    # Shield arguments

    shield_K_epochs = args.shield_K_epochs
    shield_discount_factor_cost = args.shield_discount_factor_cost
    shield_unsafe_tresh = args.shield_unsafe_tresh
    shield_update_timestep = args.shield_update_timestep
    shield_batch_size = args.shield_batch_size
    shield_buffer_size = args.shield_buffer_size
    shield_lr = args.shield_lr

    # cost arguments - cartpole

    safe_limit_x = args.safe_limit_x
    safe_limit_theta = args.safe_limit_theta
    shield_minimum_buffer_samples = args.shield_minimum_buffer_samples
    shield_sub_batch_size = args.shield_sub_batch_size

    if args.env == "CartPoleWithCost-v0":
        register_env(args.env)
        env = gym.make(args.env, safe_limit_x=safe_limit_x, safe_limit_theta=safe_limit_theta)
    elif args.env == "highway-v0" and args.highway_observation_type == "Kinematics":
        env = gym.make(args.env)
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,  # number of visible vehicles to observaiton
                "features": ["presence", "x", "y", "vx", "vy"],  # Essential features only
                "features_range": {
                    "x": [-50, 50],
                    "y": [-50, 50],
                    "vx": [-10, 10],
                    "vy": [-10, 10]
                },
                "action": {"type": "Discrete"},
            },
            "vehicles_count": 10  # total numbers of vehicles in env
        }
        for k, v in config.items():
            config_t = config[k]
            if type(config_t) == list:
                for k1, v1 in config_t.items():
                    env.unwrapped.config[k][k1] = v1
            else:
                env.unwrapped.config[k] = v
        env.reset()
    elif args.env == "highway-v0" and args.highway_observation_type == "OccupancyGrid":
        env = gym.make(args.env)
        config = {
            "vehicles_count": 10,  # Total number of vehicles in the environment
            "observation": {
                "vehicles_count": 5,
                "type": "OccupancyGrid",
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-50, 50],
                    "y": [-50, 50],
                    "vx": [-10, 10],
                    "vy": [-10, 10]
                },
                "grid_size": [[-20, 20], [-20, 20]],
                "grid_step": [5, 5],
                "absolute": False
            }
        }
        for k, v in config.items():
            config_t = config[k]
            if type(config_t) == list:
                for k1, v1 in config_t.items():
                    env.unwrapped.config[k][k1] = v1
            else:
                env.unwrapped.config[k] = v
        env.reset()
    elif args.env == "highway-v0" and args.highway_observation_type == "TimeToCollision":
        env = gym.make(args.env)
        config = {"observation": {"type": "TimeToCollision", "horizon": 10}}
        for k, v in config.items():
            config_t = config[k]
            for k1, v1 in config_t.items():
                env.unwrapped.config[k][k1] = v1
        env.reset()
    else:
        raise ValueError(
            f"Unsupported environment: {args.env}. Please choose a valid environment (e.g., 'CartPoleWithCost-v0' or 'highway-v0').")
    # env set up

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    shield_convergence_threshold = args.shield_convergence_threshold
    shield_convergence_episodes_interval = args.shield_convergence_episodes_interval

    base_path = args.base_path + f"/observation_type={config['observation']['type']}"
    # create log paths

    if not os.path.exists(base_path):
        print(f"Given base path directory '{base_path}' did not exist. Creating it.. ")
        os.makedirs(base_path)

    save_model_path = f"./{base_path}/model.pth"
    save_shield_path = f"./{base_path}/shield.pth"
    save_args_path = f"./{base_path}/commandline_args.txt"
    save_shield_buffer_samples_path = f"./{base_path}/shield_buffer_samples.pkl"
    save_stats_path = f"./{base_path}/stats.log"
    os.makedirs(base_path + "/Videos", exist_ok=True)

    # Define random seed
    random_seed = seed

    # save arguments to text file

    save_args_path = base_path + "/commandline_args.txt"  # You can customize this path
    with open(save_args_path, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    if random_seed:
        print("random seed is set to ", random_seed)
        torch.manual_seed(random_seed)
        # CUDA seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        env.action_space.seed(random_seed)

    """
    # define agent according to given argument
    if agent == "PPO":
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)

    elif agent == "ShieldPPO":
        ppo_agent = ShieldPPO(state_dim=state_dim, action_dim=action_dim, lr_actor=lr_actor, lr_critic=lr_critic,
                              gamma=gamma, eps_clip=eps_clip, k_epochs_ppo=K_epochs, k_epochs_shield=K_epochs_shield,
                              k_epochs_gen=K_epochs_gen,
                              has_continuous_action_space=has_continuous_action_space, lr_shield=lr_shield,
                              lr_gen=lr_gen, latent_dim=latent_dim, discount_factor_cost=discount_factor_cost,
                              action_std_init=action_std, masking_threshold=shield_masking_tresh,
                              unsafe_tresh=unsafe_tresh, use_gen_v2 = use_gen_v2,shield_buffer_size = shield_buffer_size, shield_batch_size = shield_batch_size,  param_ranges=param_ranges)

    else:
        print("Accepting one of the following agents as input - PPO, ShieldPPO, RuleBasedShieldPPO")
    """

    # Define agents

    if args.env == 'highway-v0' and args.highway_observation_type == "Kinematics":
        # + 1 for concatenated action
        input_size = action_dim * env.unwrapped.config['observation']['vehicles_count'] + 1

    elif args.env == 'highway-v0' and args.highway_observation_type == "OccupancyGrid":
        observation_config = env.unwrapped.config['observation']
        features_channels = len(observation_config['features'])
        grid_size, grid_step = observation_config["grid_size"], observation_config["grid_step"]
        x_span, y_span = grid_size[0][1] - grid_size[0][0], grid_size[1][1] - grid_size[1][0]
        num_x_cells = x_span // grid_step[0]
        num_y_cells = y_span // grid_step[1]
        grid_dim = num_x_cells * num_y_cells
        input_size = grid_dim * features_channels + 1  # + 1 for concatenated action
    elif args.env == 'highway-v0' and args.highway_observation_type == "TimeToCollision":
        input_size = 91
    else:  # cartpole
        input_size = 5
    shield_net = Shield(input_size=input_size, loss_fn=nn.MSELoss(), lr=shield_lr, k_epochs=shield_K_epochs,
                        batch_size=shield_batch_size, buffer_size=shield_buffer_size,
                        sub_batch_size=shield_sub_batch_size).to(device)
    ppo_agent = PPO(state_dim, action_dim, ppo_lr_actor, ppo_lr_critic, ppo_gamma, ppo_K_epochs, ppo_eps_clip, False)

    # counters
    time_step = 0
    # logs
    cost_log = []  # List to log (time_step, cost) at each step
    reward_log = []  # List to log (time_step, reward) at each step
    shield_loss_log = []  # List to log (time_step, loss) at each shield update

    time_step = 0
    costs_per_episode = []
    episode_cnt = 0
    episodes_len = []

    shield_num_updates = 0
    ppo_update_log = pd.DataFrame(columns=['update_timestep', 'loss'])

    while time_step < max_training_timesteps:
        episode_cnt += 1
        if time_step != 0:
            episodes_len.append((episode_cnt, t + 1))
        shield_episode_trajectory = []
        # print(f"Starting episode {episode + 1}...")
        state, info = env.reset()
        done = False
        ep_cumulative_cost = 0
        for t in range(1, max_ep_len + 1):
            if t != 1 and time_step % args.save_buffer_pickle_n == 0:
                batch_samples, _, _ = shield_net.buffer.sample(shield_net.buffer.get_buffer_len())
                batch_samples_states = torch.stack([sample[0] for sample in batch_samples])
                batch_samples_actions = torch.stack([sample[1] for sample in batch_samples])
                batch_samples_costs = torch.stack([sample[2] for sample in batch_samples]).to(device)
                amount_of_samples = len(batch_samples)
                buffer_pkl_filename = base_path + "/" + f"amount_of_samples={amount_of_samples}.pickle"

                with open(buffer_pkl_filename, "wb") as f:
                    pickle.dump(batch_samples, f)
                    print(f"Agent buffer saved to {buffer_pkl_filename} successfully with {len(batch_samples)} samples")

            start_time = time.time()
            if t % shield_update_timestep == 0:
                loss = shield_net.update()
            valid_actions = get_valid_actions(env)
            if select_action_algo == "PPO":
                action = ppo_agent.select_action(state)
            elif select_action_algo == "Random":
                action = random.choice(valid_actions)
            else:
                raise ValueError(
                    f"Unsupported action selection algorithm: {select_action_algo}. Please choose a valid algorithm (e.g., 'PPO' or 'Random').")

            prev_state = state.copy()
            # TO DO : add return of trunk?  state, reward,term ,trunk ,_  = env.step(action)
            if args.env == 'highway-v0':
                state, reward, done, truncated, info = env.step(action)
                cost = info['crashed']
            else:  # other envs
                state, reward, done, info = env.step(action)
                cost = info['cost']
            # add feedback from environment to PPO agent (the rest is added from ppo.select_action method

            if select_action_algo == 'PPO':
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

            if select_action_algo == 'PPO' and time_step % ppo_update_timestep == 0:
                ppo_loss = ppo_agent.update()
                ppo_update_log = pd.concat(
                    [ppo_update_log, pd.DataFrame({'update_timestep': [time_step], 'loss': [ppo_loss]})],
                    ignore_index=True)

            shield_episode_trajectory.append((torch.tensor(prev_state), torch.tensor([action]), cost, done))
            # Log the cost and reward at this time step

            ep_cumulative_cost += cost
            reward_log.append((time_step, reward))  # Store (time_step, reward)
            time_step += 1
            if time_step % 500 == 0:
                print(f"time_step is now {time_step}")

            if done:
                break

        # print(f"Episode {episode_cnt} length is {t+1}, finished due to {done} condition;")
        costs_per_episode.append((episode_cnt + 1, ep_cumulative_cost))
        episode_samples = get_discounted_costs_episode_samples(shield_episode_trajectory, shield_discount_factor_cost)
        states_ = [step[0] for step in episode_samples]
        actions_ = [step[1] for step in episode_samples]
        costs_ = [step[2] for step in episode_samples]

        if args.env == 'highway-v0':
            states_tensor = torch.stack(states_)
            actions_ten = torch.stack(actions_)
            costs_ten = torch.stack(costs_).to(device)
            states_flat = states_tensor.view(len(states_tensor), -1)
            x = torch.cat((states_flat, actions_ten), dim=1).to(device)
            x = x.float()
        else:
            states_ten = torch.squeeze(torch.stack(states_, dim=0)).detach().to(device)
            costs_ten = torch.squeeze(torch.stack(costs_, dim=0)).detach().to(device)
            actions_ten = torch.squeeze(torch.stack(actions_, dim=0)).detach().unsqueeze(1).to(device)
            x = torch.cat([states_ten, actions_ten], -1).to(device)
            x = x.float()
        predictions = shield_net(x).squeeze()
        episode_errors = torch.abs(predictions - costs_ten).cpu().data.numpy()

        for i, sample in enumerate(episode_samples):
            state, action, cost = sample
            sample_error = episode_errors[i]
            shield_net.add_to_buffer(torch.tensor(sample_error), sample)

        # print(f"Finished episode {episode} after {episode_len} time steps")
        """
        # relevant if the update is in the end of each episode
        if shield_net.buffer.get_buffer_len() >= shield_minimum_buffer_samples:
            shield_loss = shield_net.update(num_update=shield_num_updates)
            # Log the shield loss for this update, along with the current time step
            # int(f"Log loss for episode {episode_cnt} = {shield_loss}")
            shield_loss_log.append((time_step, shield_loss))  # Store (time_step, loss)
            shield_num_updates += 1
            print(
                f"Num Update {shield_num_updates}. Average loss for updating Shield after episode {episode_cnt + 1} is: {shield_loss}")
        """
        if episode_cnt % log_freq == 0:
            # print(f"Saving logs to {base_path} in the end of episode {episode_cnt+1}.")
            cost_df = pd.DataFrame(costs_per_episode, columns=['Episode', 'Cumulative Cost'])
            episodes_len_df = pd.DataFrame(episodes_len, columns=['Episode', 'Episode Length'])
            rewards_df = pd.DataFrame(reward_log, columns=['Time Step', 'Reward'])
            shield_losses_df = pd.DataFrame(shield_loss_log, columns=['Update Time Step', 'Loss'])

            # save DataFrames to CSV files
            costs_csv_path = base_path + "/costs_log.csv"
            rewards_csv_path = base_path + "/rewards_log.csv"
            losses_csv_path = base_path + "/shield_losses_updates.csv"
            episodes_len_path = base_path + "/episodes_len.csv"

            cost_df.to_csv(costs_csv_path, index=False)
            rewards_df.to_csv(rewards_csv_path, index=False)
            shield_losses_df.to_csv(losses_csv_path, index=False)
            episodes_len_df.to_csv(episodes_len_path, index=False)

    if select_action_algo == 'PPO':
        # save ppo loss
        ppo_update_log.to_csv(f"{base_path}/ppo_update_log.csv", index=False)
    # save pickle in the end
    batch_samples, _, _ = shield_net.buffer.sample(shield_net.buffer.get_buffer_len())
    batch_samples_states = torch.stack([sample[0] for sample in batch_samples])
    batch_samples_actions = torch.stack([sample[1] for sample in batch_samples])
    batch_samples_costs = torch.stack([sample[2] for sample in batch_samples]).to(device)
    amount_of_samples = len(batch_samples)
    buffer_pkl_filename = base_path + "/" + f"amount_of_samples={amount_of_samples}.pickle"
    with open(buffer_pkl_filename, "wb") as f:
        pickle.dump(batch_samples, f)
        print(f"Agent buffer saved to {buffer_pkl_filename} successfully with {len(batch_samples)} samples")

    # shield_losses_df = pd.DataFrame(shield_loss_log, columns=['Update Time Step', 'Loss'])
    # shield_losses_df.to_csv(losses_csv_path, index=False)


if __name__ == '__main__':
    train()
