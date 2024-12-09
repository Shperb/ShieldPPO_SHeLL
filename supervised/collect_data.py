import pickle
import torch
import gymnasium as gym
from gymnasium import spaces, register
import highway_env

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


def env_setup(env_name, cp_safe_limit_x=None, cp_safe_limit_theta=None, highway_obs_type="Kinematics"):
    if env_name == "CartPoleWithCost-v0":
        register_env(env_name)
        env = gym.make(env_name, safe_limit_x=cp_safe_limit_x, safe_limit_theta=cp_safe_limit_theta)
    elif env_name == "highway-v0" and highway_obs_type == "Kinematics":
        env = gym.make(env_name)
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,  # number of visible vehicles to observation
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
    elif env_name == "highway-v0" and highway_obs_type == "OccupancyGrid":
        env = gym.make(env_name)
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
    elif env_name == "highway-v0" and highway_obs_type == "TimeToCollision":
        env = gym.make(env_name)
        config = {"observation": {"type": "TimeToCollision", "horizon": 10}}
        for k, v in config.items():
            config_t = config[k]
            for k1, v1 in config_t.items():
                env.unwrapped.config[k][k1] = v1
        env.reset()
    else:
        raise ValueError(
            f"Unsupported environment: {env_name}. Please choose a valid environment (e.g., 'CartPoleWithCost-v0' or 'highway-v0').")

    return env


def dump_data_to_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {path}")


def compute_discounted_cost(episode_len, discount_factor=0.7):
    # Last state cost is always 1
    discounted = []
    for i in range(episode_len):
        # distance from the last state
        steps_from_end = (episode_len - 1) - i
        discounted_cost = discount_factor ** steps_from_end
        discounted.append(discounted_cost)
    return discounted


def collect_data(render=False):
    env = env_setup("highway-v0", highway_obs_type="Kinematics")

    data_path = "data/states_actions_cost.pkl"
    data = []
    max_steps = 1000000  # 1e6
    log_freq = 1e4
    print_freq = 1e3

    print(f"Collecting data with {max_steps} time steps")
    global_step = 1
    while global_step <= max_steps:
        state, _ = env.reset()
        ep_traj = []
        ep_t = 0
        done = False
        truncated = False

        while not done and not truncated and global_step <= max_steps:
            if global_step % print_freq == 0:
                print(f"Collected {global_step} samples")

            if global_step % log_freq == 0:
                dump_data_to_pkl(data_path, data)

            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)

            cost = 1.0 if info.get('crashed', False) else 0.0
            ep_traj.append((global_step, ep_t, state, action, cost))

            state = next_state
            global_step += 1
            ep_t += 1

        discounted_costs = compute_discounted_cost(len(ep_traj))

        # Combine discounted costs into the final data
        # Each record: (global_step, episode_step, state, action, cost, discounted_cost)
        for i, step in enumerate(ep_traj):
            g_t, e_t, s, a, c = step
            d_c = discounted_costs[i]
            data.append((g_t, e_t, s, a, c, d_c))

    # Loading back the data
    with open(data_path, 'rb') as f:
        loaded_data = pickle.load(f)
    print("Loaded data length:", len(loaded_data))


if __name__ == '__main__':
    collect_data()
