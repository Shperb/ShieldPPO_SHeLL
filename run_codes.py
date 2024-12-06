import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

# set device to cpu or cuda
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")

fig, axs = plt.subplots(1, 3, figsize=(16, 6))  # Adjust figsize as needed
LOG_FILE_NAME = 'stats'
SHIELD_LOG_FILE_NAME = 'shield_loss_stats'


def smooth(x, window_size=200):
    x = np.array(x)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - window_size):min(n, i + window_size)].mean()
    return b


def plot_rewards(ax, df, obs, window_size=15000):
    ax.plot(df[0], smooth(df[1], window_size), label=obs)  # 500
    ax.set_xlabel(r'Time Steps ($\times 10^3$)')
    ax.set_ylabel('Avg Episodic Reward')
    # ax.set_title(f'{obs} Observation')
    ax.set_title('Rewards Graph')


def plot_collisions(ax, df, obs, window_size=15000):
    ax.plot(df[0], smooth(df[2], window_size))
    ax.set_xlabel(r'Time Steps ($\times 10^3$)')
    ax.set_ylabel('Collisions')
    ax.set_title('Collisions Graph')


def plot_ep_len(ax, df, obs, window_size=15000):
    ax.plot(df[0], smooth(df[3], window_size))
    ax.set_xlabel(r'Time Steps ($\times 10^3$)')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length Graph')


def plot_shield_loss(ax, df, obs, window_size=150):
    ax.plot(df[0], smooth(df[1], window_size), label=obs)  # 10
    ax.set_xlabel(r'Time Steps ($\times 10^3$)')
    ax.set_ylabel('Shield Loss')
    ax.set_title('Shield Loss Graph')


def get_first_ten_percent(stats):
    threshold = stats[0][-1] * 0.1
    slice_index = next((i for i, x in enumerate(stats[0]) if x > threshold), len(stats[0]))
    sliced_data = tuple(lst[:slice_index] for lst in stats)
    return sliced_data


def plot_runs(paths, plots, test_name, close=False):
    window_size = 2000 if close else 15000
    window_size_shield = 150 if close else 1500

    # for folder_path in paths:
    for folder_path, p in zip(paths, plots):
        print(folder_path)
        stats = get_stats_from_log_files(folder_path, LOG_FILE_NAME)
        shield_log_stats = get_stats_from_log_files(folder_path, SHIELD_LOG_FILE_NAME)
        observation = p

        if close:
            stats = get_first_ten_percent(stats)
            shield_log_stats = get_first_ten_percent(shield_log_stats)

        plot_collisions(axs[0], stats, observation, window_size)
        plot_rewards(axs[1], stats, observation, window_size)
        plot_shield_loss(axs[2], shield_log_stats, observation, window_size_shield)
        # plot_ep_len(axs[3], stats, observation)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    # axs[3].legend()

    # Save and show the combined plot
    save_filename = f'{test_name}_close.png' if close else f'{test_name}.png'
    main_dir = os.path.dirname(os.path.dirname(folder_path))
    save_path = os.path.join(main_dir, save_filename)
    plt.savefig(save_path)
    plt.show()


def get_stats_from_log_files(folder_path, starts_with):
    stats = tuple()
    log_files = [file for file in os.listdir(folder_path) if file.startswith(starts_with) and file.endswith('.json')]
    if not log_files:
        for file in os.listdir(folder_path):
            if os.path.isdir(f"{folder_path}/{file}"):
                for f in os.listdir(f"{folder_path}/{file}"):
                    if f.startswith(starts_with) and f.endswith('.json'):
                        log_files.append(f"{file}/{f}")
    for i in range(1, len(log_files) + 1):
        file_path = f'{folder_path}/{log_files[i - 1]}'
        with open(file_path, 'r') as f:
            current_stats = tuple(json.load(f))

        modified_timestep = [value / 1e3 for value in current_stats[0]]

        # Create a new tuple with the modified first list and the other lists unchanged
        current_stats = (modified_timestep, *current_stats[1:])

        if i == 1:
            stats = stats + current_stats
        else:
            stats = [list1 + list2 for list1, list2 in zip(stats, current_stats)]
        print(i)
        if i == 5:
            break

    return stats


def find_latest_edited_folder(directory, k=1):
    if not os.path.exists(directory):
        return "Directory does not exist"

    # Get all folders in the directory
    folders = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not folders:
        return "No folders found in the directory"

    # Sort folders by last modified time in descending order
    sorted_folders = sorted(folders, key=os.path.getmtime, reverse=True)

    return [f.replace('\\', '/') for f in sorted_folders[:k]]  # Select the top k folders


def plot_last_runs(stats_type, k=1):
    paths = find_latest_edited_folder(f"models/{stats_type}_stats/occ3kin2_3_20240613-131135/", k)
    # paths.append(f'models/{stats_type}_stats/occu1_3_20240613-213400/Occupancy1_CO')
    # paths.append(f'models/{stats_type}_stats/Occupancy_20240610-183636')
    paths = find_latest_edited_folder(f"models/{stats_type}_stats/1606/", k)

    plots = ["kin", "kin", "occu", "occu", "occu", 'occureg', 'PPO']
    plot_runs(paths, plots)


def plot_paths(stats_type):
    folder = "models/{}_stats/0410/{}"
    subfolder = 'occu3_3_20240922-132538'
    test_name = "occu_3v1"
    # paths = ["occ3kin2_5_20240615-204126\Occupancy1_CO", "occu1_5_20240615-204026\Occupancy1_CO", "ppo_5_20240615-204509\Occupancy1_CO"]
    # paths = ["0607\occu2kin2_lcl_4_20240706-123333\Kinematics3_CO", "0607\kin1_lcl_4_20240706-123342\Kinematics1_CO", "0707\occu2kin2_lcl_4_20240707-001150\Kinematics3_CO"]
    # paths = ["0607\occu1_lcl_4_20240706-121526\Occupancy1_CO", "0607\occu3_lcl_4_20240706-121526\Occupancy1_CO", "0707\occu3_lcl_4_20240707-001150\Occupancy1_CO", "0707\occu1_lcl_4_20240707-001150\Occupancy1_CO", r"0707\ppo_4_20240706-213348"]
    # paths = ["0707\occu3_lcl_4_20240707-001150\Occupancy1_CO", "0707\occu1_lcl_4_20240707-001150\Occupancy1_CO"]
    # paths = ["0807\occu3_lcl_4_20240708-104824\Occupancy1_CO", r"0807\occu1_lcl_4_20240708-104836\Occupancy1_CO", "0707\ppo_4_20240706-213348", "1207\occu3_lcl_4_20240710-065528\Occupancy1_CO"]
    # paths = ["occu3_ppr_3_20240905-183637\Occupancy1_CO", "occu3_ppr_3_20240905-183637\Occupancy2_CO", "occu3_ppr_3_20240905-183637\Occupancy3_CO", "occu1_ppr_3_20240905-183626\Occupancy1_CO"]
    paths = [f"{subfolder}\Occupancy1_CO", f"{subfolder}\Occupancy2_CO", f"{subfolder}\Occupancy3_CO", "occu1_3_20240922-132538\Occupancy1_CO"]

    paths = [folder.format(stats_type, p) for p in paths]
    # plots = ["solo", "trio", "solonew", "trionew", "ppo"]
    plots = ["M1", "M2", "M3", "S1"]
    # plot_runs(paths, plots, test_name, close=False)
    plot_runs(paths, plots, test_name, close=True)


obs = "Camera"
# obs = "Kinematics"
# stats_type = 'CartPole'
stats_type = 'Highway'
# stats_type = 'CarRacing'

# plot_last_runs(stats_type, 3)
plot_paths(stats_type)
