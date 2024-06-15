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


def smooth(x, window_size=200):
    x = np.array(x)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - window_size):min(n, i + window_size)].mean()
    return b


def plot_rewards(ax, df, obs):
    ax.plot(df[0], smooth(df[1], 5000), label=obs)  # 500
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg Episodic Reward')
    # ax.set_title(f'{obs} Observation')
    ax.set_title(f'Rewards Graph')


def plot_collisions(ax, df, obs):
    ax.plot(df[0], smooth(df[2], 1000))
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Collisions')
    ax.set_title(f'Collisions Graph')


def plot_shield_loss_new(ax, df, obs):
    ax.plot(df[0], smooth(df[1], 20), label=obs)  # 10
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Shield Loss')
    ax.set_title(f'Shield Loss Graph')


def plot_runs_old(paths):
    plots = ["Solo", "Multi", "Multi"]
    # for folder_path in paths:
    for folder_path, p in zip(paths, plots):
        stats = torch.load(f"{folder_path}/stats.log", map_location=torch.device(device))
        shield_loss_stats_df = torch.load(f"{folder_path}/shield_loss_stats.log", map_location=torch.device(device))
        # observation = folder_path.split('/')[3].split('_')[0]  # Kinematics or Camera

        observation = p
        plot_collisions(axs[0], stats, observation)
        plot_rewards(axs[1], stats, observation)
        plot_shield_loss_new(axs[2], shield_loss_stats_df, observation)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    # Save and show the combined plot
    # plt.savefig(f"models/CartPole_stats/shield_check/all.png")
    plt.savefig(f"{folder_path}/{observation}.png")
    plt.show()


def plot_runs(paths, plots):
    # for folder_path in paths:
    for folder_path, p in zip(paths, plots):
        stats = get_stats_from_log_files(folder_path, 'stats')
        shield_log_stats = get_stats_from_log_files(folder_path, 'shield_loss_stats')
        # index = 0
        # for i in range(len(stats[0])):
        #     if stats[0][i] > 500000:
        #         index = i
        # stats = [st[index:] for st in stats]
        observation = p
        plot_collisions(axs[0], stats, observation)
        plot_rewards(axs[1], stats, observation)
        plot_shield_loss_new(axs[2], shield_log_stats, observation)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    # Save and show the combined plot
    # plt.savefig(f"models/CartPole_stats/shield_check/all.png")
    plt.savefig(f"{folder_path}/{observation}.png")
    plt.show()


def get_stats_from_log_files(folder_path, starts_with):
    stats = tuple()
    log_files = [file for file in os.listdir(folder_path) if file.startswith(starts_with) and file.endswith('.json')]
    for i in range(1, len(log_files) + 1):
        file_path = f'{folder_path}/{starts_with}{i}.json'
        with open(file_path, 'r') as f:
            current_stats = tuple(json.load(f))

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
    paths.append(f'models/{stats_type}_stats/occu1_3_20240613-213400/Occupancy1_CO')
    paths.append(f'models/{stats_type}_stats/Occupancy_20240610-183636')
    plots = ["kin", "kin", "occu", "occu", "occu", 'occureg', 'PPO']
    plot_runs(paths, plots)


def plot_paths(stats_type):
    folder = "models/{}_stats/{}"
    paths = ["Occupancy_20240610-183636", "Occupancy_20240609-141931_CO"]
    # paths = ["Occupancy_20240609-141120_CO", "Occupancy_20240609-141931_CO"]
    # paths = ["Occupancy_20240609-141931_CO"]

    paths = [folder.format(stats_type, p) for p in paths]
    # plots = ["0.01"]
    plots = ["PPO", "0.01"]
    plot_runs(paths, plots)


obs = "Camera"
# obs = "Kinematics"
# stats_type = 'CartPole'
stats_type = 'Highway'
# stats_type = 'CarRacing'

plot_last_runs(stats_type, 5)
# plot_paths(stats_type)
