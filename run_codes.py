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


fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Adjust figsize as needed


def smooth(x, window_size=200):
    x = np.array(x)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - window_size):min(n, i + window_size)].mean()
    return b


def plot_rewards(ax, df, obs):
    ax.plot(df[0], smooth(df[1], 50))
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Avg Episodic Reward')
    ax.set_title(f'{obs} Observation')


def plot_collisions(ax, df, obs):
    ax.plot(df[0], smooth(df[2]))
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Collisions')
    ax.set_title(f'{obs} Observation')


def plot_shield_loss_new(ax, df, obs):
    ax.plot(df[0], smooth(df[1], 5))
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Shield Loss')
    ax.set_title(f'{obs} Observation')


def plot_kinematics(paths, obs):
    for folder_path in paths:
        stats = torch.load(f"{folder_path}/stats.log", map_location=torch.device(device))
        # shield_loss_stats_df = torch.load(f"{folder_path}/shield_loss_stats.log", map_location=torch.device(dev))

        plot_collisions(axs[0], stats, obs)
        plot_rewards(axs[1], stats, obs)
        # plot_shield_loss_new(axs[2], shield_loss_stats_df, obs)

    axs[0].legend()
    axs[1].legend()

    # Save and show the combined plot
    plt.savefig(f"{folder_path}/all.png")
    plt.show()


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


def plot_last_runs(stats_type, obs, k=1):
    paths = find_latest_edited_folder(f"models/{stats_type}_stats", k)
    plot_kinematics(paths, obs)


def plot_paths(stats_type, obs):
    folder = "models/{}_stats/{}"
    paths = ["Kinematics_20240111-144135", "Kinematics_20240111-145554"]
    paths = [folder.format(stats_type, p) for p in paths]
    plot_kinematics(paths, obs)


# obs = "Camera"
obs = "Kinematics"
stats_type = 'CartPole'
# stats_type = 'Highway'

plot_last_runs(stats_type, obs, 3)
# plot_paths(stats_type, obs)
