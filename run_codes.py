import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

################################## set device ##################################

# set device to cpu or cuda

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")


#    torch.save((time_steps, rewards, costs, tasks, total_training_time, episodes_len), save_stats_path)

def smooth(x, window_size=200):
    x = np.array(x)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - window_size):min(n, i + window_size)].mean()
    return b


def create_loss_stats_df(data_path):
    log_stats = torch.load(data_path, map_location=torch.device('cpu'))
    log_stats_df = pd.DataFrame([(key, *value) for key, value in log_stats.items()],
                                columns=["Time Step", "Episode", "Step in Episode", "Shield Loss", "Len of Pos Samples",
                                         "Avg Prediction for Pos Samples", "Len of Neg Samples",
                                         "Avg Prediction for Neg Samples"])
    return log_stats_df


def plot_predictions(shield_loss_stats_df, y_axis, plot_name):  # Create separate subplots for each line
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    # Plot 'Len of Pos Samples'
    first_line, sec_line = y_axis[0], y_axis[1]
    axes[0].plot(shield_loss_stats_df['Time Step'], shield_loss_stats_df[first_line], label=first_line, color='blue')
    axes[0].set_ylabel(first_line)
    axes[0].legend()

    # Plot 'Avg Prediction for Pos Samples'
    axes[1].plot(shield_loss_stats_df['Time Step'], shield_loss_stats_df[sec_line], label=sec_line, color='orange')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('sec_line')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()


def plot_shield_loss(shield_loss_stats_df, save_path):
    # save_path - path to be saved
    plt.plot(shield_loss_stats_df['Time Step'], smooth(shield_loss_stats_df['Shield Loss']), label='Shield Loss', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Shield Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def plot_rewards(df, save_path, obs):
    plt.plot(df[0], smooth(df[1], 5000))
    plt.xlabel('Time Step')
    plt.ylabel('Avg Episodic Reward')
    plt.title(f'{obs} Observation')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def plot_collisions(df, save_path, obs):
    plt.plot(df[0], smooth(df[2], 1000))
    plt.xlabel('Time Step')
    plt.ylabel('Collisions')
    plt.title(f'{obs} Observation')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def plot_shield_loss_new(df, save_path, obs):
    plt.plot(df[0], smooth(df[1], 50))
    plt.xlabel('Time Step')
    plt.ylabel('Shield Loss')
    plt.title(f'{obs} Observation')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


dev = 'cpu'


def plot_camera(stats_type):
    obs = 'Camera'
    folder_path = f"models/{stats_type}_stats/Camera_20231225-195134"
    stats = torch.load(f"{folder_path}/stats.log", map_location=torch.device(dev))
    plot_collisions(stats, f"{folder_path}/collisions.png", obs)
    plot_rewards(stats, f"{folder_path}/rewards.png", obs)
    shield_loss_stats_df = torch.load(f"{folder_path}/shield_loss_stats.log", map_location=torch.device(dev))
    plot_shield_loss_new(shield_loss_stats_df, f"{folder_path}/shield_loss.png", obs)


def plot_kinematics(stats_type):
    obs = 'Kinematics'
    folder_path = f"models/{stats_type}_stats/2812/Kinematics_20231227-220531"
    stats = torch.load(f"{folder_path}/stats.log", map_location=torch.device(dev))
    plot_collisions(stats, f"{folder_path}/collisions.png", obs)
    plot_rewards(stats, f"{folder_path}/rewards.png", obs)
    shield_loss_stats_df = torch.load(f"{folder_path}/shield_loss_stats.log", map_location=torch.device(dev))
    plot_shield_loss_new(shield_loss_stats_df, f"{folder_path}/shield_loss.png", obs)


stats_type = 'Cartpole'
# stats_type = 'Highway'
# plot_camera(stats_type)
plot_kinematics(stats_type)

# obs_name = "Camera"
# stats_camera = torch.load("models/safety_thresholds/Camera_20231217-160114/stats.log", map_location=torch.device('cpu'))
# plot_shield_loss(shield_loss_stats_df, "models/long run/masking=0/loss_over_time.png")
# plot_all(stats_camera)

# plot_shield_loss(shield_loss_stats_df, "models/safety_thresholds/Kinematics_20231218-144311/shield_loss.png")

# check_update - masking = 0
# shield_loss_stats_df = create_loss_stats_df("models/long run/masking=0/shield_loss_stats.log")
# shield_loss_stats_df.to_csv("models/long run/masking=0/shield_loss_stats.csv")
# plot_predictions(shield_loss_stats_df,['Len of Pos Samples','Avg Prediction for Pos Samples'], "models/long run/masking=0/pos_samples.png")
# plot_predictions(shield_loss_stats_df,['Len of Neg Samples','Avg Prediction for Neg Samples'], "models/long run/masking=0/neg_samples.png")
# plot_shield_loss(shield_loss_stats_df, "models/long run/masking=0/loss_over_time.png")

# check_update - masking = 100K
# shield_loss_stats_df = create_loss_stats_df("models/5.12/check_fupdate_masking=100K/shield_loss_stats.log")
# shield_loss_stats_df.to_csv("models/5.12/check_fupdate_masking=100K/shield_loss_stats.csv")
# plot_predictions(shield_loss_stats_df,['Len of Pos Samples','Avg Prediction for Pos Samples'], "models/5.12/check_fupdate_masking=100K/pos_samples.png")
# plot_predictions(shield_loss_stats_df,['Len of Neg Samples','Avg Prediction for Neg Samples'], "models/5.12/check_fupdate_masking=100K/neg_samples.png")
# plot_shield_loss(shield_loss_stats_df, "models/5.12/check_fupdate_masking=100K/loss_over_time.png")


# check_no shield
# shield_loss_stats_df = create_loss_stats_df("models/5.12/check_no_shield/shield_loss_stats.log")
# shield_loss_stats_df.to_csv("models/5.12/check_no_shield/shield_loss_stats.csv")
# plot_predictions(shield_loss_stats_df,['Len of Pos Samples','Avg Prediction for Pos Samples'], "models/5.12/check_no_shield/pos_samples.png")
# plot_predictions(shield_loss_stats_df,['Len of Neg Samples','Avg Prediction for Neg Samples'], "models/5.12/check_no_shield/neg_samples.png")
# plot_shield_loss(shield_loss_stats_df, "models/5.12/check_no_shield/loss_over_time.png")


# GOOD:
# Kinematics_20231218-144311
# Kinematics_20231223-125403
# Kinematics_20231225-195220  cart pole kinematics 1st time on cluster

