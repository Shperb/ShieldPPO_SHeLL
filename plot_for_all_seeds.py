# Imports
import numpy as np
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import the color map module
import os


"""

# Arguments
arguments = None
parser = argparse.ArgumentParser()
# --seeds_list input example: seed1, seed2, seed3
parser.add_argument(
    '--seeds_list',
    nargs='+',
    help='A list of seeds (as string)',
    required=True
)

parser.add_argument(
    '--runs_base_paths',
    nargs='+',
    help='A list base path, each base-path is for one run (with all seeds)',
    required=True
)

parser.add_argument(
    '--save_path',
    type=str,
    help='A path that indicates the location to store the plots.',
    required=True
)

args = parser.parse_args(arguments)
seeds_list = args.seeds_list
save_path = args.save_path
# TODO - Define CONFGS_BASE_PATHS

runs_base_paths = args.runs_base_paths


# ------------------------------------ aid functions

def plot_reward_collisions(stats_avg_across_seeds_lst, stats_renv_avg_across_seeds_lst, save_path, labels_lst):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = cm.rainbow(np.linspace(0, 1, len(stats_avg_across_seeds_lst)))
    # Plot Training Rewards and Collisions
    for i, stats_avg_across_seeds in enumerate(stats_avg_across_seeds_lst):
        df = stats_avg_across_seeds
        # Plot Rewards
        #     (t_mean_stats,r_mean_stats,c_mean_stats, r_std_stats, c_std_stats)
        axes[0, 0].plot(df[0], smooth(df[1]), label=labels_lst[i], color=colors[i], alpha=0.7)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards over Time')
        axes[0,0].fill_between(df[0], smooth(df[1]) - smooth(df[3]), smooth(df[1]) + smooth(df[3]), color = colors[i], alpha = 0.1)
        # Plot Collisions (Costs)
        axes[0, 1].plot(df[0], smooth(df[2]), color=colors[i])
        axes[0,1].fill_between(df[0], smooth(df[2]) - smooth(df[4]), smooth(df[2]) + smooth(df[4]), color = colors[i], alpha = 0.1)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Collisions')
        axes[0, 1].set_title("Training Collisions over Time")

    # Plot Evaluation Rewards and Collisions
    for i, stats_renv_avg_across_seeds in enumerate(stats_renv_avg_across_seeds_lst):
        eval_df = stats_renv_avg_across_seeds
        # Plot Rewards
        axes[1, 0].plot(eval_df[0], smooth(eval_df[1]), label=labels_lst[i], color=colors[i], alpha=0.7)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Evaluation Rewards over Time')
        # Plot Collisions (Costs)
        axes[1, 1].plot(eval_df[0], smooth(eval_df[2]), color=colors[i])
        # TODO -    add STD     axes[0,1].fill_between(df[0], smooth(df[2]) - smooth(df[4]), smooth(df[2]) + smooth(df[4]), color = colors[i], alpha = 0.1)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Collisions')
        axes[1, 1].set_title("Evaluation Collisions over Time")

    # Adjust layout and save the figure
    plt.legend(labels=labels_lst, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.suptitle("Training and Evaluation Costs and Rewards over Time", y=1.02)
    plt.savefig(save_path + "/combined-cost-rewards-over-time.png")
    #plt.show()
    plt.clf()


def plot_shield_gen_losses(shield_stats_avg_across_seeds_lst, gen_stats_avg_across_seeds_lst, save_path, labels_lst):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = cm.rainbow(np.linspace(0, 1, len(shield_stats_avg_across_seeds_lst)))
    # Plot Training Rewards and Collisions
    for i, shield_stats_avg_across_seeds in enumerate(shield_stats_avg_across_seeds_lst):
        df = shield_stats_avg_across_seeds
        # Plot Rewards
        axes[0, 0].plot(df[0], smooth(df[1]), label=labels_lst[i], color=colors[i], alpha=0.7)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Shield Loss')
        axes[0, 0].set_title('Shield loss over update Time')

    # Plot Evaluation Rewards and Collisions
    for i, gen_stats_avg_across_seeds in enumerate(gen_stats_avg_across_seeds_lst):
        gen_df = gen_stats_avg_across_seeds
        # Plot Rewards
        axes[1, 0].plot(gen_df[0], smooth(gen_df[1]), label=labels_lst[i], color=colors[i], alpha=0.7)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Gen Loss')
        axes[1, 0].set_title('Gen loss over Time')

    # Adjust layout and save the figure
    plt.legend(labels=labels_lst, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_path + "/gen_and_shield_losses_over_time.png")
    #plt.show()
    plt.clf()


def smooth(x, window_size=1000):
    x = np.array(x)
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        try:
            b[i] = x[max(0, i - window_size):i].mean()
        except RuntimeWarning as e:
            print(314)
    return b

# ------------------------------------------ # main
def change_loss_logs_into_tuples(loss_stats_log):
    # get the same structures like stats structure - (time_steps, shield_loss )
    # in the shield each time_step represents an update time step, whereas in the stats it represents 1 episode
    time_steps = []
    losses_list = []
    # Iterate through the dictionary and populate the lists
    for key, (i_episode, t, loss) in loss_stats_log.items():
        # each timestep represents 1 update (not episode like in stats.log)
        time_steps.append(key)
        losses_list.append(loss)
    return (time_steps, losses_list)

def load_seeds_stats_as_df(base_path, seeds_list):
    # stats = (time_steps, rewards, costs, tasks, duration, episodes_len, amount_of_done, i_episode)
    # time_steps, rewards, costs, .. - are lists. each entry in the list represents a time step.
    #seeds_stats_tuples, seeds_stats_renv_tuples, gen_stats_tuples, shield_stats_tuples = [], [], [], []
    seeds_stats_tuples, seeds_stats_renv_tuples = [], []
    for seed in seeds_list:
        stats_path = os.path.join(base_path, f"seed={seed}", "stats.log")
        stats_renv_path = os.path.join(base_path, f"seed={seed}", "stats_renv.log")
        seeds_stats_tuples.append(torch.load(stats_path))
        seeds_stats_renv_tuples.append(torch.load(stats_renv_path))
    return seeds_stats_tuples, seeds_stats_renv_tuples

def load_seeds_loss_stats_as_df(base_path, seeds_list):
    # loss_stats structure is ([update_time_steps], [losses])
    seeds_shield_loss_stats_tuples, seeds_gen_loss_stats_tuples = [], []
    for seed in seeds_list:
        shield_stats_path = os.path.join(base_path, f"seed={seed}", "shield_loss_stats.log")
        gen_stats_path = os.path.join(base_path, f"seed={seed}", "gen_loss_stats.log")
        shield_stats_tup = change_loss_logs_into_tuples(torch.load(shield_stats_path))
        gen_stats_tup = change_loss_logs_into_tuples(torch.load(gen_stats_path))
        seeds_shield_loss_stats_tuples.append(shield_stats_tup)
        seeds_gen_loss_stats_tuples.append(gen_stats_tup)

    return seeds_shield_loss_stats_tuples, seeds_gen_loss_stats_tuples


def compute_stats_min_l(stats_list):
    stats_min_l = min([len(stats[0]) for stats in stats_list])
    return stats_min_l

def compute_avg_across_loss_stats(loss_stats_min_l, stats_list):
    t_loss_stats = np.stack([seed[0][:loss_stats_min_l] for seed in stats_list])
    loss_stats = np.stack([seed[1][:loss_stats_min_l] for seed in stats_list])
    t_mean_loss_stats = t_loss_stats.mean(axis = 0)
    t_std_loss_stats = t_loss_stats.std(axis = 0)

    loss_stats_mean = loss_stats.mean(axis = 0)
    loss_stats_std = loss_stats.std(axis = 0)
    return (t_mean_loss_stats, loss_stats_mean)


def compute_avg_across_stats(stats_min_l, stats_list):
    t_stats = np.stack([seed[0][:stats_min_l] for seed in stats_list])
    r_stats = np.stack([seed[1][:stats_min_l] for seed in stats_list])
    c_stats = np.stack([seed[2][:stats_min_l] for seed in stats_list])
    t_mean_stats = t_stats.mean(axis = 0)
    t_std_stats = t_stats.std(axis=0)
    r_mean_stats = r_stats.mean(axis=0)  # Mean across all seeds
    r_std_stats = r_stats.std(axis=0)    # Standard deviation across all seeds
    c_mean_stats = c_stats.mean(axis=0)
    c_std_stats = c_stats.std(axis=0)

    return (t_mean_stats,r_mean_stats,c_mean_stats, r_std_stats, c_std_stats)

def return_avg_across_seeds_for_one_run(base_path, seeds_list):
    seeds_stats_tuples, seeds_stats_renv_tuples = load_seeds_stats_as_df(base_path, seeds_list)
    stats_min_l, stats_renv_min_l = compute_stats_min_l(seeds_stats_tuples),compute_stats_min_l(seeds_stats_renv_tuples)
    stats_avg_across_seed, stats_renv_avg_across_seeds = compute_avg_across_stats(stats_min_l, seeds_stats_tuples), compute_avg_across_stats(stats_renv_min_l, seeds_stats_renv_tuples)
    return stats_avg_across_seed, stats_renv_avg_across_seeds


def return_avg_across_seeds_for_one_run_loss(base_path, seeds_list):
    seeds_shield_loss_stats_tuples, seeds_gen_loss_stats_tuples = load_seeds_loss_stats_as_df(base_path, seeds_list)
    shield_stats_min_l, gen_stats_min_l = compute_stats_min_l(seeds_shield_loss_stats_tuples),compute_stats_min_l(seeds_gen_loss_stats_tuples)
    shield_stats_avg_across_seed = compute_avg_across_loss_stats(shield_stats_min_l, seeds_shield_loss_stats_tuples)
    gen_stats_avg_across_seeds = compute_avg_across_loss_stats(gen_stats_min_l, seeds_gen_loss_stats_tuples)
    return shield_stats_avg_across_seed, gen_stats_avg_across_seeds



### main code

# create average stats for each path - path is a configuration.

confgs_stats = []
confgs_stats_renv = []

confgs_shield_stats = []
confgs_gen_stats = []

for base_path in runs_base_paths:
    stats_avg_across_seed, stats_renv_avg_across_seeds = return_avg_across_seeds_for_one_run(base_path,seeds_list)
    shield_stats_avg_across_seed, gen_stats_avg_across_seeds = return_avg_across_seeds_for_one_run_loss(base_path,seeds_list)

    confgs_stats.append(stats_avg_across_seed)
    confgs_stats_renv.append(stats_renv_avg_across_seeds)

    confgs_shield_stats.append(shield_stats_avg_across_seed)
    confgs_gen_stats.append(gen_stats_avg_across_seeds)

plot_reward_collisions(confgs_stats, confgs_stats_renv, save_path, ["gen", "no-gen"])

# we do average to time steps it's weird (because they change)

plot_shield_gen_losses(confgs_shield_stats, confgs_gen_stats, save_path, ["gen", "no-gen"])
"""
log_stats = torch.load("C:/Users/Shir/PycharmProjects/ShieldPPO-updated/models/22.06.24_seeds_experiments_gen_v2/gen/seed=2/shield_loss_stats.log")
print(log_stats)