import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seeds = [
    10719,
    39112,
    69235,
    93759,
    49230,
]

tasks_names = {'EgoMergeEnvNoNormalization': 'Ego Vehicle Merging',
               'HighwayEnvFastNoNormalization': 'Driving on a Highway',
               'IntersectionEnvNoNormalization': 'Turning at an Intersection',
               'MergeEnvNoNormalization': 'Other Vehicle Merging',
               'TwoWayEnvNoNormalization': 'Two-way Road',
               'UTurnEnvNoNormalization': 'U-turn'}

methods = ["PPO", "ShieldPPO", 'PPOCaR',"PPO-Lagrangian","CPO"]

results = {}

env_reward_bound = {}


def transform(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


for m in methods:
    results[m] = {}
    for seed in seeds:
        results[m][seed] = {}
        filename = f"models2/{m}_{seed}/stats.log"
        if m == 'PPO-RS':
            with open(f"models2/{m}_{seed}/test.txt", 'r') as f:
                time_steps = []
                rewards = []
                costs = []
                tasks = []
                lines = f.readlines()
                for line in lines:
                    res = [v for v in line.strip().split("\t")]
                    time_steps.append(int(res[0]))
                    rewards.append(float(res[1]))
                    costs.append(float(res[2]))
                    tasks.append(res[3])

        else:
            res = torch.load(filename)
            time_steps = res[0]
            rewards = res[1]
            costs = res[2]
            tasks = res[3]
        time_steps, rewards, costs, tasks = map(np.array, [time_steps, rewards, costs, tasks])

        unique = np.unique(tasks)
        for task in unique:
            idx = (tasks == task)
            t = time_steps[idx]
            r = rewards[idx]
            c = costs[idx]
            if task not in env_reward_bound:
                env_reward_bound[task] = (r.min(), r.max())
            else:
                past = env_reward_bound[task]
                env_reward_bound[task] = (min(r.min(), past[0]), max(r.max(), past[1]))
            results[m][seed][task] = (t, r, c)

# plot
fig, axs = plt.subplots(2, 6, figsize=(15, 5), sharex=True)

colors = [f"C{i}" for i in range(10)]


def smooth(x):
    n = x.shape[0]
    b = np.zeros((n,))
    for i in range(n):
        b[i] = x[max(0, i - 50):min(n, i + 50)].mean()
    return b


for j, m in enumerate(methods):
    for i, task in enumerate(env_reward_bound.keys()):
        min_l = np.min([len(x) for x in [results[m][sd][task][0] for sd in seeds]])
        t = np.stack([results[m][sd][task][0][:min_l] for sd in seeds])
        r = np.stack([results[m][sd][task][1][:min_l] for sd in seeds])
        c = np.stack([results[m][sd][task][2][:min_l] for sd in seeds])
        t = t.mean(0)
        for sd in range(len(seeds)):
            r[sd] = smooth(r[sd])
            c[sd] = smooth(c[sd])
        reward_std = r.std(0)
        axs[0, i].plot(t, r.mean(0), label=m, color=colors[j], linewidth=2.0)
        axs[0, i].fill_between(t, r.mean(0) - reward_std, r.mean(0) + reward_std, color=colors[j], alpha=0.1)
        axs[1, i].plot(t, c.mean(0), label=m, color=colors[j], linewidth=2.0)
        axs[1, i].fill_between(t, c.mean(0) - c.std(0), c.mean(0) + c.std(0), color=colors[j], alpha=0.1)
        xticks = ticker.MaxNLocator(6)
        axs[0, i].xaxis.set_major_locator(xticks)
        axs[1, i].xaxis.set_major_locator(xticks)
        axs[0, i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        axs[1, i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

axs[0, 0].set_ylabel("Mean Episodic Return", fontsize=12)
axs[1, 0].set_ylabel("Mistake rate", fontsize=12)
for i, task in enumerate(env_reward_bound.keys()):
    axs[0, i].set_title(tasks_names[task])


fig.text(0.5, 0.0, 'Training steps', ha='center', fontsize=12)

plt.tight_layout()

plt.savefig("results_single_agent.png", dpi=1000)
