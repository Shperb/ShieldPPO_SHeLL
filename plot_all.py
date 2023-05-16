import os

import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


seeds = [
    10719,
    #39112,
    #69235,
    93759,
    49230,
]

methods = ["PPO", "ShieldPPO"]

results = {}

env_reward_bound = {}

for m in methods:
    results[m] = {}
    for seed in seeds:
        results[m][seed] = {}
        filename = f"models2/{m}_{seed}/stats.log"
        res = torch.load(filename)
        time_steps, rewards, costs, tasks = res
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
        b[i] = x[max(0, i-50):min(n, i+50)].mean()
    return b

for j,m in enumerate(methods):
    for i, task in enumerate(env_reward_bound.keys()):
        min_l = np.min([len(x) for x in [results[m][sd][task][0] for sd in seeds]])
        t = np.stack([results[m][sd][task][0][:min_l] for sd in seeds])
        r = np.stack([results[m][sd][task][1][:min_l] for sd in seeds])
        c = np.stack([results[m][sd][task][2][:min_l] for sd in seeds])
        t = t.mean(0)

        low, up = env_reward_bound[task]

        r = (r - low) / (up - low + 1e-8)
        #kr = smooth.NonParamRegression(t, r, method=npr_methods.SpatialAverage())
        #kc = smooth.NonParamRegression(t, c, method=npr_methods.SpatialAverage())
        #kr.fit()
        #kc.fit()
        #for sd in seeds:
        for sd in range(len(seeds)):
            r[sd] = smooth(r[sd])
            c[sd] = smooth(c[sd])
        axs[0, i].plot(t, r.mean(0), label=m, color=colors[j], linewidth=2.0)
        axs[0, i].fill_between(t, r.mean(0)-r.std(0), r.mean(0)+r.std(0), color=colors[j], alpha=0.1)
        axs[1, i].plot(t, c.mean(0), label=m, color=colors[j], linewidth=2.0)
        axs[1, i].fill_between(t, c.mean(0)-c.std(0), c.mean(0)+c.std(0), color=colors[j], alpha=0.1)

axs[0, 0].set_ylabel("Reward", fontsize=10)
axs[1, 0].set_ylabel("Cost", fontsize=10)
for i, task in enumerate(env_reward_bound.keys()):
    tttt = task.find("Env")
    axs[0,i].set_title(task[:tttt+3])

axs[0, 0].legend(fontsize=10)
fig.text(0.5, 0.0, 'Training steps', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("tmp.png", dpi=500)
    

