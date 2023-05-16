import itertools
import sys
import time

import highway

from iteration_utilities import grouper
from multiprocessing import Process

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return [iter(iterable)]*n

def tune_hyper_parameters(number_of_processes = 3):
    K_epochss = [10, 50, 100]
    eps_clips = [0.1, 0.2]
    gammas = [0.9, 0.95, 0.99]
    lr_actors = [1e-4, 5e-4, 1e-3]
    lr_critics = [1e-4, 5e-4, 1e-3]
    max_training_timesteps = int(2e5)
    seed = 0
    algo = 'PPO'
    configs = grouper(itertools.product(K_epochss, eps_clips, gammas, lr_actors, lr_critics),number_of_processes)
    for n_configs in configs:
        processes = []
        for config in n_configs:
            K_epochs, eps_clip, gamma, lr_actor, lr_critic = config
            args = f"--algo {algo} --seed {seed} --max_training_timesteps {max_training_timesteps} --lr_critic {lr_critic} --lr_actor {lr_actor} --gamma {gamma} --eps_clip {eps_clip} --K_epochs {K_epochs}"
            p = Process(target=highway.train, args = [args.split()])
            processes.append(p)
            p.start()
            time.sleep(2)
        for p in processes:
            p.join()
    #     sys.argv = args.split()
    #     highway.train()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        tune_hyper_parameters(int(sys.argv[1]))
    else:
        tune_hyper_parameters()