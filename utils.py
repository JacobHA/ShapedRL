import os
import random
import time

import numpy as np
import torch
import wandb
import gymnasium as gym
from stable_baselines3.common.logger import configure


def sample_wandb_hyperparams(params, int_hparams=None):
    sampled = {}
    for k, v in params.items():
        if 'values' in v:
            sampled[k] = random.choice(v['values'])
        elif 'distribution' in v:
            if v['distribution'] == 'uniform' or v['distribution'] == 'uniform_values':
                sampled[k] = random.uniform(v['min'], v['max'])
            elif v['distribution'] == 'normal':
                sampled[k] = random.normalvariate(v['mean'], v['std'])
            elif v['distribution'] == 'log_uniform_values':
                emin, emax = np.log(v['max']), np.log(v['min'])
                sample = np.exp(random.uniform(emin, emax))
                sampled[k] = sample
            else:
                raise NotImplementedError(f"Distribution sampling not implemented for {v['distribution']}, required by {k}")
        else:
            raise NotImplementedError(f"Unsupported hparam sampling range format in {k}")
        if k in int_hparams:
            sampled[k] = int(sampled[k])
    return sampled



def log_class_vars(self, logger, params, use_wandb=False):
    for key, value in params.items():
        value = self.__dict__[value]
        # first check if value is a tensor:
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.record(key, value)
        if use_wandb:
            wandb.log({key: value})



def env_id_to_envs(env_id, render):
    if isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
        return env, eval_env
    
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id, render_mode='human' if render else None)
        return env, eval_env


def logger_at_folder(log_dir=None, algo_name=None):
    # ensure no _ in algo_name:
    if '_' in algo_name:
        print("WARNING: '_' not allowed in algo_name (used for indexing). Replacing with '-'.")
    algo_name = algo_name.replace('_', '-')
    # Generate a logger object at the specified folder:
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        files = os.listdir(log_dir)
        # Get the number of existing "LogU" directories:
        # another run may be creating a folder:
        time.sleep(0.5)
        num = len([int(f.split('_')[1]) for f in files if algo_name in f]) + 1
        tmp_path = f"{log_dir}/{algo_name}_{num}"

        # If the path exists already, increment the number:
        while os.path.exists(tmp_path):
            # another run may be creating a folder:
            time.sleep(0.5)
            num += 1
            tmp_path = f"{log_dir}/{algo_name}_{num}"

        logger = configure(tmp_path, ["stdout"])#, "tensorboard"])
    else:
        # print the logs to stdout:
        logger = configure(format_strings=["stdout"])

    return logger
