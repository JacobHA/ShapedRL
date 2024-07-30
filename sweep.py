import argparse
import os

import numpy as np

import wandb
import yaml
import copy
from run import run
from wandb_utils import sample_wandb_hyperparams

# use the global wandb log directory
log_dir = os.environ.get("WANDB_DIR", "./logs")

exp_to_config = {
    # all of the 62 atari environments + w/wo done mask + w/wo shaping
    "atari-full": "atari-full-sweep.yml",
    # all of the 62 atari environments + w/wo shaping
    "atari-shape": "atari-shape-sweep.yml",
    # 22 shorter atari environments, from https://arxiv.org/pdf/1903.00374
    "atari-scale-shorter": "atari-scale-shorter.yml",
    # 22 + 10 shorter atari envs
    "atari-scale-32": "atari-scale-32.yml",
    # 10 new other atari envs
    "atari-scale-10": "atari-scale-10.yml",
    # three of the atari environments
    "atari-mini": "atari-mini-sweep.yml",
    # pong only:
    "atari-pong": "atari-pong-sweep.yml",
    # 42 of the non v4 atari environments
    "atari-v5": "atari-v5-sweep.yml",
    # Pong only, sweeping scale parameter:
    "pong-eta": "pong-scale-sweep.yml",
    # All envs, eta sweep:
    "eta-sweep": "scale-sweep.yml",
    # classic control:
    "classic": "classic-sweep.yml",
    "classic-sql": "classic-sql.yml",
    "pong-target": "pong-target.yml",

}
int_hparams = {'batch_size', 'buffer_size', 'gradient_steps',
               'target_update_interval'}
device = None


def get_sweep_config(sweepcfg, default_config, project_name):
    cfg = default_config
    params = cfg['parameters']
    params.update(sweepcfg['parameters'])
    cfg.update(sweepcfg)
    cfg['parameters'] = params
    cfg['name'] = project_name
    return cfg


def wandb_train(local_cfg=None, n_hparam_runs=1):
    """:param local_cfg: pass config sweep if running locally to sample without wandb"""
    # make a consistent seed for reproducibility
    wandb_kwargs = {"project": project, "group": experiment_name, "dir": log_dir}
    if local_cfg is not None:
        local_cfg["controller"] = {'type': 'local'}
        sampled_params = sample_wandb_hyperparams(local_cfg["parameters"], int_hparams=int_hparams)
        print(f"locally sampled params: {sampled_params}")
        wandb_kwargs['config'] = sampled_params
    for i in range(n_hparam_runs):
        seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print(f"hparam run {i+1}/{n_hparam_runs}")
        with wandb.init(**wandb_kwargs, sync_tensorboard=True) as r:
            r.config.update({"seed": seed})
            config = wandb.config.as_dict()
            env_str = config.pop('env_id')
            total_timesteps = config.pop('total_timesteps')
            print(f"training {total_timesteps} steps on {env_str}")
            run(env_str, config, total_timesteps, log_freq=1000, device=device, log_dir=f'{log_dir}/{experiment_name}')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sweep", type=str, default=None)
    args.add_argument("--n_runs", type=int, default=1)
    args.add_argument("--n_hparam_runs", type=int, default=3)
    args.add_argument("--proj", type=str, default="shaping")
    args.add_argument("--local-wandb", type=bool, default=True)
    args.add_argument("--exp-name", type=str, default="atari-scale-32")
    args.add_argument("-d", "--device", type=str, default='cuda')
    args = args.parse_args()
    project = args.proj
    experiment_name = args.exp_name
    device = args.device
    # load the default config
    default_config = {'parameters': {}}
    with open("sweeps/atari-default.yml", "r") as f:
        default_config = yaml.load(f, yaml.SafeLoader)
    # load the experiment config
    with open(f"sweeps/{exp_to_config[experiment_name]}", "r") as f:
        expsweepcfg = yaml.load(f, yaml.SafeLoader)
    # combine the two
    sweepcfg = get_sweep_config(expsweepcfg, default_config, project)
    # generate a new sweep if one was not passed as an argument
    if args.sweep is None and not args.local_wandb:
        sweep_id = wandb.sweep(sweepcfg, project=project)
        print(f"created new sweep {sweep_id}")
        wandb.agent(sweep_id, project=args.proj,
                    count=args.n_runs, function=wandb_train)
    elif args.local_wandb:
        for i in range(args.n_runs):
            try:
                print(f"running local sweep {i}")
                wandb_train(local_cfg=copy.deepcopy(sweepcfg), n_hparam_runs=args.n_hparam_runs)
            except Exception as e:
                print(f"failed to run local sweep {i}")
                print(e)
    else:
        print(f"continuing sweep {args.sweep}")
        wandb.agent(args.sweep, project=args.proj,
                    count=args.n_runs, function=wandb_train)
