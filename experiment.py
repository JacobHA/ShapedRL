import random

import yaml

import wandb
import gym
from stable_baselines3.dqn.dqn import DQN
from ShapedSAC import ShapedSAC
from wandb.integration.sb3 import WandbCallback
import argparse


PROJ = "SAC-mujoco"
ENTITY = "reward-shapers"


def shaping(env_name="FrozenLake-v1",
            gamma=0.99, ent_coef='auto',
            train_steps=1000000):

    env = gym.make(env_name)

    with wandb.init(
            sync_tensorboard=True,
            project=PROJ) as run:
        config = wandb.config
        model = ShapedSAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=gamma,
            buffer_size=50000,
            ent_coef=ent_coef,
            shaped=config.shaped,
            tensorboard_log="./logs/",)

        model.learn(total_timesteps=train_steps, callback=None)


def make_new_sweep(yaml_file='sac_sweep.yml'):
    # load yaml config
    # Concatenate with sweep_configs folder:
    filename = f'sweep_configs/{yaml_file}'
    with open(filename, "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project=PROJ, entity=ENTITY)
    return sweep_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.sweep_id is None:
        sweep_id = make_new_sweep()
    else:
        sweep_id = args.sweep_id

    def wandb_func():
        # Hyperparams were based on https://openreview.net/pdf?id=HJjvxl-Cb
        shaping(env_name='Reacher-v2',
                train_steps=150_000, gamma=0.99, ent_coef=1/100)

    wandb.agent(sweep_id, function=wandb_func, count=args.count)
