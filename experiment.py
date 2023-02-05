import wandb
import os
import argparse
from config import experiment
from utils import make_new_sweep, configured_model
PROJ = experiment['PROJ']
model = experiment['ALGO']
env_name = experiment['ENV']
sweep_id = experiment['SWEEPID']
count = experiment['COUNT']
train_steps = experiment['TRAIN_STEPS']


def shaping(train_steps=1_000_000):

    with wandb.init(
            sync_tensorboard=True,
            project=PROJ) as run:

        model = configured_model(wandb.config)

        model.learn(total_timesteps=train_steps, callback=None)


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
        shaping(train_steps=train_steps)

    wandb.agent(sweep_id, function=wandb_func, count=args.count)
