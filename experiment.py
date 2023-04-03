import wandb
import gym
from ShapedDQN import ShapedDQN
from ShapedSAC import ShapedSAC
from wandb.integration.sb3 import WandbCallback
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    # There are three methods of getting a sweep id:
    # 1. If you have a sweep id, you can pass it in as an argument
    # 2. If you have a sweep id, you can set it in config.py
    # 3. If you don't have a sweep id, automatically make a new one
    if sweep_id is None:
        if args.sweep_id is not None:
            sweep_id = args.sweep_id
        elif sweep_id is None:
            sweep_id = make_new_sweep()
            print(f"New sweep id:\n{sweep_id}")
            print("Set this variable in config or " +
                  "pass it as an argument to experiment.py.\nExiting...")
            exit()

    SWEEP_ID = f"reward-shapers/{PROJ}/{sweep_id}"

    def wandb_func():
        shaping(train_steps=train_steps)

    wandb.agent(SWEEP_ID, function=wandb_func, count=args.count)
