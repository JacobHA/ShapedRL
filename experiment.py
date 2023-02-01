import wandb
import gym
from stable_baselines3.dqn.dqn import DQN
from ShapedSAC import ShapedSAC
from wandb.integration.sb3 import WandbCallback
import argparse


def shaping(apply_shaping=True, env_name="FrozenLake-v1",
            learning_rate=0.0001,
            gamma=0.99, ent_coef='auto',
            is_slippery=False, train_steps=1000000):

    env = gym.make(env_name)

    with wandb.init(
            config={
                "env_name": env_name,
                "shaped": apply_shaping,
                "is_slippery": is_slippery,
            },
            sync_tensorboard=True
    ) as run:
        config = wandb.config

        cb = WandbCallback(
            gradient_save_freq=15000,
            model_save_path=f"models/{env_name}.pkl",
            verbose=2,
        )
        cb = None

        model = ShapedSAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            gamma=gamma,
            buffer_size=50000,
            ent_coef=ent_coef,
            shape=apply_shaping,
            tensorboard_log="./logs/",)

        model.learn(total_timesteps=train_steps, callback=cb)


if __name__ == "__main__":
    # Parse in the sweepid args
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweepid', type=str, default=None)
    args = parser.parse_args()
    sweepid = args.sweepid

    def wandb_func():
        # Based on https://openreview.net/pdf?id=HJjvxl-Cb
        shaping(apply_shaping=1, env_name='Reacher-v2',
                train_steps=1_000_000, gamma=0.99, ent_coef=1/100)

    wandb.agent(sweepid, function=wandb_func, count=1)
    wandb.finish()
