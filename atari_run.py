import json

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import vec_transpose, vec_frame_stack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

env_str = "PongNoFrameskip-v4"
# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env(env_str, n_envs=1, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)
with open(f"configs/{env_str}.json", 'r') as f:
    hparams = json.load(f)

model = ShapedDQN("CnnPolicy", env, do_shape=1, verbose=4, tensorboard_log="./runs", **hparams, device='cuda')
# log eval callbacks in the same tensorboard:
eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
eval_env = vec_frame_stack.VecFrameStack(eval_env, n_stack=4)
eval_env = vec_transpose.VecTransposeImage(eval_env)

eval_callback = EvalCallback(eval_env, n_eval_episodes=3,
                log_path=f'./runs/',
                eval_freq=5_000,
                deterministic=True)
model.learn(10_000_000, log_interval=10, callback=eval_callback)