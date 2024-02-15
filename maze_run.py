import json

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from stable_baselines3.common.atari_wrappers import FireResetEnv

from wrappers import FrameStack, PermuteAtariObs
from ray.rllib.env.wrappers.atari_wrappers import is_atari
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.dqn import DQN

import argparse


atari_envs = {"PongNoFrameskip-v4"}

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=bool, default=True)
parser.add_argument("-t", type=bool, default=True)
parser.add_argument("-e", "--env", type=str, default="PongNoFrameskip-v4")
args = parser.parse_args()
do_shape = args.s
no_termination_val = args.t
env_str = args.env
env_kwargs = {}
det=False
if env_str == "FrozenLake-v1":
    env_kwargs['is_slippery'] = False
    det=True

if env_str in atari_envs:
    # env = make_atari_env(env_str, n_envs=1, seed=0)
    # eval_env = make_atari_env(env_str, n_envs=1, seed=0)
    # # Stack 4 frames
    # env = VecFrameStack(env, n_stack=4)
    # eval_env = VecFrameStack(eval_env, n_stack=4)
    framestack_k = 4
    frameskip = 4
    render=False
    permute_dims = False
    grayscale_obs = True
    env = gym.make(env_str, frameskip=frameskip)
    env = AtariPreprocessing(env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs,
                             grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
    if framestack_k:
        env = FrameStack(env, framestack_k)
    # permute dims for nature CNN in sb3
    if permute_dims:
        env = PermuteAtariObs(env)
    eval_env = gym.make(env_str, render_mode='human' if render else None, frameskip=frameskip)
    eval_env = AtariPreprocessing(eval_env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs,
                                  grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
    if framestack_k:
        eval_env = FrameStack(eval_env, framestack_k)
    if permute_dims:
        eval_env = PermuteAtariObs(eval_env)
    env = FireResetEnv(env)
    eval_env = FireResetEnv(eval_env)
    policy = "CnnPolicy"
else:
    env = gym.make(env_str, **env_kwargs)
    eval_env = gym.make(env_str, **env_kwargs)
    policy = "MlpPolicy"

with open(f"configs/{env_str}.json", 'r') as f:
    hparams = json.load(f)


total_timesteps = hparams.pop('total_timesteps')

eval_callback = EvalCallback(eval_env, n_eval_episodes=20,
                log_path=f'./runs/',
                eval_freq=1_000,
                deterministic=True,
                verbose=1,)

model = ShapedDQN(policy, env, do_shape=do_shape, no_termination_val=no_termination_val, verbose=4, **hparams, device='cuda', tensorboard_log="./runs")
model.learn(total_timesteps, log_interval=10, callback=eval_callback, tb_log_name=str(model)+env_str+f'-det={det}')
# model = DQN("MlpPolicy", env, verbose=4, **hparams, device='cuda', tensorboard_log="./runs")
# model.learn(40_000, log_interval=10, callback=eval_callback, tb_log_name="DQN"+env_str+f'-det={det}')