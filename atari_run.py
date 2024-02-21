import json

import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import wandb
from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import vec_transpose, vec_frame_stack
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

# env_str = "PongNoFrameskip-v4"
env_str = "SkiingNoFrameskip-v4"
# There already exists an environment generator that will make and wrap atari environments correctly.

# env_str = "BreakoutNoFrameskip-v4"
from all_atari_envs import atari_env_strs
# idx = 9
# env_str = atari_env_strs[idx]
# env_str = "MontezumaRevengeNoFrameskip-v4"
env = make_atari_env(env_str, n_envs=1, seed=42)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)
hparams = {
    "buffer_size": 100_000,
    "batch_size": 32,
    "gamma": 0.99,
    "learning_rate": 1e-4,
    "target_update_interval": 1000,
    "learning_starts": 100000,
    "train_freq": 4,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.01,
    # "frame_stack": 4,
    "gradient_steps": 1,
    "train_freq": 4,    
}

shaping_mode = 'online'
use_dones = True

# log eval callbacks in the same tensorboard:
eval_env = make_atari_env(env_str, n_envs=1, seed=0)
eval_env = vec_frame_stack.VecFrameStack(eval_env, n_stack=4)
eval_env = vec_transpose.VecTransposeImage(eval_env)

eval_callback = EvalCallback(eval_env, n_eval_episodes=5,
                log_path=f'./runs/',
                eval_freq=50_000,
                deterministic=True)


# wandb.init(project='bs-rs', entity='jacobhadamczyk', sync_tensorboard=True)
# wandb.log({'env_id': env_str, 'shaping_mode': shaping_mode, 'use_dones': use_dones})

model = ShapedDQN("CnnPolicy", env, do_shape=True, no_done_mask=True,
                  use_oracle=False,
                  verbose=4, tensorboard_log="./runs", **hparams, device='cuda')
total_timesteps = 10_000_000
model.learn(total_timesteps, log_interval=100, callback=eval_callback, tb_log_name=f"{shaping_mode}")

# wandb.finish()

# model.save_final_model(f"{env_str}.pth")