import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import vec_transpose, vec_frame_stack
# There already exists an environment generator that will make and wrap atari environments correctly.

env_str = "BreakoutNoFrameskip-v4"
env = make_atari_env(env_str, n_envs=1, seed=0)
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
model = ShapedDQN("CnnPolicy", env, shaping_mode=shaping_mode, verbose=4, tensorboard_log="./runs", **hparams, device='cuda')
# log eval callbacks in the same tensorboard:
eval_env = make_atari_env(env_str, n_envs=1, seed=0)
eval_env = vec_frame_stack.VecFrameStack(eval_env, n_stack=4)
eval_env = vec_transpose.VecTransposeImage(eval_env)

eval_callback = EvalCallback(eval_env, n_eval_episodes=3,
                log_path=f'./runs/',
                eval_freq=10_000,
                deterministic=True)
model.learn(10_000_000, log_interval=100, callback=eval_callback, tb_log_name=f"{shaping_mode}")