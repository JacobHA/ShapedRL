# test the model saved at {env_str} to ensure it achieves maximal performance
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import wandb
from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import vec_transpose, vec_frame_stack
# There already exists an environment generator that will make and wrap atari environments correctly.

env_str = 'PongNoFrameskip-v4'
env = make_atari_env(env_str, n_envs=1, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)
env = vec_transpose.VecTransposeImage(env)

model_loc = f"{env_str}.pth"

# Load the model:
# model = ShapedDQN.load(model_loc, env=env)
import torch
qfunc = torch.load(model_loc)#, map_location='cpu')
# Run the model in the environment:
n_episodes = 10
import numpy as np
ep_rewards = np.zeros(n_episodes)
for ep in range(n_episodes):
    obs= env.reset()
    done = False
    ep_reward = 0
    while not done:
        # action, _states = model.predict(obs, deterministic=True)
        obs = torch.tensor(obs, dtype=torch.float32).squeeze(0)
        action = qfunc(obs).argmax().item()
        # print(action)
        obs, reward, done, info = env.step([action])
        # done = term or trunc
        # env.render()
        ep_reward += reward
    ep_rewards[ep] = ep_reward
    print(f"Episode {ep} reward: {ep_reward}")

env.close()
print(f"Mean reward: {ep_rewards.mean()}")
print(f"Std reward: {ep_rewards.std()}")