import gymnasium as gym
from algos.ShapedSAC import ShapedSAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb

env_str = 'HalfCheetah-v4'
env_str = 'Humanoid-v4'

env = gym.make(env_str)

hparams = {'learning_starts': 10000,
        #    'policy_kwargs':
        #    {'net_arch': [256, 256]},
}

shaping_mode = 'none'
use_dones = False
# sweep_id = '5ahwaszt'

wandb.init(project='Shaping', entity='jacobhadamczyk', sync_tensorboard=True)
wandb.log({'env_id': env_str, 'shaping_mode': shaping_mode, 'use_dones': use_dones})

model = ShapedSAC("MlpPolicy", env, shaping_mode=shaping_mode, 
                  verbose=4, tensorboard_log="./runs", 
                  **hparams, device='cuda')
eval_env = gym.make(env_str)
eval_callback = EvalCallback(Monitor(eval_env), n_eval_episodes=3,
                            log_path=f'./runs/',
                            eval_freq=5_000,
                            deterministic=True)

total_timesteps = 2e6
model.learn(total_timesteps, log_interval=100, 
            callback=eval_callback, tb_log_name=f"{shaping_mode}")
wandb.finish()