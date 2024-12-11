import gymnasium as gym
from algos.ShapedTD3 import ShapedTD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb

env_str = 'HalfCheetah-v4'
env_str = 'Humanoid-v4'
env_str = 'Pendulum-v1'

env = gym.make(env_str)

hparams = {
    # 'learning_starts': 10000,
    #    'policy_kwargs':
    #    {'net_arch': [256, 256]},
}

shaping_mode = True
use_dones = False
# sweep_id = '5ahwaszt'
# wandb.tensorboard.patch(root_logdir='./runs')

for eta in [0, 4, 5, 7, 2.5]:
    for _ in range(3):
        wandb.init(project='MJ-Shaping', entity='jacobhadamczyk', sync_tensorboard=True)
        wandb.log({'env_id': env_str, 'shaping_mode': shaping_mode, 'use_dones': use_dones})

        wandb.log({'shape_scale': eta})
        model = ShapedTD3("MlpPolicy", env, shaping_mode=shaping_mode, shape_scale=eta,
                        verbose=4, tensorboard_log="./runs",
                        **hparams, device='cuda')

        eval_env = gym.make(env_str)
        eval_callback = EvalCallback(Monitor(eval_env), n_eval_episodes=2,
                                    log_path=f'./runs/',
                                    eval_freq=500,
                                    deterministic=True)

        total_timesteps = 10_000
        model.learn(total_timesteps, log_interval=100,
                    callback=eval_callback, tb_log_name=f"{shaping_mode}")

        wandb.finish()