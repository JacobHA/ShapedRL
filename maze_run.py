import gymnasium as gym
from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback


env_str = "FrozenLake-v1"
det=True
env = gym.make(env_str, is_slippery=not det)
eval_env = gym.make(env_str, is_slippery=not det)

hparams = {
    'batch_size': 64,
    'buffer_size': 50000,
    'exploration_final_eps': 0.04,
    'exploration_fraction': 0.16,
    'gamma': 0.99,
    'gradient_steps': 1,
    'policy_kwargs': {'net_arch': [32,32]},
    'learning_rate': 0.0023,
    'learning_starts': 1000,
    'target_update_interval': 10,
    'tau': 1.0,
    'train_freq': 1,
}


from stable_baselines3.common.monitor import Monitor

eval_callback = EvalCallback(eval_env, n_eval_episodes=20,
                log_path=f'./runs/',
                eval_freq=1_000,
                deterministic=True,
                verbose=1,
                render=True,)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", type=bool, default=False)
args = parser.parse_args()
do_shape = args.s

model = ShapedDQN("MlpPolicy", env, do_shape=do_shape, verbose=4, **hparams, device='cuda', tensorboard_log="./runs")
model.learn(40_000, log_interval=10, callback=eval_callback, tb_log_name=str(do_shape)+env_str+f'-det={det}')
