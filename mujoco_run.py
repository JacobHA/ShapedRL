import gymnasium as gym
from algos.ShapedSAC import ShapedSAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

env_str = 'HalfCheetah-v4'

env = gym.make(env_str)

hparams = {'learning_starts': 10000,
           'policy_kwargs':
           {'net_arch': [256, 256, 32]},
}

do_shape = 0
model = ShapedSAC("MlpPolicy", env, do_shape=do_shape, verbose=4, **hparams, device='cuda', tensorboard_log="./runs")
eval_callback = EvalCallback(Monitor(env), n_eval_episodes=3,
                            log_path=f'./runs/',
                            eval_freq=5_000,
                            deterministic=True)

model.learn(1_000_000, log_interval=10, callback=eval_callback, tb_log_name=str(do_shape)+env_str)