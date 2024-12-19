import gymnasium as gym
import numpy as np
from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback


env_str = "FrozenLake-v1"
env = gym.make(env_str)#, map_name='8x8')
net_arch = [64,64]#,256]
hparams = {
    'batch_size': 64,
    'buffer_size': 100000,
    'exploration_final_eps': 0.01,
    'exploration_fraction': 0.2,
    # 'gamma': 0.98,
    'gradient_steps': 1,
    'policy_kwargs': {'net_arch': net_arch},
    'learning_rate': 0.001,
    'learning_starts': 0,
    'target_update_interval': 100,
    'tau': 1.0,
    'train_freq': 1,
}


eval_callback = EvalCallback(env, n_eval_episodes=3,
                log_path=f'./runs/',
                eval_freq=500,
                deterministic=True)

# ETAS = np.linspace(1,3,30)[1:]
# ETAS = np.linspace(-1,1,30)
# ETAS = np.logspace(-5,-1, 30)[::-1]
ETAS = np.logspace(-3, -0.5, 30)
# if 0 not in ETAS:
#     ETAS = np.concatenate(([0.0], ETAS))
for num in range(10):
    for eta in ETAS:
        gamma = 1/(1+eta)
        model = ShapedDQN("MlpPolicy", env, do_shape=True, use_target=False, shape_scale=0.5, verbose=4, **hparams, 
                          device='cuda', 
                          gamma=gamma,
                          tensorboard_log="./gamma-runs")
        model.learn(10_000, log_interval=5, callback=eval_callback, tb_log_name=f"{env_str}_{num}-alpha={eta}")


        model = ShapedDQN("MlpPolicy", env, do_shape=True, use_target=False, shape_scale=0, verbose=4, **hparams, 
                          device='cuda', 
                          gamma=gamma,
                          tensorboard_log="./gamma-runs")
        model.learn(10_000, log_interval=5, callback=eval_callback, tb_log_name=f"{env_str}_{num}-NOalpha={eta}")