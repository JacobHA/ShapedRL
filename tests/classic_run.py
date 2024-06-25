import gymnasium as gym
from algos.ShapedDQN import ShapedDQN
from stable_baselines3.common.callbacks import EvalCallback


env_str = "CartPole-v1"
env = gym.make(env_str)
net_arch = [64,64,64,64]#,256]
hparams = {
    'batch_size': 64,
    'buffer_size': 100000,
    'exploration_final_eps': 0.04,
    'exploration_fraction': 0.16,
    'gamma': 0.99,
    'gradient_steps': 128,
    'policy_kwargs': {'net_arch': net_arch},
    'learning_rate': 0.001,
    'learning_starts': 1000,
    'target_update_interval': 1000,
    'tau': 1.0,
    'train_freq': 256,
}


eval_callback = EvalCallback(env, n_eval_episodes=3,
                log_path=f'./runs/',
                eval_freq=5_000,
                deterministic=True)

model = ShapedDQN("MlpPolicy", env, do_shape=0, verbose=4, **hparams, device='cuda', tensorboard_log="./runs")
model.learn(100_000, log_interval=10, callback=eval_callback, tb_log_name=f"{env_str}-{net_arch}")
