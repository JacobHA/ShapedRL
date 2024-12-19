# First calculate the optimal V(s), then use alpha*V(s) for the potential function.
# Plot the integrated reward as a function of alpha.
import sys
import os

sys.path.append('tabular/')
from utils import ModifiedFrozenLake, q_solver
from experiment_utils import plot_data, run_experiment, greedy_pi_reward
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit

map_name = '10x10empty'
GAMMA = 0.99

# First solve the MDP:
env = ModifiedFrozenLake(map_name=map_name, slippery=1, min_reward=0, max_reward=1, step_penalization=0,
                         never_done=False, cyclic_mode=False)
env = TimeLimit(env, max_episode_steps=5000)

Q,V,pi = q_solver(env, gamma=GAMMA, steps=1000000)

optimal_reward = greedy_pi_reward(env, pi, num_episodes=10)

alphas = np.linspace(-1.0, 1.0, 10)
# alphas = np.logspace(-12, 0, 8)
# alphas = [-0.05, -0.25, -0.1, 0]#, 0.5, 1, 1.5]
# add zero
if 0 not in alphas:
    alphas = np.concatenate(([0], alphas))
alphas = np.sort(alphas)



EVAL_FREQ = 250
train_timesteps = 10000

means, stds, rwd_curves, rwd_curve_stds = run_experiment(env, 
                                                         alphas, 
                                                         V, 
                                                         train_timesteps=train_timesteps, 
                                                         eval_freq=EVAL_FREQ, 
                                                         n_processes=30,
                                                         num_trials=15,
                                                         learning_rate=1)
plot_data(alphas, rwd_curves, rwd_curve_stds, means, stds, train_timesteps=train_timesteps, optimal_reward=optimal_reward, eval_freq=EVAL_FREQ)
# os.makedirs('results', exist_ok=True)
# # save the data:
# np.save(f'results/{map_name}_alpha_to_rwd_aucs.npy', 
#         {'alpha_to_rwd_aucs': alpha_to_rwd_aucs, 'alpha_to_rwd_stds': alpha_to_rwd_stds, 
#          'reward_curves': rwd_curves, 'reward_curve_stds': rwd_curve_stds, 'alphas': alphas})