# First calculate the optimal V(s), then use eta*V(s) for the potential function.
# Plot the integrated reward as a function of eta.
import sys
sys.path.append('tabular/')
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from utils import ModifiedFrozenLake, q_solver
from experiment_utils import plot_data, run_bsrs_experiment, greedy_pi_reward

map_name = '4x4empty'
map_name = '8x8empty'
GAMMA = 0.99

# First solve the MDP:
env = ModifiedFrozenLake(map_name=map_name, slippery=1, min_reward=0, max_reward=1, step_penalization=0,
                         never_done=False, cyclic_mode=False)
env = TimeLimit(env, max_episode_steps=5000)
Q,V,pi = q_solver(env, gamma=GAMMA, steps=1000000)

optimal_reward = greedy_pi_reward(env, pi, num_episodes=10)

etas = np.linspace((1-GAMMA)/(GAMMA+1), (-GAMMA)/(1+GAMMA), 24)
# etas = np.linspace(-0.05, 0.5, 6)
etas = [0, 0.5, 1, 2, 3, 4, 5, 8, 10]
# add the zero eta if not there:
if 0 not in etas:
    etas = np.concatenate(([0], etas))
# sort the etas:
etas = np.sort(etas)
train_timesteps = 5000
eval_freq = 250
means, stds, rwd_curves, rwd_curve_stds = run_bsrs_experiment(env,
                                                         etas,
                                                         train_timesteps=train_timesteps,
                                                         eval_freq=eval_freq,
                                                         n_processes=30,
                                                         gamma=GAMMA,
                                                         learning_rate=1,
                                                         num_trials=20
                                                        )
# convert to dict:
# eta_to_q = {eta: (q_mean, q_std) for eta, q_mean, q_std in zip(etas, q_means, q_stds)}
# eta_to_rwd = {eta: (r_mean, r_std) for eta, r_mean, r_std in zip(etas, r_means, r_stds)}
plot_data(etas, rwd_curves, rwd_curve_stds, means, stds, train_timesteps=train_timesteps, optimal_reward=optimal_reward, eval_freq=eval_freq)

# Do the same for the mean Q values:
# plt.figure()
# plt.title('Mean Q as a function of shaping coef')
# norm = eta_to_q[0][0]
# plt.plot(etas, q_means/norm, 'bo-')
# Use a shaded error region:
# plt.fill_between(etas, [auc/norm - std/norm for auc, std in zip(q_means, q_stds)],
#                  [auc/norm + std/norm for auc, std in zip(q_means, q_stds)],
#                  color='b', alpha=0.2)
# Draw a line y=1+eta:
# plt.plot(etas, (1 + 0.97*etas)**(-1), 'r--')
# plt.plot(etas, (1 + etas)**(-1), 'g--')

plt.ylim(0, 100)
plt.show()
