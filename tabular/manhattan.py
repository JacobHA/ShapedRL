# First calculate the optimal V(s), then use alpha*V(s) for the potential function.
# Plot the integrated reward as a function of alpha.
import sys
sys.path.append('tabular/')
from utils import ModifiedFrozenLake, q_solver
from qlearner import QLearning
from dynamic_shaping import DynamicQLearning
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit

map_name = '10x10empty'
GAMMA = 0.99

# First solve the MDP:
env = ModifiedFrozenLake(map_name=map_name, slippery=0, min_reward=0, max_reward=1, step_penalization=0,
                         never_done=False, cyclic_mode=False)
env = TimeLimit(env, max_episode_steps=100)

Q,V,pi = q_solver(env, gamma=GAMMA, steps=1000000)

# Get the manhattan distance to goal in last state (corner):
manhattan_phi = np.zeros(V.shape)
for s in range(V.shape[0]):
    # Get the x,y coordinates of the state:
    x = s % 10
    y = s // 10
    # Get the x,y coordinates of the goal:
    x_goal = 9
    y_goal = 9
    manhattan_phi[s] = np.abs(x - x_goal) + np.abs(y - y_goal)

# display the manhattan distance:
plt.imshow(manhattan_phi.reshape((10,10)))

# run the policy:
optimal_reward = 0

for ep in range(10):
    s, _ = env.reset()
    done = False
    while not done:
        a = np.argmax(pi[s])
        s, r, term, trunc, _ = env.step(a)
        optimal_reward += r
        done = term or trunc
optimal_reward /= 10
print(f'Optimal reward: {optimal_reward}')


alphas = np.linspace(-0.0,0.001, 10)
alphas = np.logspace(-12, 0, 8)
alphas = [-1.0, -0.75, -0.5, -0.1, 0]#, 0.5, 1, 1.5]
# add zero
if 0 not in alphas:
    alphas = np.concatenate(([0], alphas))
alphas = np.sort(alphas)
# alphas = [-1.5, -1,-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]#, 4, 4.5, 5]
alpha_to_rwd_aucs = {}
alpha_to_rwd_stds = {}

EVAL_FREQ = 250
train_timesteps = 10000
def experiment(alpha, num_trials = 15):
    reward_curves = np.zeros((num_trials, train_timesteps // EVAL_FREQ))
    trial_rwds = np.zeros(num_trials)
    for trial in range(num_trials):
        # Now create the Q-learning agent:
        agent = QLearning(env, gamma=GAMMA, learning_rate=1, 
                          phi=alpha * manhattan_phi.flatten(),
                        save_data=False,
                        prefix=f'a={alpha}')

        agent.train(train_timesteps, eval_freq=EVAL_FREQ)
        trial_rwds[trial] = sum(agent.reward_over_time) / len(agent.reward_over_time)
        reward_curves[trial, :] = agent.reward_over_time
    # Add to the alpha_to_rwd_curve dict:
    return np.mean(trial_rwds, axis=0), np.std(trial_rwds,axis=0) / np.sqrt(num_trials), np.mean(reward_curves, axis=0), np.std(reward_curves, axis=0) / np.sqrt(num_trials)

from multiprocessing import Pool
with Pool(20) as p:
    means, stds, rwd_curves, rwd_curve_stds = zip(*p.map(experiment, alphas))
p.close()

plt.figure()
plt.title('Reward curves as a function of shaping coef')
time_values = np.arange(0, train_timesteps, EVAL_FREQ)
for alpha, curve, std in zip(alphas, rwd_curves, rwd_curve_stds):
    if alpha == 0:
        continue
    else:    
        plt.plot(time_values, curve, label=f'alpha={alpha}')
        plt.fill_between(time_values, curve - std, curve + std, alpha=0.2)

# Plot the alpha=0 curve:
plt.plot(time_values, rwd_curves[0], label='alpha=0', color='k', linestyle='--', linewidth=3)
plt.fill_between(time_values, rwd_curves[0] - rwd_curve_stds[0], rwd_curves[0] + rwd_curve_stds[0], alpha=0.2, color='k')
plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Oracle')

plt.legend()
plt.show()

for alpha, mean, std in zip(alphas, means, stds):
    alpha_to_rwd_aucs[alpha] = mean
    alpha_to_rwd_stds[alpha] = std
plt.figure()
plt.title('Performance as a function of shaping coef')
plt.ylabel('Average reward during training')
plt.xlabel('Shaping coef (eta)')
means = np.array([means]).squeeze()
stds = np.array([stds]).squeeze()
plt.plot(alphas, means, 'bo-')
plt.fill_between(alphas, means - stds, means + stds, color='b', alpha=0.2)

# Plot the optimal reward:
plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Oracle')
plt.xscale('log')

# put a vertical line at the alpha=0 (no shaping):
plt.axvline(x=0, color='k', linestyle='--', label='No shaping')
plt.axhline(y=alpha_to_rwd_aucs[0], color='k', linestyle='--')#, label='No shaping')
plt.legend()
plt.show()
###

plt.figure()
plt.title('Performance as a function of shaping coef')
plt.ylabel('Average reward during training')
plt.xlabel('Shaping coef (eta)')
means = np.array([means]).squeeze()
stds = np.array([stds]).squeeze()
plt.plot(alphas, means, 'bo-')
plt.fill_between(alphas, means - stds, means + stds, color='b', alpha=0.2)

# Plot the optimal reward:
plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Oracle')

# put a vertical line at the alpha=0 (no shaping):
plt.axvline(x=0, color='k', linestyle='--', label='No shaping')
plt.axhline(y=alpha_to_rwd_aucs[0], color='k', linestyle='--')#, label='No shaping')
plt.legend()
plt.show()


###
import os
os.makedirs('results', exist_ok=True)
# save the data:
np.save(f'results/{map_name}_alpha_to_rwd_aucs.npy', 
        {'alpha_to_rwd_aucs': alpha_to_rwd_aucs, 'alpha_to_rwd_stds': alpha_to_rwd_stds, 
         'reward_curves': rwd_curves, 'reward_curve_stds': rwd_curve_stds, 'alphas': alphas})