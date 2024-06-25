# First calculate the optimal V(s), then use eta*V(s) for the potential function.
# Plot the integrated reward as a function of eta.
import sys
sys.path.append('tabular/')
from dynamic_shaping import DynamicQLearning
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from utils import ModifiedFrozenLake

map_name = '7x7wall'
env = ModifiedFrozenLake(map_name=map_name, slippery=0)
env = TimeLimit(env, max_episode_steps=500)

etas = np.linspace(-0.95, 0.5, 24)
# etas = np.linspace(-0.1, 0.05, 6)

# add the zero eta if not there:
if 0 not in etas:
    etas = np.concatenate(([0], etas))
# sort the etas:
etas = np.sort(etas)

def experiment(eta, num_trials = 5):
    trial_rwds = np.zeros(num_trials)
    trial_meanQ = np.zeros(num_trials)
    for trial in range(num_trials):
        # Now create the Q-learning agent:
        agent = DynamicQLearning(env, gamma=0.97, learning_rate=0.9, eta=eta,
                        save_data=False,
                        prefix=f'a={eta}')
        agent.train(10_000)
        trial_meanQ[trial] = np.mean(np.abs(agent.V_from_Q(agent.Q)))
        trial_rwds[trial] = sum(agent.reward_over_time) / len(agent.reward_over_time)

    return np.mean(trial_meanQ), np.std(trial_meanQ), np.mean(trial_rwds), np.std(trial_rwds)

from multiprocessing import Pool
with Pool(16) as p:
    q_means, q_stds, r_means, r_stds = zip(*p.map(experiment, etas))


# convert to dict:
eta_to_q = {eta: (q_mean, q_std) for eta, q_mean, q_std in zip(etas, q_means, q_stds)}
eta_to_rwd = {eta: (r_mean, r_std) for eta, r_mean, r_std in zip(etas, r_means, r_stds)}


plt.figure()
plt.title('Mean reward as a function of shaping coef')
plt.plot(etas, r_means, 'bo-')
# Use a shaded error region:
plt.fill_between(etas, [auc - std for auc, std in zip(r_means, r_stds)],
                 [auc + std for auc, std in zip(r_means, r_stds)],
                 color='b', alpha=0.2)
# Plot the optimal reward:
# plt.axhline(y=total_reward, color='r', linestyle='--', label='Oracle')
# put a vertical line at the eta=0 (no shaping):
plt.axvline(x=0, color='k', linestyle='--', label='No shaping')
plt.axhline(y=eta_to_rwd[0][0], color='k', linestyle='--')
plt.legend()
plt.show()

# Do the same for the mean Q values:
plt.figure()
plt.title('Mean Q as a function of shaping coef')
norm = eta_to_q[0][0]
plt.plot(etas, q_means/norm, 'bo-')
# Use a shaded error region:
plt.fill_between(etas, [auc/norm - std/norm for auc, std in zip(q_means, q_stds)],
                 [auc/norm + std/norm for auc, std in zip(q_means, q_stds)],
                 color='b', alpha=0.2)
# Draw a line y=1+eta:
# plt.plot(etas, (1 + 0.97*etas)**(-1), 'r--')
plt.plot(etas, (1 + etas)**(-1), 'g--')

plt.ylim(0, 100)
plt.show()
