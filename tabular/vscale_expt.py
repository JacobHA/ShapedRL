# First calculate the optimal V(s), then use alpha*V(s) for the potential function.
# Plot the integrated reward as a function of alpha.
import sys
sys.path.append('tabular/')
from utils import ModifiedFrozenLake, q_solver
from qlearner import QLearning
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit

map_name = '7x7wall'


# First solve the MDP:
env = ModifiedFrozenLake(map_name=map_name, slippery=0)
env = TimeLimit(env, max_episode_steps=100)

Q,V,pi = q_solver(env, gamma=0.99, steps=10000)

# run the policy:
total_reward = 0

for ep in range(10):
    s, _ = env.reset()
    done = False
    while not done:
        a = np.argmax(pi[s])
        s, r, term, trunc, _ = env.step(a)
        total_reward += r
        done = term or trunc
total_reward /= 10
print(f'Optimal reward: {total_reward}')



alphas = np.linspace(-0.15, 0.15, 25)
alphas = [-1.5, -1,-0.5,0,0.5,1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
alpha_to_rwd_aucs = {}
alpha_to_rwd_stds = {}
def experiment(alpha, num_trials = 6):
    trial_rwds = np.zeros(num_trials)
    for trial in range(num_trials):
        # Now create the Q-learning agent:
        agent = QLearning(env, gamma=0.95, learning_rate=1, phi=alpha * V.flatten(), # + alpha * np.random.uniform(-1, 1, V.shape).flatten(),
                        save_data=False,
                        prefix=f'a={alpha}')
        agent.train(30000)
        trial_rwds[trial] = sum(agent.reward_over_time) / len(agent.reward_over_time)
    # alpha_to_rwd_aucs[alpha] = trial_avg / num_trials
    return np.mean(trial_rwds), np.std(trial_rwds)

from multiprocessing import Pool
with Pool(20) as p:
    means, stds = zip(*p.map(experiment, alphas))

for alpha, mean, std in zip(alphas, means, stds):
    alpha_to_rwd_aucs[alpha] = mean
    alpha_to_rwd_stds[alpha] = std

plt.figure()
plt.title('Performance as a function of shaping coef')
plt.plot(alphas, alpha_to_rwd_aucs.values(), 'bo-')
# Use a shaded error region:
plt.fill_between(alphas, [auc - std for auc, std in zip(alpha_to_rwd_aucs.values(), alpha_to_rwd_stds.values())],
                 [auc + std for auc, std in zip(alpha_to_rwd_aucs.values(), alpha_to_rwd_stds.values())],
                 color='b', alpha=0.2)
# Plot the optimal reward:
plt.axhline(y=total_reward, color='r', linestyle='--', label='Oracle')
# put a vertical line at the alpha=0 (no shaping):
plt.axvline(x=0, color='k', linestyle='--', label='No shaping')
plt.axhline(y=alpha_to_rwd_aucs[0], color='k', linestyle='--')#, label='No shaping')
plt.legend()
plt.show()