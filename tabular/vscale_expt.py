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



alphas = np.linspace(-0.01, 0.03, 8)
# add zero
if 0 not in alphas:
    alphas = np.concatenate(([0], alphas))
alphas = np.sort(alphas)
# alphas = [-1.5, -1,-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]#, 4, 4.5, 5]
alpha_to_rwd_aucs = {}
alpha_to_rwd_stds = {}


def experiment(alpha, num_trials = 20):
    train_timesteps = 3000
    reward_curves = np.zeros((num_trials, train_timesteps // 100))
    trial_rwds = np.zeros(num_trials)
    for trial in range(num_trials):
        # Now create the Q-learning agent:
        agent = QLearning(env, gamma=0.96, learning_rate=1, phi=alpha * V.flatten(), # + alpha * np.random.uniform(-1, 1, V.shape).flatten(),
                        save_data=False,
                        prefix=f'a={alpha}')
        agent.train(train_timesteps)
        trial_rwds[trial] = sum(agent.reward_over_time) / len(agent.reward_over_time)
        reward_curves[trial, :] = agent.reward_over_time
        print(reward_curves)
    # Add to the alpha_to_rwd_curve dict:
    return np.mean(trial_rwds), np.std(trial_rwds), np.mean(reward_curves, axis=0), np.std(reward_curves, axis=0)

from multiprocessing import Pool
with Pool(20) as p:
    means, stds, rwd_curves, rwd_curve_stds = zip(*p.map(experiment, alphas))
p.close()

plt.figure()
plt.title('Reward curves as a function of shaping coef')
for alpha, curve, std in zip(alphas, rwd_curves, rwd_curve_stds):
    plt.plot(np.arange(0, 3000, 100), curve, label=f'alpha={alpha}')
    plt.fill_between(np.arange(0, 3000, 100), curve - std, curve + std, alpha=0.2)

plt.legend()
plt.show()

for alpha, mean, std in zip(alphas, means, stds):
    alpha_to_rwd_aucs[alpha] = mean
    alpha_to_rwd_stds[alpha] = std
plt.figure()
plt.title('Performance as a function of shaping coef')
means = np.array([means]).squeeze()
stds = np.array([stds]).squeeze()
plt.plot(alphas, means, 'bo-')
plt.fill_between(alphas, means - stds, means + stds, color='b', alpha=0.2)

# Plot the optimal reward:
plt.axhline(y=total_reward, color='r', linestyle='--', label='Oracle')
# put a vertical line at the alpha=0 (no shaping):
plt.axvline(x=0, color='k', linestyle='--', label='No shaping')
plt.axhline(y=alpha_to_rwd_aucs[0], color='k', linestyle='--')#, label='No shaping')
plt.legend()
plt.show()