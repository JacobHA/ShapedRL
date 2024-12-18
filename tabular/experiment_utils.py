import numpy as np
from multiprocessing import Pool
from qlearner import QLearning
from dynamic_shaping import DynamicQLearning
from soft_dynamic_shaping import DynamicSoftQLearning
import matplotlib.pyplot as plt

def single_experiment(env,
                      scalar, 
                      base_potential,
                      gamma=0.99,
                      num_trials = 15, 
                      train_timesteps=10_000, 
                      learning_rate=1,
                      eval_freq=1000):
    reward_curves = np.zeros((num_trials, train_timesteps // eval_freq))
    trial_rwds = np.zeros(num_trials)
    for trial in range(num_trials):
        # Now create the Q-learning agent:
        agent = QLearning(env, gamma=gamma, learning_rate=learning_rate, 
                          phi=scalar * base_potential,
                          save_data=False,
                          prefix=f'a={scalar}')

        agent.train(train_timesteps, eval_freq=eval_freq)
        trial_rwds[trial] = sum(agent.reward_over_time) / len(agent.reward_over_time)
        reward_curves[trial, :] = agent.reward_over_time

    return (np.mean(trial_rwds, axis=0), 
            np.std(trial_rwds, axis=0) / np.sqrt(num_trials), 
            np.mean(reward_curves, axis=0), 
            np.std(reward_curves, axis=0) / np.sqrt(num_trials)
            )

def run_experiment(env,
                   scalars, 
                   base_potential, 
                   train_timesteps=10_000, 
                   eval_freq=1000, 
                   num_trials=15,
                   gamma=0.99,
                   learning_rate=1,
                   n_processes=1):
    # try to flatten potential if it is not already:
    base_potential = base_potential.flatten()

    with Pool(n_processes) as p:
        results = p.starmap(single_experiment, [(env, scalar, base_potential, gamma, num_trials, train_timesteps, learning_rate, eval_freq) for scalar in scalars])
        # results = p.map(configured_experiment, scalars)
    # 
    means, stds, rwd_curves, rwd_curve_stds = zip(*results)

    return means, stds, rwd_curves, rwd_curve_stds

def greedy_pi_reward(env, pi, num_episodes=10):
    # run the policy:
    optimal_reward = 0

    for ep in range(num_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = np.argmax(pi[s])
            s, r, term, trunc, _ = env.step(a)
            optimal_reward += r
            done = term or trunc
    optimal_reward /= num_episodes
    print(f'Optimal reward: {optimal_reward}')

    return optimal_reward


def single_bsrs_experiment(env,
                            scalar, 
                            gamma=0.99,
                            num_trials = 15, 
                            train_timesteps=10_000, 
                            learning_rate=1,
                            eval_freq=1000,
                            soft=False):
    reward_curves = np.zeros((num_trials, train_timesteps // eval_freq))
    trial_rwds = np.zeros(num_trials)
    for trial in range(num_trials):
        # Now create the Q-learning agent:
        learner = DynamicQLearning if soft else DynamicSoftQLearning
        agent = learner(env, gamma=gamma, learning_rate=learning_rate, 
                          eta=scalar,
                          save_data=False,
                          prefix=f'a={scalar}')

        agent.train(train_timesteps, eval_freq=eval_freq)
        trial_rwds[trial] = sum(agent.reward_over_time) / len(agent.reward_over_time)
        reward_curves[trial, :] = agent.reward_over_time

    return (np.mean(trial_rwds, axis=0), 
            np.std(trial_rwds, axis=0) / np.sqrt(num_trials), 
            np.mean(reward_curves, axis=0), 
            np.std(reward_curves, axis=0) / np.sqrt(num_trials)
            )

def run_bsrs_experiment(env,
                        scalars, 
                        train_timesteps=10_000, 
                        eval_freq=1000, 
                        num_trials=15,
                        gamma=0.99,
                        learning_rate=1,
                        n_processes=1):

    with Pool(n_processes) as p:
        results = p.starmap(single_bsrs_experiment, [(env, scalar, gamma, num_trials, train_timesteps, learning_rate, eval_freq) for scalar in scalars])
        # results = p.map(configured_experiment, scalars)
    # 
    means, stds, rwd_curves, rwd_curve_stds = zip(*results)

    return means, stds, rwd_curves, rwd_curve_stds

def plot_data(scalars,
              rwd_curves,
              rwd_curve_stds,
              means,
              stds,
              train_timesteps=10_000,
              eval_freq=1000,
              optimal_reward=None,          
    ):
    scalar_to_rwd_aucs = {}
    scalar_to_rwd_stds = {}

    plt.figure()
    plt.title('Reward curves as a function of shaping coef')
    time_values = np.arange(0, train_timesteps, eval_freq)
    for alpha, curve, std in zip(scalars, rwd_curves, rwd_curve_stds):
        if alpha == 0:
            plt.plot(time_values, curve, label='alpha=0', color='k', linestyle='--', linewidth=3)
            plt.fill_between(time_values, curve - std, curve + std, alpha=0.2, color='k')
        else:    
            plt.plot(time_values, curve, label=f'alpha={alpha}')
            plt.fill_between(time_values, curve - std, curve + std, alpha=0.2)

    # Plot the alpha=0 curve:

    plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Oracle')

    plt.legend()
    plt.show()

    for scalar, mean, std in zip(scalars, means, stds):
        scalar_to_rwd_aucs[scalar] = mean
        scalar_to_rwd_stds[scalar] = std
    plt.figure()
    plt.title('Performance as a function of shaping coef')
    plt.ylabel('Average reward during training')
    plt.xlabel('Shaping coef (eta)')
    means = np.array([means]).squeeze()
    stds = np.array([stds]).squeeze()
    plt.plot(scalars, means, 'bo-')
    plt.fill_between(scalars, means - stds, means + stds, color='b', alpha=0.2)

    # Plot the optimal reward:
    plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Oracle')
    plt.xscale('log')

    # put a vertical line at the alpha=0 (no shaping):
    plt.axvline(x=0, color='k', linestyle='--', label='No shaping')
    plt.axhline(y=scalar_to_rwd_aucs[0], color='k', linestyle='--')#, label='No shaping')
    plt.legend()
    plt.show()
    ###

    plt.figure()
    plt.title('Performance as a function of shaping coef')
    plt.ylabel('Average reward during training')
    plt.xlabel('Shaping coef (eta)')
    means = np.array([means]).squeeze()
    stds = np.array([stds]).squeeze()
    plt.plot(scalars, means, 'bo-')
    plt.fill_between(scalars, means - stds, means + stds, color='b', alpha=0.2)

    # Plot the optimal reward:
    plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Oracle')

    # put a vertical line at the alpha=0 (no shaping):
    plt.axvline(x=0, color='k', linestyle='--', label='No shaping')
    plt.axhline(y=scalar_to_rwd_aucs[0], color='k', linestyle='--')#, label='No shaping')
    plt.legend()
    plt.show()

