# Parameterize the potential function phi(s) and learn the optimal one by maximizing area under reward curve.
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from qlearner import QLearning
from utils import plot_dist
from utils import ModifiedFrozenLake

map_name = '7x7wall'


# First solve the MDP:
env = ModifiedFrozenLake(map_name=map_name, slippery=0, min_reward=0, max_reward=1, step_penalization=0)
env = TimeLimit(env, max_episode_steps=100)



num_trials = 20
train_timesteps = 3000

def score_potential_function(potential_function, return_std=False):
    trial_rwds = np.zeros(num_trials)
    # use multiprocessing to run the trials:
    with Pool(num_trials) as p:
        # run the trials in parallel:
        trial_rwds = p.map(run_trial, [(potential_function, i) for i in range(num_trials)])
    p.close()

    if return_std:
        return np.mean(trial_rwds), np.std(trial_rwds)
    else:
        return -np.mean(trial_rwds)

def run_trial(args):
    potential_function, trial_idx = args
    # Now create the Q-learning agent:
    np.random.seed(trial_idx)  # Ensure deterministic behavior per trial
    agent = QLearning(env, gamma=0.95, learning_rate=1, phi=potential_function,
                    save_data=False)
    
    agent.train(train_timesteps)
    return sum(agent.reward_over_time) / len(agent.reward_over_time)

    
# score the zero potential function:
zero_potential = np.zeros(env.nS)
zero_score, zero_std = score_potential_function(zero_potential, return_std=True)
print(f'No shaping score: {zero_score} +/- {zero_std}')
# exit()
# List to store the historical scores
historical_scores = []


# Define a custom callback function to store the scores
def callback(potential_vec):
    score = score_potential_function(potential_vec)
    historical_scores.append(score)
    plot_dist(env.desc, potential_vec, filename='potential.png', show_plot=False, main_title=f'Score: {score}, Iteration: {len(historical_scores)}')
    

num_potentials = 1000000
potential_scores = np.zeros(num_potentials)
potential_stds = np.zeros(num_potentials)
potential_functions = [np.random.rand(env.nS) for _ in range(num_potentials)]

# List to keep track of inset positions and their corresponding images
inset_positions = []
inset_images = []

#TODO: Make genetic algorithm
for num, phi in enumerate(potential_functions):
    
    score, std = score_potential_function(phi, return_std=True)
    potential_scores[num] = score
    potential_stds[num] = std

    if num % 100 == 1:
        print(num)
        # Get the best potential function
        best_potential_idx = np.argmax(potential_scores)
        optimized_potential_vec = potential_functions[best_potential_idx]
        
        # Save the best potential function plot with a unique name
        inset_filename = f'potential_{num}.png'
        fig = plot_dist(env.desc, optimized_potential_vec, filename=inset_filename, show_plot=False, main_title=f'Score: {potential_scores[best_potential_idx]}, Iteration: {num}')
        # close the fig:
        plt.close(fig)
        
        # Save the inset position and corresponding image
        inset_x = num / num_potentials  # Normalized x position based on the iteration number
        # Get y by using the best potential score
        inset_y = potential_scores[best_potential_idx]
        # scale it by current y axis limits:
        # inset_y = (inset_y - np.min(potential_scores)) / (np.max(potential_scores) - np.min(potential_scores))
        inset_positions.append([inset_x, 0.5, 0.3, 0.3])  # Adjust size and position as needed
        inset_images.append(inset_filename)
        
        # Plot the sorted potential scores
        plt.figure()
        # plot_scores = potential_scores.copy()
        # sort scores and stds by scores:
        scores, stds = zip(*sorted(zip(potential_scores, potential_stds)))
        scores = np.array(scores)
        stds = np.array(stds)

        # pop out zeros:
        stds = stds[scores != 0]
        scores = scores[scores != 0]

        plt.plot(scores[::num], label='Potential scores')
        plt.fill_between(np.arange(len(scores)), scores - stds, scores + stds, alpha=0.2)
        # plot horizontal line at zero score:
        plt.axhline(y=zero_score, color='r', linestyle='--', label='No shaping')
        plt.fill_between(np.arange(len(scores)), zero_score - zero_std, zero_score + zero_std, color='r', alpha=0.2)
        
        # Add all previous insets
        # ax = plt.gca()
        # for pos, img in zip(inset_positions, inset_images):
        #     # get current y axis scale:
        #     ymin, ymax = ax.get_ylim()
        #     # scale the y position between 0 and 1 based on current ylim:
        #     axins = ax.inset_axes(pos)
        #     axins.imshow(plt.imread(img))
        #     axins.axis('off')

        # Save the plot with insets
        plt.legend()
        plt.savefig(f'potential_scores_tracker.png', bbox_inches='tight')
        # plt.close()
        # plt.cla()
