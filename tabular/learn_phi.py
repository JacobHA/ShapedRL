# Parameterize the potential function phi(s) and learn the optimal one by maximizing area under reward curve.
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from qlearner import QLearning
from utils import plot_dist
from utils import ModifiedFrozenLake

map_name = 'hallway1'


# First solve the MDP:
env = ModifiedFrozenLake(map_name=map_name, slippery=0)
env = TimeLimit(env, max_episode_steps=100)


num_trials = 25
train_timesteps = 3000

def score_potential_function(potential_function):
    trial_rwds = np.zeros(num_trials)
    # use multiprocessing to run the trials:
    with Pool(num_trials) as p:
        # run the trials in parallel:
        trial_rwds = p.map(run_trial, [potential_function] * num_trials)
    print(np.mean(trial_rwds))
    return -np.mean(trial_rwds)

def run_trial(potential_function):
    for trial in range(num_trials):
        # Now create the Q-learning agent:
        agent = QLearning(env, gamma=0.95, learning_rate=1, phi=potential_function,
                        save_data=False)
        
        agent.train(train_timesteps)
        return sum(agent.reward_over_time) / len(agent.reward_over_time)

# Use an optimizer to find the best potential function:
# Initial potential vector (random initialization)

initial_potential_vec = np.random.rand(env.nS)

# Perform the optimization using scipy.optimize.minimize
from scipy import optimize
max_iter = 2000

# List to store the historical scores
historical_scores = []

# Set up the figure for plotting

# Define a custom callback function to store the scores
def callback(potential_vec):
    score = score_potential_function(potential_vec)
    historical_scores.append(score)
    
    # Clear the previous image
    # ax.clear()
    
    # Plot the potential_vec in the maze
    # im = ax.imshow(potential_vec.reshape(), cmap='viridis')
    plot_dist(env.desc, potential_vec, filename='potential.png', show_plot=False)
    
    # Add a colorbar
    # plt.colorbar(im, ax=ax)
    
    # Pause to update the plot
    # plt.pause(0.05)
    # plt.draw()

result = optimize.minimize(score_potential_function, initial_potential_vec, method='BFGS',
                            # options={'maxfev': max_iter, 'disp': True},
                            options={'gtol': 1e-6, 'disp': True, 'maxiter': max_iter},
                            callback=callback)

# Get the optimized potential vector
optimized_potential_vec = result.x

# Print the results
print("Initial potential vector:", initial_potential_vec)
print("Optimized potential vector:", optimized_potential_vec)
print("Initial score:", score_potential_function(initial_potential_vec))
print("Optimized score:", score_potential_function(optimized_potential_vec))

# # Plot the score over time:
# plt.plot(historical_scores)
# plt.xlabel('Iterations')
# plt.ylabel('Score')
# plt.title('Score over iterations')
# plt.show()

