import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
import wandb
# np.random.seed(10)

import matplotlib.pyplot as plt

from printplots import visualize_path, plot_rewards

window_size = 500
episodes = 20000


sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'reward',
        'goal': 'maximize'
    },
    'parameters': {
        'is_slippery': {
            'values': [False, True]
        },
        'alpha': {
            'values': [0.01, 0.05, 0.1]
        },
        'gamma': {
            'values': [0.85, 0.90, 0.95]
        },
        'shaping_function': {
            'values': ['manhattan_distance', 'self_shaping']
        },
        'shaping': {
            'values': [True, False]
        }
    }
}

def manhattan_distance(state1, state2, gridsize, Q):
    x1, y1 = divmod(state1, gridsize)
    x2, y2 = divmod(state2, gridsize)
    return 1 - (abs(x1 - x2) + abs(y1 - y2)) / (2 * gridsize)

def euclidean_distance(state1, state2, gridsize, Q):
    x1, y1 = divmod(state1, gridsize)
    x2, y2 = divmod(state2, gridsize)
    return 1 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / (np.sqrt(2) * gridsize)

def self_shaping(state, _, gridsize, Q):
    # calculate the potential of the current state as V(s) = max_a Q(s, a)
    return np.max(Q[state])

def calculate_entropy(policy):
    return -np.sum([p * np.log(p) for p in policy if p > 0])

def evaluate_agent(env, Q):
    total_rewards = 0
    state, info = env.reset()
    while True:
        action = np.argmax(Q[state])
        new_state, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        state = new_state
        if terminated or truncated:
            break
    return total_rewards


def reward_shaping(is_slippery, alpha, gamma, shaping_function, shaping):
    # make new map
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, map_name="8x8")

    # print the state where we get the reward

    # Q = np.zeros((env.observation_space.n, env.action_space.n)) # empty q_table
    # initialize with small random values for Q
    Q = np.random.rand(env.observation_space.n, env.action_space.n) / 1000

    epsilon = 1
    epsilon_decay = (2 * epsilon) / episodes
    epsilon_min = 0.001

    state, info = env.reset()

    rewards = []    
    avg_rewards = [] 

    episode_reward = 0  # Track total reward for the current window

    grid_size = int(np.sqrt(env.observation_space.n))
    goal = grid_size ** 2 - 1

    reward_without_potential = 0

    state_visits = np.zeros(env.observation_space.n)
    state_probabilities_over_time = [[] for _ in range(env.observation_space.n)]

    for episode in tqdm(range(episodes)):
        if epsilon > epsilon_min:
            epsilon -= epsilon_decay

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            new_state, reward, terminated, truncated, info = env.step(action)
            reward_without_potential = reward
            # q_table_copy = Q.copy()
            if shaping:
                new_state_potential = shaping_function(new_state, action, grid_size, Q)
                old_state_potential = shaping_function(state, action, grid_size, Q)

                potential_reward = gamma * new_state_potential - old_state_potential
                reward += potential_reward

            target = reward + gamma * np.max(Q[new_state]) * (1 - terminated)

            Q[state, action] += alpha * (target - Q[state, action])

            episode_reward += reward_without_potential

            state = new_state
            state_visits[state] += 1


            if terminated or truncated:
                state, info = env.reset()
                break

        if (episode + 1) % window_size == 0:
            avg_reward = episode_reward / window_size
            avg_rewards.append(avg_reward)
            episode_reward = 0
            wandb.log({'average_reward': avg_reward})

        if (episode + 1) % 500 == 0:
            # use greedy policy to evaluate the agent
            # log into wandb
            wandb.log({'greedy_reward': evaluate_agent(env, Q)})
            rewards.append(evaluate_agent(env, Q))

        if (episode + 1) % 500 == 0:
            total_visits = np.sum(state_visits)
            state_probabilities = state_visits / total_visits
            for i in range(env.observation_space.n):
                state_probabilities_over_time[i].append(state_probabilities[i])
            # reset state_visits

            # log the state probabilities into wandb for each state
            for i in range(env.observation_space.n):
                wandb.log({f'state_{i}_probability': state_probabilities[i]})
            state_visits = np.zeros(env.observation_space.n)
    
    env.close()
    # visualize_path(env, Q)
    return rewards, avg_rewards, state_probabilities_over_time

def wandb_logging():
    with wandb.init() as run:
        alpha = run.config.alpha
        gamma = run.config.gamma
        shaping_function = run.config.shaping_function
        shaping = run.config.shaping
        is_slippery = run.config.is_slippery
        if shaping:
            run.name = f'{shaping_function}_shaping'
            if shaping_function == 'manhattan_distance':
                shaping_function = manhattan_distance
            else:
                shaping_function = self_shaping
        else:
            run.name = 'no_shaping'
        rewards, avg_rewards, state_probabilities = reward_shaping(is_slippery, alpha, gamma, shaping_function, shaping)


if __name__ == "__main__":
    # rewards, avg_rewards, state_probabilities = reward_shaping(is_slippery=False, alpha=0.05, gamma=0.90, beta=0.01, shaping_function=self_shaping, shaping=False)  

    sweep_id = wandb.sweep(sweep_config, project="self-shaping-frozen-lake")

    wandb.agent(sweep_id, function=wandb_logging)

    



