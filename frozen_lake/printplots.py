import numpy as np
import matplotlib.pyplot as plt

def visualize_path(env, Q):
    state, info = env.reset()
    path = [state]
    actions = ['L', 'D', 'R', 'U']  # Corresponding actions: 0=Left, 1=Down, 2=Right, 3=Up
    
    grid_size = int(np.sqrt(env.observation_space.n))
    path_grid = np.full((grid_size, grid_size), ' ')

    while True:
        action = np.argmax(Q[state])
        path_grid[state // grid_size, state % grid_size] = actions[action]
        new_state, reward, terminated, truncated, info = env.step(action)
        path.append(new_state)
        state = new_state
        if terminated or truncated:
            break

    # Mark the start and goal
    path_grid[0, 0] = 'S'
    path_grid[-1, -1] = 'G'

    # Print the path grid
    # for row in path_grid:
    #     print(' '.join(row))
    
    # Plot the path grid
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.table(cellText=path_grid, loc='center', cellLoc='center')
    ax.axis('off')


def plot_rewards(window_size, episodes, rewards, avg_rewards, state_probabilities):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(window_size, episodes + 1, window_size), avg_rewards, label='Average Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time (Window Size = 500)')
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(500, episodes + 1, 500), rewards, label='Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.grid(True)
    plt.legend()
    for i in range(len(rewards)):
        plt.plot(i * window_size + window_size, rewards[i], 'ro')

    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(state_probabilities[i])
        ax.set_title(f'State {i}')
    plt.tight_layout()

    # plot a heatmap for state_probabilities[-1]
    last_iteration_probabilities = []
    for i in range(len(state_probabilities)):
        last_iteration_probabilities.append(state_probabilities[i][-1])
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(last_iteration_probabilities).reshape(8, 8), cmap='hot', interpolation='nearest')
    plt.colorbar


    last_iteration_probabilities = []
    for i in range(len(state_probabilities)):
        last_iteration_probabilities.append(state_probabilities[i][3])
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(last_iteration_probabilities).reshape(8, 8), cmap='hot', interpolation='nearest')
    plt.colorbar

    plt.show()