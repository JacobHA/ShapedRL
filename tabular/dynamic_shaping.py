import os
import numpy as np
from gymnasium.wrappers import TimeLimit
import sys
sys.path.append('tabular/')

from utils import ModifiedFrozenLake, get_mdp_generator

class DynamicQLearning():
    def __init__(self, env, gamma, learning_rate, 
                 eta=0,
                 save_data=False, 
                 prefix=''):
        self.env = env
        self.eval_env = env
        self.eta = eta
        self.nS = env.observation_space.n
        self.nA = env.action_space.n

        self.gamma = gamma
        self.learning_rate = learning_rate
        

        self.save_data = save_data
        if save_data:
            # count how many files are in data folder:
            if not os.path.exists(f'{prefix}-data'):
                os.makedirs(f'{prefix}-data')
            num = len(os.listdir(f'{prefix}-data')) + 1
            # append a number to the filename:
            path = f'{prefix}-data/q_{num}.npy'
            # if path exists, increment the number:
            while os.path.exists(path):
                num += 1
                path = f'{prefix}-data/q_{num}.npy'
            print(f'Saving data to {path}')
            self.path = path
        
        # random initialization:
        # self.Q = np.zeros((self.nS, self.nA))
        self.Q = np.random.rand(self.nS, self.nA) - np.ones((self.nS, self.nA))/ (1 - self.gamma)
        # self.Q /= (1+self.eta)
        # self.Q = np.ones((self.nS, self.nA)) * 0.5 / (1 - self.gamma)

        self.reward_over_time = []
        self.loss_over_time = []
        self.norm = 0
        self.prev_norm = 0

    def V_from_Q(self, Q):
        return np.max(Q, axis=1)
    
    def pi_from_Q(self, Q, V=None):
        if V is None:
            V = self.V_from_Q(Q)
        # Greedy policy:
        pi = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            pi[s, np.argmax(Q[s])] = 1

        return pi
    
    def draw_action(self, pi, state, greedy=False):
        if greedy:
            return np.argmax(pi[state])
        else:
            return np.random.choice(self.nA)
        
    def learn(self, state, action, reward, next_state, done):
        V = self.V_from_Q(self.Q)
        norm = self.norm if self.norm != 0 else 1
        norm = np.tanh(self.prev_norm / norm)
        phi = self.eta * V * norm #/ np.log(self.times[-1])
        print(norm)
        # shape the reward:
        reward += self.gamma * phi[next_state] - phi[state]

        # Compute the TD error:
        next_V = V[next_state]
        target = reward + (1 - done) * self.gamma * next_V
        delta = target - self.Q[state, action]
        self.norm = max(self.norm, np.abs(delta))
        self.prev_norm = np.abs(delta)
        # Update the Q value:
        self.Q[state, action] += self.learning_rate * delta
        
        return delta
    
    def train(self, max_steps, render=False, greedy_eval=True, eval_freq=50):
        self.times = np.arange(max_steps, step=eval_freq)
        
        state, _ = self.env.reset()
        steps = 0
        total_reward = 0
        done = False
        while steps < max_steps:
            pi = self.pi_from_Q(self.Q)
            # linearly decay epsilon from 1 to 0.1 at half max steps
            epsilon = max(0.05, 1 - steps / (max_steps / 2))
            # print(epsilon)
            if np.random.rand() < epsilon:
                action = self.draw_action(pi, state, greedy=False)
            else:
                action = self.draw_action(pi, state, greedy=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            delta = self.learn(state, action, reward, next_state, terminated)
            state = next_state
            steps += 1
            if render:
                self.env.render()
            if done:
                state, _ = self.env.reset()
                done = False
                
            if steps % eval_freq == 0:
                eval_rwd = self.evaluate(1, render=False, greedy=greedy_eval)
                total_reward += eval_rwd
                # print(f'steps={steps}, eval_rwd={eval_rwd:.2f}')
                self.reward_over_time.append(eval_rwd)
                # print(np.abs(self.Q).mean())

                if self.save_data:
                   
                    self.loss_over_time.append(0)#np.max(np.abs(self.Qstar - self.Q)))
                   
                    # Save the data:
                    self.save()

        return total_reward
    
    
    def evaluate(self, num_episodes, render=False, greedy=True):
        total_reward = 0
        pi = self.pi_from_Q(self.Q)

        for ep in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.draw_action(pi, state, greedy)
                state, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                if render:
                    self.eval_env.render()
                done = terminated or truncated
        return total_reward / num_episodes
    

                
    def save(self):
        # Save the data with np:
        # calculate the timesteps data:
        data = np.array([self.reward_over_time, 
                         self.loss_over_time, 
                         self.times[:len(self.reward_over_time)]])
        np.save(self.path, data)
        
        

def main(env_str, gamma, save=True, lr=None, prefix=None):
    # 11x11dzigzag
    env = ModifiedFrozenLake(map_name=env_str,
                             cyclic_mode=False,
                             slippery=0)
    
    env = TimeLimit(env, max_episode_steps=100)
    
    sarsa = DynamicQLearning(env, gamma, lr,
                      save_data=save,
                      prefix=prefix,
                      )
    max_steps = 10000_000

    total_reward = sarsa.train(max_steps, render=False, greedy_eval=True, eval_freq=100)
    return total_reward


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='7x7wall')
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('-n', type=int, default=1)
    args = parser.parse_args()

    for _ in range(args.n):
        main(env_str=args.env, gamma=args.gamma, lr=1)