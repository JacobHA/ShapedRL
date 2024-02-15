import torch as th
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np

from stable_baselines3.dqn import DQN


class ShapedDQN(DQN):
    """
    DQN variant with shaped rewards at each Q update:
    (^ denotes the current estimate as given by the FA.)
    assuming V^ = max_a Q^(s,a), and that the environment is deterministic,
    r --> r + gamma V^(s') - V(s)
    """

    def __init__(self, *args, shaping_mode='none', **kwargs):
        super(ShapedDQN, self).__init__(*args, **kwargs)
        self.shaping_mode = shaping_mode


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_value = self.q_net_target(replay_data.next_observations)
                # min_q_value, idx_min = next_q_value.min(dim=1, keepdim=True)
                # Follow greedy policy: use the one with the highest value
                max_q_value, _ = next_q_value.max(dim=1, keepdim=True)
                # Avoid potential broadcast issue
                max_q_value = max_q_value.reshape(-1, 1)

                if self.shaping_mode == 'online':
                    curr_q_values = self.q_net(replay_data.observations)
                    next_q_values = self.q_net(replay_data.next_observations)
                elif self.shaping_mode == 'target':
                    curr_q_values = self.q_net_target(replay_data.observations)
                    next_q_values = self.q_net_target(replay_data.next_observations)
                elif self.shaping_mode == 'none':
                    pass
                
                # TODO: Do we include the 1-dones here?
                rewards = replay_data.rewards

                if self.shaping_mode != 'none':
                    curr_v_max, _ = curr_q_values.max(dim=1, keepdim=True)
                    next_v_max, _ = next_q_values.max(dim=1, keepdim=True)

                    rewards +=  ((1 - replay_data.dones) * self.gamma * next_v_max - curr_v_max)

                target_q_values = rewards + \
                    (1 - replay_data.dones) * self.gamma * max_q_value

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
