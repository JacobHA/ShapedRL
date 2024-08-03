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

    def __init__(self, *args, do_shape:bool=False, no_done_mask:bool=False,
                 shape_scale:float=1.0,
                 grad_reward:bool=False,
                 use_oracle=False, oracle_path:str='', **kwargs):
        self.do_shape = do_shape
        self.shape_scale = shape_scale
        self.grad_reward = grad_reward
        self.no_done_mask = no_done_mask
        self.use_oracle = use_oracle
        self.oracle_path = oracle_path
        # import the saved Q network:
        if self.use_oracle:
            self.shaping_net = th.load(self.oracle_path)
        self.kwargs = locals()
        self.kwargs.pop('self')
        self.kwargs.pop('__class__')
        super(ShapedDQN, self).__init__(*args, **kwargs)

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

            rewards = replay_data.rewards

            if self.do_shape:
                if self.grad_reward:
                    # Build the shaping potential function, using online net:
                    curr_q_values = self.policy.q_net(replay_data.observations)
                    next_q_values = self.policy.q_net(replay_data.next_observations)
                    curr_v_max, _ = curr_q_values.max(dim=1, keepdim=True)
                    next_v_max, _ = next_q_values.max(dim=1, keepdim=True)

                else:
                    with th.no_grad():
                        # Build the shaping potential function, using online net:
                        curr_q_values = self.policy.q_net(replay_data.observations)
                        next_q_values = self.policy.q_net(replay_data.next_observations)
                        curr_v_max, _ = curr_q_values.max(dim=1, keepdim=True)
                        next_v_max, _ = next_q_values.max(dim=1, keepdim=True)

                dones_mask = 1 if self.no_done_mask else 1 - replay_data.dones

                # TODO: Experiment with weight schedule?
                eta = self.shape_scale
                rewards += eta * (dones_mask * self.gamma * next_v_max - curr_v_max)
        
            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_value = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                max_q_value, _ = next_q_value.max(dim=1, keepdim=True)
                # Avoid potential broadcast issue
                max_q_value = max_q_value.reshape(-1, 1)

                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * max_q_value

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

    def __str__(self):
        return f"s{1 if self.do_shape else 0}t{1 if self.no_done_mask else 0}"

    def save(self, path='./model'):
        state = self.state_dict()
        th.save([self.kwargs, state], path)
