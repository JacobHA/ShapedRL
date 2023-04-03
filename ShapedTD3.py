from stable_baselines3.td3 import TD3
from stable_baselines3.common.utils import polyak_update
import torch as th
from torch.nn import functional as F
import numpy as np


class ShapedTD3(TD3):
    def __init__(self, *args, shaped: int = 0, **kwargs):
        super(ShapedTD3, self).__init__(*args, **kwargs)
        self.shaped = shaped

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip,
                                    self.target_noise_clip)
                next_actions = (self.actor_target(
                    replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(
                    replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                current_q_vals = th.cat(current_q_values, dim=1)
                current_q_vals, _ = th.min(current_q_vals, dim=1, keepdim=True)
                curr_v_max, _ = current_q_vals.max(dim=1, keepdim=True)

                next_v_max, _ = next_q_values.max(dim=1, keepdim=True)
                rewards = replay_data.rewards
                if self.shaped:
                    rewards += self.gamma * next_v_max - curr_v_max

                target_q_values = rewards + \
                    (1 - replay_data.dones) * self.gamma * next_q_values

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values)
                              for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(),
                              self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(),
                              self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats,
                              self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats,
                              self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
