import gymnasium
import numpy as np
import torch
import wandb
from .BaseAgent import BaseAgent
from .Models import SoftQNet, OnlineSoftQNets, Optimizers, TargetNets
from utils import logger_at_folder

class ShapedSQL(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 do_shape: bool = False,
                 soft_weight: float = 1.0,
                 use_one_minus_dones: bool = True,
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.algo_name = 'SQL'
        self.gamma = gamma
        self.do_shape = do_shape
        self.soft_weight = soft_weight
        self.use_one_minus_dones = use_one_minus_dones

        self.total_clips = 0
        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_softqs = OnlineSoftQNets([SoftQNet(self.env, 
                                                  hidden_dim=self.hidden_dim, 
                                                  device=self.device,
                                                  )
                                              for _ in range(self.num_nets)],
                                            beta=self.beta,
                                            aggregator_fn=self.aggregator_fn)
        # alias for compatibility as self.model:
        self.model = self.online_softqs

        self.target_softqs = TargetNets([SoftQNet(self.env, 
                                                  hidden_dim=self.hidden_dim, 
                                                  device=self.device,
                                                  )
                                        for _ in range(self.num_nets)])
        self.target_softqs.load_state_dicts(
            [softq.state_dict() for softq in self.online_softqs])
        # Make (all) softqs learnable:
        opts = [torch.optim.Adam(softq.parameters(), lr=self.learning_rate)
                for softq in self.online_softqs]
        self.optimizers = Optimizers(opts, self.scheduler_str)

    def exploration_policy(self, state: np.ndarray) -> int:
        # return self.env.action_space.sample()
        return self.online_softqs.choose_action(state)

    def evaluation_policy(self, state: np.ndarray) -> int:
        return self.online_softqs.choose_action(state, greedy=True)

    def gradient_descent(self, batch):
        states, actions, next_states, dones, rewards = batch
        curr_softq = torch.stack([softq(states).squeeze().gather(1, actions.long())
                                for softq in self.online_softqs], dim=0)
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()
            online_softq_next = torch.stack([softq(next_states)
                                            for softq in self.online_softqs], dim=0)
            online_curr_softq = torch.stack([softq(states).gather(1, actions)
                                            for softq in self.online_softqs], dim=0)

            online_curr_softq = online_curr_softq.squeeze(-1)

            target_next_softqs = [target_softq(next_states)
                                 for target_softq in self.target_softqs]
            target_next_softqs = torch.stack(target_next_softqs, dim=0)

            # aggregate the target next softqs:
            target_next_softq = self.aggregator_fn(target_next_softqs, dim=0)
            next_v = 1/self.beta * (torch.logsumexp(
                self.beta * target_next_softq, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            if self.do_shape:
                online_next_v = 1/self.beta * (torch.logsumexp(
                self.beta * online_softq_next, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
                online_next_v = online_next_v.reshape(-1, 1)
                curr_v = 1/self.beta * (torch.logsumexp(
                    self.beta * online_curr_softq, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
                
                rewards += self.soft_weight * \
                    (self.use_one_minus_dones * self.gamma * online_next_v - curr_v)
                
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)
            expected_curr_softq = expected_curr_softq.squeeze(1)

        # num_nets, batch_size, 1 (leftover from actions)
        curr_softq = curr_softq.squeeze(2)

        self.logger.record("train/online_q_mean", curr_softq.mean().item())
        wandb.log({"train/online_q_mean": curr_softq.mean().item()})

        # Calculate the softq ("critic") loss:
        loss = 0.5*sum(self.loss_fn(softq, expected_curr_softq)
                       for softq in curr_softq)

        return loss

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_softqs.polyak(self.online_softqs, self.tau)
