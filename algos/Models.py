import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.preprocessing import preprocess_obs

import torch
import torch.nn as nn
import torch.nn.functional as F


class Optimizers():
    def __init__(self, list_of_optimizers: list, scheduler_str: str = 'none'):
        assert scheduler_str == 'none', "only 'none' is supported for now."
        self.optimizers = list_of_optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class TargetNets():
    def __init__(self, list_of_nets):
        self.nets = list_of_nets
    def __len__(self):
        return len(self.nets)
    def __iter__(self):
        return iter(self.nets)

    def load_state_dicts(self, list_of_state_dicts):
        """
        Load state dictionaries into target networks.

        Args:
            list_of_state_dicts (list): A list of state dictionaries to load into the target networks.

        Raises:
            ValueError: If the number of state dictionaries does not match the number of target networks.
        """
        if len(list_of_state_dicts) != len(self):
            raise ValueError("Number of state dictionaries does not match the number of target networks.")
        
        for online_net_dict, target_net in zip(list_of_state_dicts, self):
            
            target_net.load_state_dict(online_net_dict)

    def polyak(self, online_nets, tau):
        """
        Perform a Polyak (exponential moving average) update for target networks.

        Args:
            online_nets (list): A list of online networks whose parameters will be used for the update.
            tau (float): The update rate, typically between 0 and 1.

        Raises:
            ValueError: If the number of online networks does not match the number of target networks.
        """
        if len(online_nets) != len(self.nets):
            raise ValueError("Number of online networks does not match the number of target networks.")

        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for new_params, target_params in zip(online_nets.parameters(), self.parameters()):
                # for new_param, target_param in zip_strict(new_params, target_params):
                #     target_param.data.mul_(tau).add_(new_param.data, alpha=1.0-tau)
                #TODO: 
                polyak_update(new_params, target_params, tau)

    def parameters(self):
        """
        Get the parameters of all target networks.

        Returns:
            list: A list of network parameters for each target network.
        """
        return [net.parameters() for net in self.nets]


class OnlineNets():
    """
    A utility class for managing online networks in reinforcement learning.

    Args:
        list_of_nets (list): A list of online networks.
    """
    def __init__(self, list_of_nets, aggregator_fn, is_vector_env=False):
        self.nets = list_of_nets
        self.nA = list_of_nets[0].nA
        self.device = list_of_nets[0].device
        self.aggregator_fn = aggregator_fn
        self.is_vector_env = is_vector_env

    def __len__(self):
        return len(self.nets)
    
    def __iter__(self):
        return iter(self.nets)

    def choose_action(self, state, greedy=False, prior=None):
        raise NotImplementedError

    def parameters(self):
        return [net.parameters() for net in self]

    def clip_grad_norm(self, max_grad_norm):
        for net in self:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


class OnlineSoftQNets(OnlineNets):
    def __init__(self, list_of_nets, aggregator_fn, beta, is_vector_env=False):
        super().__init__(list_of_nets, aggregator_fn, is_vector_env)
        self.beta = beta
           
    def choose_action(self, state, greedy=False, prior=None):
        if prior is None:
            prior = 1 / self.nA
        with torch.no_grad():
            q_as = torch.stack([net.forward(state, eval=True) for net in self], dim=-1)
            # q_as = q_as.squeeze(0)
            q_a = self.aggregator_fn(q_as, dim=-1)


            if greedy:
                action = torch.argmax(q_a).cpu().item()
            else:
                # pi propto e^beta Q:
                # first subtract a baseline from q_a:
                q_a = q_a - (torch.max(q_a) + torch.min(q_a))/2
                clamped_exp = torch.clamp(self.beta * q_a, min=-20, max=20)
                pi = prior * torch.exp(clamped_exp)
                pi = pi / torch.sum(pi)
                a = Categorical(pi).sample()
                action = a.cpu().item()
        return action


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(DuelingDQN, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Scalar output
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)  # One output per action
        )

    def forward(self, x):
        # shared_output = self.shared_layers(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

    
class SoftQNet(torch.nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256, activation=nn.ReLU):
        super(SoftQNet, self).__init__()
        self.env = env
        self.nA = env.action_space.n
        self.device = device
        # Start with an empty model:
        model = None

        self.nS = env.observation_space.shape
        try:
            nS = env.observation_space.n
        except:
            nS = self.nS[0]
        input_dim = nS
   
        model = DuelingDQN(input_dim, hidden_dim, self.nA)

        model.to(self.device)
        self.model = model
    
    def forward(self, x, eval=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)  # Convert to PyTorch tensor
        
        x = preprocess_obs(x, self.env.observation_space)
        if eval:
            x = x.unsqueeze(0)
        assert x.dtype == torch.float32, "Input must be a float tensor."

        x = self.model(x)
        return x
