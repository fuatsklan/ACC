import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Actor network that maps state → action, with final tanh scaled into [action_low, action_high].
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_low: float,
                 action_high: float,
                 net_arch=(64, 64)):
        super().__init__()

        # Build a simple MLP: [state_dim → net_arch… → action_dim], last layer has tanh
        layers = []
        last = state_dim
        for h in net_arch:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

        # Store bounds as buffers for easy broadcast in forward()
        self.register_buffer("action_low",  torch.tensor(action_low,  dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch_size, state_dim)
        returns: (batch_size, action_dim) in [action_low, action_high]
        """
        raw = self.net(state)  # in [-1, +1] because of final tanh

        # scale & shift from [-1,1] → [action_low, action_high]
        #  range = (high - low) / 2
        #   mid   = (high + low) / 2
        action_range = (self.action_high - self.action_low) / 2.0
        action_mid   = (self.action_high + self.action_low) / 2.0
        return action_mid + action_range * raw


class Critic(nn.Module):
    """
    Critic network that maps (state, action) → Q-value (scalar).
    """
    def __init__(self, state_dim: int, action_dim: int, net_arch=(64, 64)):
        super().__init__()

        layers = []
        last = state_dim + action_dim
        for h in net_arch:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state:  (batch_size, state_dim)
        action: (batch_size, action_dim)
        returns: (batch_size, 1) Q-value
        """
        x = torch.cat([state, action], dim=1)
        return self.net(x)
