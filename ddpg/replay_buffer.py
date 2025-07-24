
import torch
import numpy as np

class ReplayBuffer:
    """Fixed-size cyclic buffer on GPU or CPU."""
    def __init__(self, state_dim, action_dim, max_size=int(5e5), device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.states  = torch.empty((max_size, state_dim),  dtype=torch.float32, device=device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.empty((max_size, 1),          dtype=torch.float32, device=device)
        self.next_s  = torch.empty((max_size, state_dim),  dtype=torch.float32, device=device)
        self.dones   = torch.empty((max_size, 1),          dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states [idx] = torch.as_tensor(state,  device=self.device)
        self.actions[idx] = torch.as_tensor(action, device=self.device)
        self.rewards[idx] = torch.as_tensor(reward, device=self.device).view(1)
        self.next_s[idx]  = torch.as_tensor(next_state, device=self.device)
        self.dones [idx]  = torch.as_tensor(done,  device=self.device).view(1)

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states [idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_s [idx],
                self.dones [idx])
