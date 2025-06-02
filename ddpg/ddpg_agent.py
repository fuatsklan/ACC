import torch
import torch.nn.functional as F
from networks import Actor, Critic
from replay_buffer import ReplayBuffer
import numpy as np

class DDPGAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_low: float,
                 action_high: float,
                 gamma=0.99,
                 tau=1e-3,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 batch_size=64,
                 buffer_size=int(5e5),
                 device=None,
                 expl_noise=0.02):
        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Save action bounds for clipping and for actor’s forward‐scaling
        self.action_low  = torch.as_tensor(action_low,  device=self.device)
        self.action_high = torch.as_tensor(action_high, device=self.device)

        # Build actor & critic + their target networks
        self.actor        = Actor(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, action_low, action_high).to(self.device)
        self.critic       = Critic(state_dim, action_dim).to(self.device)
        self.critic_target= Critic(state_dim, action_dim).to(self.device)

        # Initialize targets with same weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay = ReplayBuffer(state_dim, action_dim, buffer_size, self.device)

        # Hyperparams
        self.gamma = gamma
        self.tau   = tau
        self.batch_size = batch_size
        self.expl_noise = expl_noise

    @torch.no_grad()
    def select_action(self, state: np.ndarray, add_noise=True) -> np.ndarray:
        """
        state: 1D np.array of shape (state_dim,)
        returns: 1D np.array of shape (action_dim,) clipped to [action_low, action_high]
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action  = self.actor(state_t).squeeze(0).cpu().numpy()  # already in [low, high]

        if add_noise:
            action += self.expl_noise * np.random.randn(*action.shape)

        # Hard clip in case noise pushed us outside
        return np.clip(action, self.action_low.cpu().item(), self.action_high.cpu().item())

    def push_transition(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self):
        if self.replay.size < self.batch_size:
            return  # not enough samples yet

        # Sample a batch
        s, a, r, s2, d = self.replay.sample(self.batch_size)

        # 1) Critic update
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target = self.critic_target(s2, a2)
            y = r + self.gamma * (1 - d) * q_target

        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2) Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3) Soft‐update targets (Polyak averaging)
        with torch.no_grad():
            for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                target.data.mul_(1 - self.tau).add_(self.tau * param.data)
            for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.mul_(1 - self.tau).add_(self.tau * param.data)

