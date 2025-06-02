
"""
Env creatin for acc
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ACCEnv(gym.Env):
    metadata = {"render_modes": []}

    # constant parameters (paper)
    dt, tg, tau = 0.1, 1.0, 0.5             # sample, time-gap, lag
    umin, umax  = -3.0, 2.0
    e_max       = 15.0
    jerk_scale  = (umax-umin)/dt            # =50
    alpha = beta = gamma = 1/3
    eps   = 1e-8

    def __init__(self):
        super().__init__()

        hi = np.array([ 15,  15,  3], dtype=np.float32)
        lo = np.array([-15, -15, -3], dtype=np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)
        self.action_space      = spaces.Box(np.array([self.umin],dtype=np.float32),
                                            np.array([self.umax],dtype=np.float32),
                                            dtype=np.float32)
        self.reset(seed=None)

    # ---------- continuous dynamics ---------------------------------------
    def _dyn(self, x, u):
        e, ev, a = x
        return np.array([ev - self.tg*a,
                         -a,
                         (u - a)/self.tau], dtype=np.float32)

    # ---------- RL step ----------------------------------------------------
    def step(self, action):
        u = float(np.clip(action[0], self.umin, self.umax))
        x_dot = self._dyn(self.state, u)
        x_new = self.state + self.dt * x_dot

        # stage cost -------------------------------------------------------
        e, _, a = self.state
        jerk    = (a - self.prev_a) / self.dt
        u_scale = max(abs(self.umin), abs(self.umax))
        cost = ( self.alpha*np.sqrt((e/self.e_max)**2        + self.eps)
               + self.beta *np.sqrt((u/u_scale)     **2      + self.eps)
               + self.gamma*np.sqrt((jerk/self.jerk_scale)**2+ self.eps) )
        reward = -np.clip(cost, 0, 1)                       # ∈[-1,0]

        self.state, self.prev_a = x_new, a
        self.t_step += 1
        truncated   = self.t_step >= 200                    # 20 s
        return x_new.copy(), reward, False, truncated, {}

    # ---------- random reset (for training) -------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.state = np.array([rng.uniform(-5,5),
                               rng.uniform(-5,5),
                               rng.uniform(-3,2)], dtype=np.float32)
        self.prev_a = self.state[2]
        self.t_step = 0
        return self.state.copy(), {}

    # ---------- deterministic reset (for tests / plots) -------------------
    def reset_ic(self, e0, ev0, a0):
        """Reset with a specific initial condition."""
        self.state  = np.array([e0, ev0, a0], dtype=np.float32)
        self.prev_a = a0
        self.t_step = 0
        return self.state.copy(), {}