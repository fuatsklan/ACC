# test_ddpg.py -------------------------------------------------------------
"""
Roll out the trained DDPG policy from the SAME initial condition
[e0, ev0, a0] = [5, 5, 0] that MPC/IPO use.
Generates:
    plots_ddpg/ddpg_combined.pdf  (4-panel)
    plots_ddpg/gap_ddpg.png  speed_ddpg.png  u_a_ddpg.png  jerk_ddpg.png
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib

from acc_env import ACCEnv
from ddpg_agent import DDPGAgent

# --- load environment -----------------------------------------------------
env = ACCEnv()
state_dim   = env.observation_space.shape[0]     # → 3
action_dim  = env.action_space.shape[0]          # → 1
action_low  = float(env.action_space.low[0])     # → -3.0
action_high = float(env.action_space.high[0])    # →  2.0

# --- load DDPG agent ------------------------------------------------------
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    action_low=action_low,
    action_high=action_high,
    gamma=0.99,
    tau=0.001,
    actor_lr=1e-4,
    critic_lr=1e-3,
    batch_size=64,
    buffer_size=int(5e5),
    expl_noise=0.02
)
agent.actor.load_state_dict(torch.load("actor_step1000000.pt"))
agent.actor.eval()

# --- deterministic IC -----------------------------------------------------
obs, _ = env.reset_ic(e0=5.0, ev0=5.0, a0=0.0)
dt, v_lead = env.dt, 5.0

# --- roll-out loop --------------------------------------------------------
t, e, v, a, u, j = [], [], [], [], [], [0.0]
prev_a = env.state[2]

for k in range(200):
    action = agent.select_action(obs, add_noise=False)
    obs, _, _, done, _ = env.step(action)

    e_k, ev_k, a_k = env.state
    u_k            = float(action[0])

    t.append(k * dt)
    e.append(e_k)
    v.append(v_lead - ev_k)
    a.append(a_k)
    u.append(u_k)
    j.append((a_k - prev_a) / dt)
    prev_a = a_k

    if done:
        break

# --- convert to NumPy -----------------------------------------------------
t = np.array(t)
gap = np.array(e)
v_f = np.array(v)
a_f = np.array(a)
u_f = np.array(u)
j_f = np.array(j[:len(t)])

# --- plot setup -----------------------------------------------------------
mpl.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False
})

fig, axs = plt.subplots(2, 2, sharex=True)
ax_e, ax_v, ax_ua, ax_j = axs.flatten()

# (a) Gap error
ax_e.plot(t, gap, lw=1.7)
ax_e.set_title("(a) Gap error")
ax_e.set_ylabel("e [m]")

# (b) Speeds
ax_v.plot(t, v_f, lw=1.5, label="Follower")
ax_v.plot(t, v_lead * np.ones_like(t), "--", lw=1.2, label="Lead")
ax_v.set_title("(b) Speeds")
ax_v.set_ylabel("Speed [m/s]")
ax_v.legend()

# (c) Control input u and acceleration a_i
ln1 = ax_ua.step(t, u_f, where='post', lw=1.5, label="u")
ax2 = ax_ua.twinx()
ln2 = ax2.plot(t, a_f, 'k--', lw=1.2, label="a_i")
ax_ua.set_title("(c) u and a_i")
ax_ua.set_ylabel("u  [m/s²]")
ax2.set_ylabel("a_i [m/s²]")
ax_ua.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2], loc="upper right")

# (d) Jerk
ax_j.plot(t, j_f, lw=1.5)
ax_j.set_title("(d) Jerk")
ax_j.set_ylabel("jerk [m/s³]")
ax_j.set_xlabel("Time [s]")

fig.tight_layout()

# --- save plots -----------------------------------------------------------
outdir = pathlib.Path("plots_ddpg")
outdir.mkdir(exist_ok=True)

# Save combined figure
fig.savefig(outdir / "ddpg_combined.pdf")
fig.canvas.draw()

# Save individual panels
for ax, name in zip(axs.flatten(), ("gap", "speed", "u_a", "jerk")):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted())
    fig.savefig(outdir / f"{name}_ddpg.png", bbox_inches=bbox)

plt.show()
