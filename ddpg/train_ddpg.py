import numpy as np
import torch
from acc_env import ACCEnv
from ddpg_agent import DDPGAgent
import time

env = ACCEnv()
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = float(env.action_space.low[0])   # = -3.0
action_high= float(env.action_space.high[0])  # = +2.0

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

TOTAL_STEPS = 1_000_000
state, _ = env.reset()

episode_return = 0.0
episode_length = 0
log_interval = 50_000  # print every 50k steps

# --- Logging variables ---
start_time = time.time()
n_updates = 0
actor_loss, critic_loss = None, None
episode_rewards = []
episode_lengths = []

def print_log(step):
    elapsed = time.time() - start_time
    fps = int(step / elapsed) if elapsed > 0 else 0
    ep_len_mean = np.mean(episode_lengths[-10:]) if episode_lengths else 0
    ep_rew_mean = np.mean(episode_rewards[-10:]) if episode_rewards else 0
    print(f"""
rollout/           |          |
|    ep_len_mean     | {ep_len_mean:.0f}      |
|    ep_rew_mean     | {ep_rew_mean:.0f}     |
time/              |          |
|    episodes        | {len(episode_rewards)}       |
|    fps             | {fps}      |
|    time_elapsed    | {int(elapsed)}       |
|    total_timesteps | {step}     |
train/             |          |
|    actor_loss      | {actor_loss if actor_loss is not None else '-'}     |
|    critic_loss     | {critic_loss if critic_loss is not None else '-'}   |
|    learning_rate   | {agent.actor_opt.param_groups[0]['lr']:.6f}   |
|    n_updates       | {n_updates}     |
""")

for step in range(1, TOTAL_STEPS + 1):
    action = agent.select_action(state)  
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    agent.push_transition(state, action, reward, next_state, float(done))
    aloss, closs = agent.train_step()
    if aloss is not None and closs is not None:
        actor_loss, critic_loss = aloss, closs
        n_updates += 1

    episode_return += reward
    episode_length += 1
    state = next_state if not done else env.reset()[0]

    if done:
        episode_rewards.append(episode_return)
        episode_lengths.append(episode_length)
        print(f"Step {step:6d}  |  Episode return = {episode_return:.3f}  |  Length = {episode_length}")
        episode_return = 0.0
        episode_length = 0

    if step % log_interval == 0:
        print_log(step)
        # Save the actor model
        torch.save(agent.actor.state_dict(), f"actor_step{step}.pt")


