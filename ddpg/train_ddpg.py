import numpy as np
import torch
from acc_env import ACCEnv
from ddpg_agent import DDPGAgent

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
log_interval = 50_000  # print every 10k steps

for step in range(1, TOTAL_STEPS + 1):
    action = agent.select_action(state)  
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    agent.push_transition(state, action, reward, next_state, float(done))
    agent.train_step()

    episode_return += reward
    episode_length += 1
    state = next_state if not done else env.reset()[0]

    if done:
        print(f"Step {step:6d}  |  Episode return = {episode_return:.3f}  |  Length = {episode_length}")
        episode_return = 0.0
        episode_length = 0

    if step % log_interval == 0:
        print(f"*** Step {step:,} completed ***")
        # Save the actor model
        torch.save(agent.actor.state_dict(), f"actor_step{step}.pt")


