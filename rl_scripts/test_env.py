import numpy as np
from hsa_gym.envs.hsa_v1 import HSAEnv
from gymnasium.wrappers import TimeLimit

env = HSAEnv(render_mode="human", randomize_goal=False)
env = TimeLimit(env, max_episode_steps=500)

obs, info = env.reset()
print("Initial Observation:", obs.shape)
print("Initial Info:", info)

for i in range(1000):
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {i+1}:")
    print(f"Reward: {reward}")
    print(f"Distance to Goal: {info['cur_distance']}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        print("Episode finished after {} timesteps".format(i+1))
        break
env.close()
