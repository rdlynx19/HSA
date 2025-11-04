import numpy as np
from hsa_gym.envs.hsa_v1 import HSAEnv

env = HSAEnv(render_mode="human", randomize_goal=False)

obs, info = env.reset()
print("Initial Observation:", obs.shape)
print("Initial Info:", info)

for i in range(10):
    action = {
        "motors": env.action_space["motors"].sample(),
        "locks": env.action_space["locks"].sample()
    }
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {i+1}:")
    print(f"Reward: {reward}")
    print(f"Distance to Goal: {info['cur_distance']}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        break
env.close()
