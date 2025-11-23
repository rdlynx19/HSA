import numpy as np
from hsa_gym.envs.hsa_constrained import HSAEnv
from gymnasium.wrappers import TimeLimit

env = HSAEnv(xml_file="hsaTerrainModel.xml", render_mode="human",  actuator_group=[1])
env = TimeLimit(env, max_episode_steps=500)

obs, info = env.reset()
print("Initial Observation:", obs.shape)
print("Initial Info:", info)

for i in range(1000):
    action = env.action_space.sample() 
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {i+1}:")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        print("Episode finished after {} timesteps".format(i+1))
        break
env.close()
