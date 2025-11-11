import numpy as np
from hsa_gym.envs.hsa_position import HSAEnv
from gymnasium.wrappers import TimeLimit

env = HSAEnv(render_mode="human", actuator_groups=[1])
env = TimeLimit(env, max_episode_steps=50000)

obs, info = env.reset()
print("Initial Observation:", obs.shape)
print("Initial Info:", info)

for i in range(10000):
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)
    print("Observation:", obs.shape)
    print(f"Action taken: {action}")
    print(f"Step {i+1}:")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        print("Episode finished after {} timesteps".format(i+1))
        break
env.close()
