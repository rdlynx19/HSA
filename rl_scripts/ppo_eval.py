import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from gymnasium.wrappers import TimeLimit

from hsa_gym.envs.hsa_position import HSAEnv


def make_env():
    """
    Helper function to create the environment
    """
    env = HSAEnv(render_mode="human", actuator_group=[1], action_group=[1],smooth_positions=True)
    env = TimeLimit(env, max_episode_steps=2000)
    return env


def main():
    # Create the environment
    env = make_env()

    # Load the pre-trained PPO model
    model = PPO.load("model_2000000_steps.zip", env=env)

    num_episodes = 15
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1}: Total Reward = {total_reward:.3f}")
    env.close()

if __name__ == "__main__":
    main()