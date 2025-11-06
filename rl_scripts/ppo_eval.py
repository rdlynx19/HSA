import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO

from hsa_gym.envs.hsa_v1 import HSAEnv
from hsa_gym.envs.sb3_wrapper import SB3Wrapper


def make_env():
    """
    Helper function to create the environment
    """
    env = HSAEnv(render_mode="human", randomize_goal=False, use_locks=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
    env = SB3Wrapper(env)
    return env

def main():
    # Create the environment
    env = make_env()

    # Load the pre-trained PPO model
    model = PPO.load("ppo_lamb1_final", env=env)

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