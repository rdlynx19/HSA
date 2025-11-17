import gymnasium as gym
import numpy as np
import yaml
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from gymnasium.wrappers import TimeLimit
from hsa_gym.envs.hsa_constrained import HSAEnv

def load_config(checkpoint_dir: str):
    """
    Load the training configuration from the checkpoint directory
    """
    config_path = os.path.join(checkpoint_dir, "used_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_env(config, render_mode="human"):
    """
    Helper function to create the environment with exact training parameters
    """
    env_config = config["env"]
    
    env = HSAEnv(
        render_mode=render_mode,
        actuator_group=env_config["actuator_group"],
        action_group=env_config["action_group"],
        forward_reward_weight=env_config["forward_reward_weight"],
        ctrl_cost_weight=env_config["ctrl_cost_weight"],
        contact_cost_weight=env_config["contact_cost_weight"],
        yvel_cost_weight=env_config["yvel_cost_weight"],
        constraint_cost_weight=env_config["constraint_cost_weight"],
        smooth_positions=env_config["smooth_positions"],
        frame_skip=env_config["frame_skip"],
        max_increment=env_config["max_increment"]
    )

    env = TimeLimit(env, max_episode_steps=env_config["max_episode_steps"])
    return env

def main():
    # Path to your checkpoint directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "../checkpoints/ppo_constrained_10M")
    model_path = os.path.join(checkpoint_dir, "model_8500000_steps")
    
    # Load the configuration used during training
    print(f"Loading config from {checkpoint_dir}")
    config = load_config(checkpoint_dir)
    
    print("\n" + "="*60)
    print("EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Environment parameters:")
    for key, value in config["env"].items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Create the environment with the exact same parameters
    env = make_env(config, render_mode="human")
    
    # Load the pre-trained PPO model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    num_episodes = 15
    all_rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        episode_length = 0
        
        # Track reward components
        total_constraint_cost = 0.0
        total_ctrl_cost = 0.0
        total_contact_cost = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_length += 1
            
            # Accumulate costs if available in info
            if 'constraint_cost' in info:
                total_constraint_cost += info['constraint_cost']
            if 'ctrl_cost' in info:
                total_ctrl_cost += info['ctrl_cost']
            if 'contact_cost' in info:
                total_contact_cost += info['contact_cost']
            
            # Print progress every 200 steps
            if episode_length % 200 == 0:
                print(f"  Step {episode_length}: reward={reward:.3f}, cumulative={total_reward:.3f}")
        
        all_rewards.append(total_reward)
        
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{num_episodes}:")
        print(f"  Total Reward: {total_reward:.3f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Average Reward/Step: {total_reward/episode_length:.3f}")
        if total_constraint_cost > 0:
            print(f"  Total Constraint Cost: {total_constraint_cost:.3f}")
        if total_ctrl_cost > 0:
            print(f"  Total Control Cost: {total_ctrl_cost:.3f}")
        if total_contact_cost > 0:
            print(f"  Total Contact Cost: {total_contact_cost:.3f}")
        print(f"{'='*60}\n")
    
    # Summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(all_rewards):.3f}")
    print(f"Std reward: {np.std(all_rewards):.3f}")
    print(f"Min reward: {np.min(all_rewards):.3f}")
    print(f"Max reward: {np.max(all_rewards):.3f}")
    print("="*60)
    
    env.close()

if __name__ == "__main__":
    main()