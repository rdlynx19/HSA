import gymnasium as gym
import numpy as np
import yaml
import os, re, glob
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit
from hsa_gym.envs.hsa_constrained import HSAEnv

def extract_step_number(filepath: str) -> int:
    """
    Extract the step number from a checkpoint filename.
    """
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    if not numbers:
        return 0
    return int(max(numbers, key=int))

def find_matching_vecnormalize(checkpoint_dir: str, model_path: str) -> str:
    """
    Find the VecNormalize stats file that matches the model checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_path: Path to model checkpoint
    
    Returns:
        Path to matching VecNormalize file, or None if not found
    """
    try:
        model_steps = extract_step_number(model_path)
    except:
        model_steps = None
    
    # Strategy 1: Look for exact match by step number
    if model_steps:
        exact_match = os.path.join(checkpoint_dir, f"vec_normalize_{model_steps}_steps.pkl")
        if os.path.exists(exact_match):
            print(f"[VecNormalize] Found exact match: {exact_match}")
            return exact_match
    
    # Strategy 2: Look for "final" version
    final_path = os.path.join(checkpoint_dir, "vec_normalize_final.pkl")
    if os.path.exists(final_path):
        print(f"[VecNormalize] Found final version: {final_path}")
        return final_path
    
    # Strategy 3: Get the latest VecNormalize file
    vecnorm_files = glob.glob(os.path.join(checkpoint_dir, "vec_normalize_*_steps.pkl"))
    if vecnorm_files:
        vecnorm_files.sort(key=lambda x: extract_step_number(x))
        latest = vecnorm_files[-1]
        print(f"[VecNormalize] Using latest available: {latest}")
        return latest
    
    print(f"[VecNormalize] WARNING: No VecNormalize stats found in {checkpoint_dir}")
    return None
    

def load_config(checkpoint_dir: str):
    config_path = os.path.join(checkpoint_dir, "used_config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_env(config, render_mode="human"):
    env_config = config["env"]
    env = HSAEnv(
        render_mode=render_mode,
        xml_file=env_config["xml_file"],
        actuator_group=env_config["actuator_group"],
        action_group=env_config["action_group"],
        forward_reward_weight=env_config["forward_reward_weight"],
        ctrl_cost_weight=env_config["ctrl_cost_weight"],
        contact_cost_weight=env_config["contact_cost_weight"],
        yvel_cost_weight=env_config["yvel_cost_weight"],
        constraint_cost_weight=env_config["constraint_cost_weight"],
        acc_cost_weight=env_config["acc_cost_weight"],
        smooth_positions=env_config["smooth_positions"],
        frame_skip=env_config["frame_skip"],
        max_increment=env_config["max_increment"],
        enable_terrain=env_config.get("enable_terrain", False),
        terrain_type=env_config.get("terrain_type", "flat"),
        early_termination_penalty=env_config.get("early_termination_penalty", 0.0),
        alive_bonus=env_config.get("alive_bonus", 0.0),
        goal_position=env_config.get("goal_position", None),
        distance_reward_weight=env_config.get("distance_reward_weight", 0.0),
    )
    env = TimeLimit(env, max_episode_steps=env_config["max_episode_steps"])


    return env

def wrapped_angle_diff(a, c):
    """
    Calculate the wrapped angle difference.
    """
    raw_diff = a - c
    wrapped = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))
    return abs(wrapped)

def analyze_actions(checkpoint_dir, model_path, num_episodes=5):
    """
    Analyze and plot the action space usage during evaluation
    """
    print("Loading configuration and model...")
    config = load_config(checkpoint_dir)

    base_env = make_env(config, render_mode="human")
    env = DummyVecEnv([lambda: base_env])
    vecnorm_path = find_matching_vecnormalize(checkpoint_dir, model_path)
    if vecnorm_path:
        print(f"\n[VecNormalize] Loading normalization stats from:")
        print(f"  {vecnorm_path}")

        env = VecNormalize.load(vecnorm_path, env)
        # Ensure the environment is in evaluation mode
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, training=False, norm_reward=False)

    model = PPO.load(model_path, env=env)

    
    # Storage for data
    all_actions = []
    all_action_changes = []
    all_joint_positions = []
    all_joint_velocities = []
    all_constraint_diffs = []
    
    actuator_names = ['1A', '2A', '3A', '4A', '1C', '2C', '3C', '4C']
    qpos_indices = [7, 22, 10, 24, 20, 9, 21, 12]
    qvel_indices = [6, 20, 9, 22, 18, 8, 19, 11]
    
    print(f"\nRunning {num_episodes} episodes to collect data...")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_actions = []
        episode_joint_pos = []
        episode_joint_vel = []
        episode_constraint_diffs = []
        prev_action = None
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_array, info = env.step(action)
            done = done_array[0]
            if done:
                print(f"    Episode terminated at step {step_count} with termination reasons: {info[0]['termination_reasons']}")
            step_count += 1
            # Store action
            episode_actions.append(action[0].copy())

            # Store action change
            if prev_action is not None:
                action_change = np.abs(action[0] - prev_action)
                all_action_changes.append(action_change)
            prev_action = action[0].copy()

            unwrapped_env = env.envs[0].unwrapped
            # Store joint positions and velocities if available
            if hasattr(unwrapped_env, 'data'):
                data = unwrapped_env.data
                # Get joint positions for actuated joints
                joint_pos = [data.qpos[idx] for idx in qpos_indices]
                joint_vel = [data.qvel[idx] for idx in qvel_indices]
                
                # Calculate constraint differences with CORRECT pairing and wrapping
                # Order in joint_pos: [1A, 2A, 3A, 4A, 1C, 2C, 3C, 4C]
                # Pairs: (0,4)=1A-1C, (1,5)=2A-2C, (2,6)=3A-3C, (3,7)=4A-4C

                episode_joint_pos.append(joint_pos)
                episode_joint_vel.append(joint_vel)
                

                # Calculate constraint differences
                diffs = [
                    abs(joint_pos[0] - joint_pos[4]),  # 1A - 1C
                    abs(joint_pos[1] - joint_pos[5]),  # 2A - 2C
                    abs(joint_pos[2] - joint_pos[6]),  # 3A - 3C
                    abs(joint_pos[3] - joint_pos[7]),  # 4A - 4C
                ]
                episode_constraint_diffs.append(diffs)
        
        all_actions.extend(episode_actions)
        all_joint_positions.extend(episode_joint_pos)
        all_joint_velocities.extend(episode_joint_vel)
        all_constraint_diffs.extend(episode_constraint_diffs)
        
        print(f"  Episode {ep+1}/{num_episodes} completed - {step_count} steps")
    
    # Convert to numpy arrays
    all_actions = np.array(all_actions)
    all_action_changes = np.array(all_action_changes)
    all_joint_positions = np.array(all_joint_positions)
    all_joint_velocities = np.array(all_joint_velocities)
    all_constraint_diffs = np.array(all_constraint_diffs)
    
    print("\nGenerating plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ============================================================
    # 1. Action Distribution (Histograms)
    # ============================================================
    for i in range(8):
        ax = plt.subplot(4, 4, i+1)
        ax.hist(all_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Action {actuator_names[i]}')
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Frequency')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Zero')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_distributions.png', dpi=300, bbox_inches='tight')
    print("  Saved: action_distributions.png")
    plt.close()
    
    # ============================================================
    # 2. Action Time Series
    # ============================================================
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Action Time Series', fontsize=16)
    
    for i in range(8):
        row = i // 2
        col = i % 2
        axes[row, col].plot(all_actions[:, i], linewidth=0.5)
        axes[row, col].set_title(f'{actuator_names[i]}')
        axes[row, col].set_xlabel('Timestep')
        axes[row, col].set_ylabel('Action')
        axes[row, col].axhline(0, color='r', linestyle='--', alpha=0.3)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_timeseries.png', dpi=300, bbox_inches='tight')
    print("  Saved: action_timeseries.png")
    plt.close()
    
    # ============================================================
    # 3. Action Change Magnitudes
    # ============================================================
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Action Change Magnitudes (Smoothness)', fontsize=16)
    
    for i in range(8):
        row = i // 2
        col = i % 2
        axes[row, col].plot(all_action_changes[:, i], linewidth=0.5, alpha=0.7)
        axes[row, col].set_title(f'{actuator_names[i]} - Mean: {np.mean(all_action_changes[:, i]):.4f}')
        axes[row, col].set_xlabel('Timestep')
        axes[row, col].set_ylabel('|Action Change|')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_smoothness.png', dpi=300, bbox_inches='tight')
    print("  Saved: action_smoothness.png")
    plt.close()
    
    # ============================================================
    # 4. Joint Positions
    # ============================================================
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Joint Positions', fontsize=16)
    
    for i in range(8):
        row = i // 2
        col = i % 2
        axes[row, col].plot(all_joint_positions[:, i], linewidth=0.5)
        axes[row, col].set_title(f'{actuator_names[i]}')
        axes[row, col].set_xlabel('Timestep')
        axes[row, col].set_ylabel('Position (rad)')
        axes[row, col].axhline(0, color='r', linestyle='--', alpha=0.3)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joint_positions.png', dpi=300, bbox_inches='tight')
    print("  Saved: joint_positions.png")
    plt.close()
    
    # ============================================================
    # 5. Joint Velocities
    # ============================================================
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Joint Velocities', fontsize=16)
    
    for i in range(8):
        row = i // 2
        col = i % 2
        axes[row, col].plot(all_joint_velocities[:, i], linewidth=0.5)
        axes[row, col].set_title(f'{actuator_names[i]} - Max: {np.max(np.abs(all_joint_velocities[:, i])):.2f} rad/s')
        axes[row, col].set_xlabel('Timestep')
        axes[row, col].set_ylabel('Velocity (rad/s)')
        axes[row, col].axhline(0, color='r', linestyle='--', alpha=0.3)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joint_velocities.png', dpi=300, bbox_inches='tight')
    print("  Saved: joint_velocities.png")
    plt.close()
    
    # ============================================================
    # 6. Constraint Violations (|A - C| for each pair)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Constraint Monitoring: |A - C| per Pair', fontsize=16)
    
    pair_names = ['Pair 1 (1A-1C)', 'Pair 2 (2A-2C)', 'Pair 3 (3A-3C)', 'Pair 4 (4A-4C)']
    
    for i in range(4):
        row = i // 2
        col = i % 2
        axes[row, col].plot(all_constraint_diffs[:, i], linewidth=0.5)
        axes[row, col].axhline(np.pi, color='r', linestyle='--', linewidth=2, label='π limit')
        axes[row, col].set_title(f'{pair_names[i]} - Max: {np.max(all_constraint_diffs[:, i]):.3f} rad')
        axes[row, col].set_xlabel('Timestep')
        axes[row, col].set_ylabel('|A - C| (rad)')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim([0, np.pi + 0.5])
    
    plt.tight_layout()
    plt.savefig('constraint_violations.png', dpi=300, bbox_inches='tight')
    print("  Saved: constraint_violations.png")
    plt.close()
    
    # ============================================================
    # 7. Summary Statistics
    # ============================================================
    print("\n" + "="*60)
    print("FEASIBILITY ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nAction Statistics:")
    for i in range(8):
        print(f"  {actuator_names[i]:3s}: mean={np.mean(all_actions[:, i]):6.3f}, "
              f"std={np.std(all_actions[:, i]):6.3f}, "
              f"min={np.min(all_actions[:, i]):6.3f}, "
              f"max={np.max(all_actions[:, i]):6.3f}")
    
    print("\nAction Smoothness (Mean Absolute Change):")
    for i in range(8):
        print(f"  {actuator_names[i]:3s}: {np.mean(all_action_changes[:, i]):.4f} rad/step")
    
    print("\nJoint Velocity Statistics (rad/s):")
    for i in range(8):
        max_vel = np.max(np.abs(all_joint_velocities[:, i]))
        mean_vel = np.mean(np.abs(all_joint_velocities[:, i]))
        print(f"  {actuator_names[i]:3s}: max={max_vel:6.2f}, mean={mean_vel:6.2f}")
    
    print("\nConstraint Violations:")
    for i in range(4):
        max_diff = np.max(all_constraint_diffs[:, i])
        violations = np.sum(all_constraint_diffs[:, i] > np.pi)
        violation_pct = (violations / len(all_constraint_diffs)) * 100
        print(f"  {pair_names[i]:15s}: max_diff={max_diff:.3f} rad, "
              f"violations={violations} ({violation_pct:.2f}%)")
    
    print("\n" + "="*60)
    
    env.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Demo 1 No Curriculum
    # checkpoint_dir = os.path.join(script_dir, "../checkpoints/ppo_random")
    # model_path = os.path.join(checkpoint_dir, "model_27000000_steps.zip")

    # Demo 2 Craters
    checkpoint_dir = os.path.join(script_dir, "../checkpoints/ppo_curriculum_waves")
    model_path = os.path.join(checkpoint_dir, "model_500000_steps.zip")

    # # # Demo 3 Flat
    # checkpoint_dir = os.path.join(script_dir, "../checkpoints/ppo_curriculum_flat_small")
    # model_path = os.path.join(checkpoint_dir, "model_39000000_steps.zip")


    analyze_actions(checkpoint_dir, model_path, num_episodes=10)