"""
PPO Training Script for HSA Environment with Curriculum Learning.

This module provides the main entry point (`main` function) for training a Stable 
Baselines3 (SB3) PPO agent on the HSAEnv. It handles configuration loading, 
environment vectorization, observation/reward normalization, dynamic checkpoint 
resumption, and implementation of a custom goal-based curriculum manager 
integrated via Gym wrappers and callbacks.
"""
import yaml, os, glob, shutil, re

import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from hsa_gym.envs.curriculum_manager import GoalCurriculumManager


from hsa_gym.envs.hsa_constrained import HSAEnv

config_file = "./configs/ppo_position.yaml"
xml_file = "../hsa_gym/envs/assets"

def load_config(config_path: str = config_file) -> dict:
    """
    Load training configuration parameters from a YAML file.

    :param config_path: Path to the YAML configuration file.
    :type config_path: str
    :returns: A dictionary containing the parsed configuration settings.
    :rtype: dict
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_latest_checkpoint(checkpoint_dir: str = "checkpoints/") -> str | None:
    """
    Get the path to the latest PPO model checkpoint file from a directory.

    It filters out 'final' models and uses the extracted step number for sorting.

    :param checkpoint_dir: Directory containing model checkpoints.
    :type checkpoint_dir: str
    :returns: The file path of the latest checkpoint, or None if no valid checkpoints are found.
    :rtype: str or None
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*_steps.zip"))
    if not checkpoint_files:
        return None
    
    # Filter out the final model if it exists
    checkpoint_files = [f for f in checkpoint_files if 'final' not in f.lower()]
    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: extract_step_number(x))
    return checkpoint_files[-1]

def get_latest_vecnormalize(checkpoint_dir: str = "checkpoints/") -> str | None:
    """
    Get the path to the latest VecNormalize statistics file from a directory.

    :param checkpoint_dir: Directory containing VecNormalize files.
    :type checkpoint_dir: str
    :returns: The file path of the latest VecNormalize file, or None if none are found.
    :rtype: str or None
    """
    vecnorm_files = glob.glob(os.path.join(checkpoint_dir, "vec_normalize_*_steps.pkl"))
    if not vecnorm_files:
        return None
    
    vecnorm_files.sort(key=lambda x: extract_step_number(x))
    return vecnorm_files[-1]

def extract_step_number(checkpoint_path: str) -> int:
    """
    Extract the step number (timestep count) from a checkpoint filename.

    The step number is assumed to be the largest sequence of digits in the filename.

    :param checkpoint_path: The full file path of the checkpoint or VecNormalize file.
    :type checkpoint_path: str
    :returns: The extracted total timestep count.
    :rtype: int
    :raises ValueError: If no step number could be extracted from the filename.
    """
    filename = os.path.basename(checkpoint_path)
    # Find all sequences of digits
    numbers = re.findall(r'\d+', filename)

    if not numbers:
        raise ValueError(f"Could not extract step number from checkpoint: {checkpoint_path}")
    
    # Return the largest number found (assumed to be the timestep)
    return int(max(numbers, key=int))

class RewardComponentLogger(BaseCallback):
    """
    Custom Stable Baselines3 callback to log individual reward and cost components 
    (e.g., forward reward, control cost) contained within the environment's `info` 
    dictionary at every step.

    It also logs curriculum progress and checkpoint status.

    :param verbose: Verbosity level (0 for silent, 1 for info).
    :type verbose: int
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access the infos from the vectorized environment
        infos = self.locals.get('infos', [])
        for info in infos:
            # Reward logging
            if 'reward_forward' in info:
                self.logger.record('reward/forward', info['reward_forward'])
            if 'reward_ctrl_cost' in info:
                self.logger.record('reward/ctrl_cost', info['reward_ctrl_cost'])
            if 'reward_contact_cost' in info:
                self.logger.record('reward/contact_cost', info['reward_contact_cost'])
            if 'reward_lateral' in info:
                self.logger.record('reward/lateral', info['reward_lateral'])
            if 'reward_constraint_cost' in info:
                self.logger.record('reward/constraint_cost', info['reward_constraint_cost'])
            if 'reward_constraint_bonus' in info:
                self.logger.record('reward/constraint_bonus', info['reward_constraint_bonus'])
            if 'reward_acc_cost' in info:
                self.logger.record('reward/acc_cost', info['reward_acc_cost'])
            if 'reward_joint_vel_cost' in info:
                self.logger.record('reward/joint_vel_cost', info['reward_joint_vel_cost'])
            if 'reward_distance' in info:
                self.logger.record('reward/distance', info['reward_distance'])
            if 'reward_stagnation_penalty' in info:
                self.logger.record('reward/stagnation_penalty', info['reward_stagnation_penalty'])
            if 'reward_total_costs' in info:
                self.logger.record('reward/total_costs', info['reward_total_costs'])

            # Curriculum logging
            if 'curriculum/max_distance' in info:
                self.logger.record('curriculum/max_distance', info['curriculum/max_distance'])
            if 'curriculum/success_rate' in info:
                self.logger.record('curriculum/success_rate', info['curriculum/success_rate'])
            if 'curriculum/progress' in info:
                self.logger.record('curriculum/progress', info['curriculum/progress'])
            if 'curriculum/final_distance' in info:
                self.logger.record('curriculum/final_distance', info['curriculum/final_distance'])
            if 'curriculum/success_threshold' in info:
                self.logger.record('curriculum/success_threshold', info['curriculum/success_threshold'])
            
            # Checkpoints logging
            if 'checkpoints_collected' in info:
                self.logger.record('checkpoints/collected', info['checkpoints_collected'])
            if 'total_checkpoints' in info:
                self.logger.record('checkpoints/total', info['total_checkpoints'])
            if 'checkpoint_progress' in info:
                self.logger.record('checkpoints/progress', info['checkpoint_progress'])
        return True

class SaveVecNormalizeCallback(BaseCallback):
    """
    Custom Stable Baselines3 callback to save the VecNormalize observation/reward 
    statistics file alongside model checkpoints.

    :param save_freq: How often (in environment steps, counts across all environments) to save the file.
    :type save_freq: int
    :param save_path: Directory where the file should be saved.
    :type save_path: str
    :param name_prefix: Prefix for the saved filename (default: 'vec_normalize').
    :type name_prefix: str
    :param verbose: Verbosity level.
    :type verbose: int
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if isinstance(self.training_env, VecNormalize):
                # Save with same timestep number as model checkpoint
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
                self.training_env.save(path)
                if self.verbose > 0:
                    print(f"[VecNormalize] Saved to {path} at step {self.num_timesteps}")
        return True

class CurriculumWrapper(gym.Wrapper):
    """
    Gym Wrapper to integrate the GoalCurriculumManager logic into the environment's 
    `step` and `reset` lifecycle.

    This wrapper ensures the environment's internal goal sampling uses the curriculum 
    manager's state and records episode outcomes to trigger curriculum expansion/contraction.

    :param env: The Gymnasium environment instance to wrap.
    :type env: gymnasium.Env
    :param curriculum_manager: The curriculum manager instance.
    :type curriculum_manager: GoalCurriculumManager
    """
    def __init__(self, env: gym.Env, curriculum_manager: GoalCurriculumManager):
        super().__init__(env)
        self.curriculum_manager = curriculum_manager
        # Link the curriculum manager to the base environment's sampling logic
        self.env.unwrapped.set_curriculum_manager(curriculum_manager)

    def reset(self, **kwargs):
        """Resets the environment."""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Steps the environment and updates the curriculum manager upon episode termination.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # Define success: based on 'goal_reached' in termination reasons
            success = "goal_reached" in info.get('termination_reasons', [])
            # Record outcome in curriculum manager
            self.curriculum_manager.record_episode(success)
            # Add current curriculum statistics to the info dictionary for logging
            info.update(self.curriculum_manager.get_curriculum_info())

        return obs, reward, terminated, truncated, info

def make_curriculum_env(env_kwargs: dict, curriculum_manager: GoalCurriculumManager, max_episode_steps: int):
    """
    Factory function to create a single environment instance wrapped with both 
    TimeLimit and the CurriculumWrapper.

    :param env_kwargs: Dictionary of keyword arguments for initializing HSAEnv.
    :type env_kwargs: dict
    :param curriculum_manager: The pre-initialized curriculum manager instance.
    :type curriculum_manager: GoalCurriculumManager
    :param max_episode_steps: The maximum number of steps before environment truncation.
    :type max_episode_steps: int
    :returns: A function (`_init`) that returns the wrapped environment instance.
    :rtype: callable
    """
    def _init():
        env = HSAEnv(**env_kwargs)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = CurriculumWrapper(env, curriculum_manager)
        return env
    return _init

class SaveCurriculumCallback(BaseCallback):
    """
    Custom Stable Baselines3 callback to save the state of the GoalCurriculumManager 
    alongside model checkpoints, allowing for seamless curriculum resumption.

    :param curriculum_manager: The curriculum manager instance, or None if curriculum is disabled.
    :type curriculum_manager: GoalCurriculumManager or None
    :param save_freq: How often (in environment steps) to save the file.
    :type save_freq: int
    :param save_path: Directory where the file should be saved.
    :type save_path: str
    :param name_prefix: Prefix for the saved filename.
    :type name_prefix: str
    :param verbose: Verbosity level.
    :type verbose: int
    """
    def __init__(self, curriculum_manager: GoalCurriculumManager | None, save_freq: int, save_path: str, 
                 name_prefix: str = "curriculum", verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
    
    def _on_step(self) -> bool:
        if self.curriculum_manager is not None and self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.curriculum_manager.save(path)
            if self.verbose > 0:
                print(f"[Curriculum] Saved to {path} at step {self.num_timesteps}")
        return True
    
def get_latest_curriculum(checkpoint_dir: str = "checkpoints/") -> str | None:
    """
    Get the path to the latest saved curriculum state file.

    :param checkpoint_dir: Directory containing curriculum state files.
    :type checkpoint_dir: str
    :returns: The file path of the latest curriculum state file, or None if none are found.
    :rtype: str or None
    """
    curriculum_files = glob.glob(os.path.join(checkpoint_dir, "curriculum_*_steps.pkl"))
    if not curriculum_files:
        return None
    
    curriculum_files.sort(key=lambda x: extract_step_number(x))
    return curriculum_files[-1]

def main():
    """
    The main function to set up, resume, and run the Stable Baselines3 PPO training loop.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    config = load_config(config_path)

    base_checkpoint_dir = config["train"]["checkpoint_dir"]
    run_name = config["train"]["run_name"]
    checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    checkpoint_freq = config["train"]["checkpoint_freq"]
    resume = config["train"].get("resume", False)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save configuration files for archival
    shutil.copy(config_path, os.path.join(checkpoint_dir, "used_config.yaml"))
    print(f"[Config] Training configuration saved to {os.path.join(checkpoint_dir, 'used_config.yaml')}")
    
    xml_model = config["env"]["xml_file"]
    xml_path = os.path.join(script_dir, xml_file, xml_model)
    shutil.copy(xml_path, os.path.join(checkpoint_dir, "used_model.xml"))
    print(f"[Config] XML model file saved to {os.path.join(checkpoint_dir, 'used_model.xml')}")

    # Create curriculum manager if enabled
    curriculum_config = config.get("curriculum", {})
    use_curriculum = curriculum_config.get("enabled", False)
    if use_curriculum:
        curriculum = GoalCurriculumManager(
            initial_range=tuple(curriculum_config.get("initial_range", 
                                                      [1.5, 2.0])),
            target_range=tuple(curriculum_config.get("target_range", 
                                                     [1.5, 4.5])),
            success_threshold=curriculum_config.get("success_threshold", 0.85),
            failure_threshold=curriculum_config.get("failure_threshold", 0.50),
            expansion_step=curriculum_config.get("expansion_step", 0.3),
            window_size=curriculum_config.get("window_size", 100),
            min_episodes_before_expand=curriculum_config.get("min_episodes_before_expand", 50),
            dead_zone_radius=curriculum_config.get("dead_zone_radius", 1.2)
        )
    else:
        curriculum = None
        print(f"[Curriculum] Curriculum learning is disabled - using fixed goal sampling.")

    # Environment keyword arguments dictionary
    env_kwargs={
            "xml_file": config["env"]["xml_file"],
            "actuator_group": config["env"]["actuator_group"],
            "action_group": config["env"]["action_group"],
            "forward_reward_weight": config["env"]["forward_reward_weight"],
            "ctrl_cost_weight": config["env"]["ctrl_cost_weight"],
            "contact_cost_weight": config["env"]["contact_cost_weight"],
            "smooth_positions": config["env"]["smooth_positions"],
            "frame_skip": config["env"]["frame_skip"],
            "yvel_cost_weight": config["env"]["yvel_cost_weight"],
            "constraint_cost_weight": config["env"]["constraint_cost_weight"],
            "acc_cost_weight": config["env"]["acc_cost_weight"],
            "joint_vel_cost_weight": config["env"]["joint_vel_cost_weight"],
            "max_increment": config["env"]["max_increment"],
            "early_termination_penalty": config["env"]["early_termination_penalty"],
            "alive_bonus": config["env"]["alive_bonus"],
            "enable_terrain": config["env"]["enable_terrain"],
            "terrain_type": config["env"]["terrain_type"],
            "goal_position": config["env"]["goal_position"],
            "distance_reward_weight": config["env"]["distance_reward_weight"],
            "stagnation_penalty_weight": config["env"]["stagnation_penalty_weight"],
            "num_turns": config["env"].get("num_turns", 2.0),
            "start_radius": config["env"].get("start_radius", 0.5),
            "end_radius": config["env"].get("end_radius", 4.0),
            "num_checkpoints": config["env"].get("num_checkpoints", 8),
            "max_episode_steps": config["env"]["max_episode_steps"],
            "checkpoint_reward": config["env"].get("checkpoint_reward", 30.0),
            "checkpoint_radius": config["env"].get("checkpoint_radius", 0.4),
            "ensure_flat_spawn": config["env"].get("ensure_flat_spawn", True)
        }

    # Setup Vectorized Environment
    if use_curriculum:
        # Create environment instances with CurriculumWrapper
        env = SubprocVecEnv([
            make_curriculum_env(
                env_kwargs=env_kwargs,
                curriculum_manager=curriculum,
                max_episode_steps=config["env"]["max_episode_steps"]
            )
            for _ in range(config["env"]["n_envs"])
        ])
    else:
        # Create standard vectorized environment (TimeLimit is applied via wrapper_class)
        env = make_vec_env(
            HSAEnv,
            n_envs=config["env"]["n_envs"],
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs,
            wrapper_class=TimeLimit,
            wrapper_kwargs={"max_episode_steps": config["env"]["max_episode_steps"]},
        )

    # Monitor wrapper to track episode statistics
    env = VecMonitor(env)

    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config["model"]["gamma"],
        epsilon=1e-8,
        norm_obs_keys=None
    )

    # Load model if resuming from checkpoint
    model = None
    trained_steps = 0
    reset_timesteps = True # Default for fresh training

    if resume:
        latest = get_latest_checkpoint(checkpoint_dir)
        latest_vecnorm = get_latest_vecnormalize(checkpoint_dir)
        latest_curriculum = get_latest_curriculum(checkpoint_dir)
        if latest:
            print(f"[Resume] Loading latest checkpoint: {latest}")

            try:
                trained_steps = extract_step_number(latest)
                print(f"[Resume] Model has been trained for {trained_steps} steps.")
                
                # Load VecNormalize state
                if latest_vecnorm:
                    print(f"[Resume] Loading VecNormalize stats from: {latest_vecnorm}")
                    env = VecNormalize.load(latest_vecnorm, env)
                else:
                    print(f"[Resume] WARNING: No VecNormalize stats found! Starting with fresh normalization stats.")

                # Load Curriculum state
                if use_curriculum and latest_curriculum:
                    print(f"[Resume] Loading curriculum state from: {latest_curriculum}")
                    curriculum.load(latest_curriculum)
                elif use_curriculum:
                    print(f"[Resume] WARNING: No curriculum state found! Curriculum will start from initial settings. This may cause performance degradation!")

                print(f"[Resume] Loading model ...")
                model = PPO.load(latest, env=env, device='cpu')
                reset_timesteps = False # Keep the original timestep count

            except ValueError as e:
                print(f"[Resume] Warning: {e}. Assuming 0 trained steps.")
                resume = False
        else:
            print(f"[Resume] WARNING: No checkpoint found in {checkpoint_dir}, starting fresh training.")
            resume = False

    if use_curriculum:
        # Display curriculum status
        print(f"\n{'='*60}")
        print(f"CURRICULUM LEARNING ENABLED")
        print(f"{'='*60}")
        print(f"Current range:    {curriculum.min_distance:.1f}m - {curriculum.current_max_distance:.1f}m")
        print(f"Target range:     {curriculum.target_range[0]:.1f}m - {curriculum.target_range[1]:.1f}m")
        print(f"Success threshold: {curriculum.success_threshold:.0%}")
        print(f"Failure threshold: {curriculum.failure_threshold:.0%}")
        print(f"Episodes tracked: {curriculum.episode_count}")
        if curriculum.recent_successes:
            print(f"Recent success rate: {np.mean(curriculum.recent_successes):.1%} (last {len(curriculum.recent_successes)} episodes)")
        print(f"{'='*60}\n")

    if model is None: 
        # Initialize the PPO model
        model = PPO(
            policy=config["model"]["policy"],
            env=env,
            verbose=1,
            n_steps=config["model"]["n_steps"],
            batch_size=config["model"]["batch_size"],
            learning_rate=config["model"]["learning_rate"],
            gamma=config["model"]["gamma"],
            ent_coef=config["model"]["ent_coef"],
            clip_range=config["model"]["clip_range"],
            tensorboard_log=config["train"]["log_dir"],
            device='cpu'
        )

    # Calculate remaining timesteps
    total_timesteps = config["train"]["total_timesteps"]
    if resume and trained_steps > 0:
        remaining_timesteps = total_timesteps - trained_steps

        if remaining_timesteps <= 0:
            print(f"[Info] Model has already been trained for {trained_steps} steps.")
            print(f"[Info] Desired total timesteps: {total_timesteps}")
            print(f"[Info] No further training needed.")
            return
    
        print(f"\n{'='*60}")
        print(f"RESUMING TRAINING")
        print(f"{'='*60}")
        print(f"Already trained:     {trained_steps:,} steps")
        print(f"Total desired:       {total_timesteps:,} steps")
        print(f"Will train for:      {remaining_timesteps:,} additional steps")
        print(f"Reset timesteps:     {reset_timesteps}")
        print(f"Next checkpoint at:  {trained_steps + checkpoint_freq:,} steps")
        print(f"{'='*60}\n")

        timesteps_to_train = remaining_timesteps
    else:
        print(f"\n{'='*60}")
        print(f"STARTING FRESH TRAINING")
        print(f"{'='*60}")
        print(f"Will train for:      {total_timesteps:,} steps")
        print(f"Reset timesteps:     {reset_timesteps}")
        print(f"First checkpoint at: {checkpoint_freq:,} steps")
        print(f"{'='*60}\n")

        timesteps_to_train = total_timesteps
        

    # Setup Callbacks
    checkpoint_cb = CheckpointCallback(
        # Divide by number of envs because SB3 counts steps across all envs
        save_freq=checkpoint_freq // config["env"]["n_envs"],
        save_path=checkpoint_dir,
        name_prefix="model"
    )

    vecnorm_cb = SaveVecNormalizeCallback(
        save_freq=checkpoint_freq // config["env"]["n_envs"],
        save_path=checkpoint_dir,
        name_prefix="vec_normalize",
        verbose=1
    )

    curriculum_cb = SaveCurriculumCallback(
        curriculum_manager=curriculum if use_curriculum else None,
        save_freq=checkpoint_freq // config["env"]["n_envs"],
        save_path=checkpoint_dir,
        name_prefix="curriculum",
        verbose=1
    ) 

    reward_cb = RewardComponentLogger()
    callbacks = CallbackList([checkpoint_cb, reward_cb, vecnorm_cb, curriculum_cb])
    
    # Train the Model
    model.learn(
        total_timesteps=timesteps_to_train,
        tb_log_name=config["train"]["run_name"],
        callback=callbacks,
        reset_num_timesteps=reset_timesteps
    )

    # Final Save
    final_steps = trained_steps + timesteps_to_train
    print(f"[Info] Training complete. Total trained steps: {final_steps:,}")
    model.save(os.path.join(checkpoint_dir, 
                            f"{config['train']['run_name']}_final_{final_steps}_steps"))
    
    # Save the VecNormalize stats
    vecnorm_path = os.path.join(
        checkpoint_dir, 
        "vec_normalize_final.pkl"
    )
    env.save(vecnorm_path)
    print(f"[Info] VecNormalize stats saved to {vecnorm_path}")

if __name__ == "__main__":
    main()