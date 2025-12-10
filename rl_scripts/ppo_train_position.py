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

def load_config(config_path: str = config_file):
    """
    Load training configuration from a YAML file
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_latest_checkpoint(checkpoint_dir: str = "checkpoints/"):
    """
    Get the latest checkpoint file from the checkpoint directory
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

def get_latest_vecnormalize(checkpoint_dir: str = "checkpoints/"):
    """
    Get the latest VecNormalize file from the checkpoint directory
    """
    vecnorm_files = glob.glob(os.path.join(checkpoint_dir, "vec_normalize_*_steps.pkl"))
    if not vecnorm_files:
        return None
    
    vecnorm_files.sort(key=lambda x: extract_step_number(x))
    return vecnorm_files[-1]

def extract_step_number(checkpoint_path: str) -> int:
    """
    Extract the step number from a checkpoint filename
    """
    filename = os.path.basename(checkpoint_path)
    # Find all sequences of digits
    numbers = re.findall(r'\d+', filename)

    if not numbers:
        raise ValueError(f"Could not extract step number from checkpoint: {checkpoint_path}")
    
    return int(max(numbers, key=int))

class RewardComponentLogger(BaseCallback):
    """
    Custom callback to log individual reward components during training
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access the infos from the vectorized environment
        infos = self.locals.get('infos', [])
        for info in infos:
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
            
        return True

class SaveVecNormalizeCallback(BaseCallback):
    """
    Custom callback to save the VecNormalize stats alongside model checkpoints
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose=0):
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
    Gym Wrapper to integrate curriculum learning into the environment
    """
    def __init__(self, env: gym.Env, curriculum_manager):
        super().__init__(env)
        self.curriculum_manager = curriculum_manager
        self.env.unwrapped.set_curriculum_manager(curriculum_manager)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # Define success: reached within 0.15m of goal
            success = "goal_reached" in info.get('termination_reasons', [])
            # Record in curriculum
            self.curriculum_manager.record_episode(success)
            # Add curriculum info to logging
            info.update(self.curriculum_manager.get_curriculum_info())

        return obs, reward, terminated, truncated, info

def make_curriculum_env(env_kwargs, curriculum_manager, max_episode_steps):
    """
    Factory function to create environment with curriculum wrapper
    """
    def _init():
        env = HSAEnv(**env_kwargs)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = CurriculumWrapper(env, curriculum_manager)
        return env
    return _init

class SaveCurriculumCallback(BaseCallback):
    """Custom callback to save curriculum state alongside model checkpoints"""
    def __init__(self, curriculum_manager, save_freq: int, save_path: str, 
                 name_prefix: str = "curriculum", verbose=0):
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
    
def get_latest_curriculum(checkpoint_dir: str = "checkpoints/"):
    """Get the latest curriculum state file"""
    curriculum_files = glob.glob(os.path.join(checkpoint_dir, "curriculum_*_steps.pkl"))
    if not curriculum_files:
        return None
    
    curriculum_files.sort(key=lambda x: extract_step_number(x))
    return curriculum_files[-1]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    config = load_config(config_path)

    base_checkpoint_dir = config["train"]["checkpoint_dir"]
    run_name = config["train"]["run_name"]
    checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    checkpoint_freq = config["train"]["checkpoint_freq"]
    resume = config["train"].get("resume", False)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save a copy of the config file to the checkpoint directory
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
            "stagnation_penalty_weight": config["env"]["stagnation_penalty_weight"]
        }

    if use_curriculum:
        env = SubprocVecEnv([
            make_curriculum_env(
                env_kwargs=env_kwargs,
                curriculum_manager=curriculum,
                max_episode_steps=config["env"]["max_episode_steps"]
            )
            for _ in range(config["env"]["n_envs"])
        ])
    else:
        # Create standard vectorized environment without curriculum
        env = make_vec_env(
            HSAEnv,
            n_envs=config["env"]["n_envs"],
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs,
            wrapper_class=TimeLimit,
            wrapper_kwargs={"max_episode_steps": config["env"]["max_episode_steps"]},
        )

    # Keep track of episode statistics
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
                if latest_vecnorm:
                    print(f"[Resume] Loading VecNormalize stats from: {latest_vecnorm}")
                    env = VecNormalize.load(latest_vecnorm, env)
                else:
                    print(f"[Resume] WARNING: No VecNormalize stats found!")
                    print(f"[Resume] Starting with fresh normalization stats.")

                if use_curriculum and latest_curriculum:
                    print(f"[Resume] Loading curriculum state from: {latest_curriculum}")
                    curriculum.load(latest_curriculum)
                elif use_curriculum:
                    print(f"[Resume] WARNING: No curriculum state found!")
                    print(f"[Resume] Curriculum will start from initial settings.")
                    print(f"[Resume] This may cause performance degradation!")

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
    # Train the Model for a few timesteps
    model.learn(
        total_timesteps=timesteps_to_train,
        tb_log_name=config["train"]["run_name"],
        callback=callbacks,
        reset_num_timesteps=reset_timesteps
    )

    # Save the trained model
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
