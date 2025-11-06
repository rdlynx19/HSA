import yaml, os, glob

import gymnasium as gym

from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from hsa_gym.envs.hsa_v1 import HSAEnv
from hsa_gym.envs.sb3_wrapper import SB3Wrapper

class CheckpointCallback(BaseCallback):
    """
    Custom callback for saving model checkpoints.
    """
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_step_{self.num_timesteps}")
            self.model.save(save_file)
            if self.verbose:
                print(f"[Checkpoint] Saved model at step {self.num_timesteps} to {save_file}")
            if self.logger and hasattr(self.logger, 'writer'):
                writer = self.logger.writer
                writer.add_text('checkpoint', f'Saved at step {self.num_timesteps}', self.num_timesteps)
        return True

class EpisodeLoggerCallback(BaseCallback):
    """
    Custom callback for logging episode rewards and lengths.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # infos is a list of dicts for each env in VecEnv
        for info in self.locals['infos']:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_count += 1

                # Get the TensorBoard writer
                if self.logger and hasattr(self.logger, 'writer'):
                    writer = self.logger.writer

                    writer.add_scalar('episode/reward', ep_info['r'], self.episode_count)
                    writer.add_scalar('episode/length', ep_info['l'], self.episode_count)
        return True

def make_vec_env(actuator_groups: list[int] = [1], use_locks: bool = True, max_episode_steps: int = 2000):
    """
    Helper function to create a vectorized environment
    """
    def _init():
        env = HSAEnv(actuator_groups=actuator_groups,
                     use_locks=use_locks)
        # Wrap the environment for time limits
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        # Wrap the environment for SB3 compatibility
        env = SB3Wrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init

def load_config(config_path: str = "configs/ppo_hsa.yaml"):
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
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_step_*.zip"))
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoint_files[-1]

def main():
    config = load_config("configs/ppo_hsa.yaml")

    checkpoint_dir = config["train"]["checkpoint_dir"]
    checkpoint_freq = config["train"]["checkpoint_freq"]
    resume = config["train"].get("resume", False)
    # Create a vectorized environment with 4 parallel environments
    # Position control only, with lock control 
    env_fns = [
        make_vec_env(
            config["env"]["actuator_groups"], 
            config["env"]["use_locks"], 
            config["env"]["max_episode_steps"]
        )
        for _ in range(config["env"]["n_envs"])
    ]

    env = SubprocVecEnv(env_fns)
    # Keep track of episode statistics
    env = VecMonitor(env)

    # Load model if resuming from checkpoint
    model = None
    if resume:
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest:
            print(f"[Resume] Loading latest checkpoint: {latest}")
            model = PPO.load(latest, env=env)
        else:
            print(f"[Resume] WARNING: No checkpoint found in {checkpoint_dir}, starting fresh training.")

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
            tensorboard_log=config["train"]["log_dir"]
        )

    episode_logger = EpisodeLoggerCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
    )
    # Train the Model for a few timesteps
    model.learn(
        total_timesteps=config["train"]["total_timesteps"],
        tb_log_name=config["train"]["run_name"],
        callback=[episode_logger, checkpoint_cb]
    )

    # Save the trained model
    model.save(os.path.join(checkpoint_dir, f"{config['train']['run_name']}_final"))

if __name__ == "__main__":
    main()
