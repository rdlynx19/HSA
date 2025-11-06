import yaml

import gymnasium as gym

from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from hsa_gym.envs.hsa_v1 import HSAEnv
from hsa_gym.envs.sb3_wrapper import SB3Wrapper

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

def make_vec_env(actuator_groups: list[int] = [1], use_locks: bool = True, max_steps: int = 2000):
    """
    Helper function to create a vectorized environment
    """
    def _init():
        env = HSAEnv(actuator_groups=actuator_groups,
                     use_locks=use_locks)
        # Wrap the environment for time limits
        env = TimeLimit(env, max_episode_steps=max_steps)
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

def main():
    config = load_config("configs/ppo_hsa.yaml")

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

    callback = EpisodeLoggerCallback()
    # Train the Model for a few timesteps
    model.learn(
        total_timesteps=config["train"]["total_timesteps"],
        tb_log_name=config["train"]["run_name"],
        callback=callback
    )

    # Save the trained model
    model.save(config["train"]["run_name"])

if __name__ == "__main__":
    main()
