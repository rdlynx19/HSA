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

def make_vec_env(actuator_groups: list[int] = [1], use_locks: bool = False, n_envs: int = 1):
    """
    Helper function to create a vectorized environment
    """
    def _init():
        env = HSAEnv(actuator_groups=actuator_groups,
                     use_locks=use_locks)
        # Wrap the environment for time limits
        env = TimeLimit(env, max_episode_steps=500)
        # Wrap the environment for SB3 compatibility
        env = SB3Wrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init

def main():
    # Create a vectorized environment with 4 parallel environments
    n_envs = 4
    # Position control only, with lock control enabled
    actuator_groups = [1] 
    use_locks = True

    # Create the vectorized environment
    env = SubprocVecEnv([make_vec_env(actuator_groups, use_locks, i) for i in range(n_envs)])
    # Keep track of episode statistics
    env = VecMonitor(env)

    # Initialize the PPO model
    model = PPO("MlpPolicy", 
                env, verbose=1,
                n_steps=512,
                batch_size=64,
                learning_rate=3e-4,
                ent_coef=0.0,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log="./logs/ppo_hsa/",
                )

    callback = EpisodeLoggerCallback()
    # Train the Model for a few timesteps
    model.learn(total_timesteps=50000, 
                tb_log_name="ppo_hsa_setup",
                callback=callback)

    # Save the trained model
    model.save("ppo_hsa_setup")

if __name__ == "__main__":
    main()
