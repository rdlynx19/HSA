import yaml, os, glob, shutil

import gymnasium as gym

from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from hsa_gym.envs.hsa_position import HSAEnv

config_file = "./configs/ppo_position.yaml"

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
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_step_*.zip"))
    if not checkpoint_files:
        return None
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoint_files[-1]

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
    # Create a vectorized environment with 4 parallel environments
    # Position control only
    env = make_vec_env(
        HSAEnv,
        n_envs=config["env"]["n_envs"],
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "xml_file": config["env"]["xml_file"],
            "actuator_group": config["env"]["actuator_group"],
            "action_group": config["env"]["action_group"],
            "forward_reward_weight": config["env"]["forward_reward_weight"],
            "ctrl_cost_weight": config["env"]["ctrl_cost_weight"],
            "contact_cost_weight": config["env"]["contact_cost_weight"],
            "smooth_positions": config["env"]["smooth_positions"],
            "frame_skip": config["env"]["frame_skip"],
            "yvel_cost_weight": config["env"]["yvel_cost_weight"]
        },
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": config["env"]["max_episode_steps"]},
    )

    # Keep track of episode statistics
    env = VecMonitor(env)

    # Load model if resuming from checkpoint
    model = None
    if resume:
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest:
            print(f"[Resume] Loading latest checkpoint: {latest}")
            model = PPO.load(latest, env=env, device='cpu')
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
            tensorboard_log=config["train"]["log_dir"],
            device='cpu'
        )

    checkpoint_cb = CheckpointCallback(
        # Divide by number of envs because SB3 counts steps across all envs
        save_freq=checkpoint_freq // config["env"]["n_envs"],
        save_path=checkpoint_dir,
        name_prefix="model"
    )
    # Train the Model for a few timesteps
    model.learn(
        total_timesteps=config["train"]["total_timesteps"],
        tb_log_name=config["train"]["run_name"],
        callback=checkpoint_cb,
    )

    # Save the trained model

    model.save(os.path.join(checkpoint_dir, 
                            f"{config['train']['run_name']}_final"))

if __name__ == "__main__":
    main()
