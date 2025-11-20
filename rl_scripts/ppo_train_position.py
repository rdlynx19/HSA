import yaml, os, glob, shutil, re

import gymnasium as gym

from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from hsa_gym.envs.hsa_constrained import HSAEnv

config_file = "./configs/ppo_position.yaml"
xml_file = "../hsa_gym/envs/assets/hsaModel.xml"

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
    
    xml_path = os.path.join(script_dir, xml_file)
    shutil.copy(xml_path, os.path.join(checkpoint_dir, "used_model.xml"))
    print(f"[Config] XML model file saved to {os.path.join(checkpoint_dir, 'used_model.xml')}")
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
            "yvel_cost_weight": config["env"]["yvel_cost_weight"],
            "constraint_cost_weight": config["env"]["constraint_cost_weight"],
            "acc_cost_weight": config["env"]["acc_cost_weight"],
            "max_increment": config["env"]["max_increment"]
        },
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": config["env"]["max_episode_steps"]},
    )

    # Keep track of episode statistics
    env = VecMonitor(env)

    # Load model if resuming from checkpoint
    model = None
    trained_steps = 0
    reset_timesteps = True # Default for fresh training

    if resume:
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest:
            print(f"[Resume] Loading latest checkpoint: {latest}")

            try:
                trained_steps = extract_step_number(latest)
                print(f"[Resume] Model has been trained for {trained_steps} steps.")
            except ValueError as e:
                print(f"[Resume] Warning: {e}. Assuming 0 trained steps.")
                resume = False

            if resume:
                print(f"[Resume] Loading model ...")
                model = PPO.load(latest, env=env, device='cpu')
                reset_timesteps = False # Keep the original timestep count
        else:
            print(f"[Resume] WARNING: No checkpoint found in {checkpoint_dir}, starting fresh training.")
            resume = False

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
    # Train the Model for a few timesteps
    model.learn(
        total_timesteps=timesteps_to_train,
        tb_log_name=config["train"]["run_name"],
        callback=checkpoint_cb,
        reset_num_timesteps=reset_timesteps
    )

    # Save the trained model
    final_steps = trained_steps + timesteps_to_train
    print(f"[Info] Training complete. Total trained steps: {final_steps:,}")
    model.save(os.path.join(checkpoint_dir, 
                            f"{config['train']['run_name']}_final_{final_steps}_steps"))

if __name__ == "__main__":
    main()
