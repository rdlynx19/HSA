import numpy as np
from hsa_gym.envs.hsa_position import HSAEnv

env = HSAEnv(
    actuator_groups=[1],
    frame_skip=4,
    render_mode="human",
)

obs, info = env.reset()

# Command robot to neutral position (action=0 = default positions)
for step in range(5000):
    action = np.ones(8) * 0.0 # Neutral position command
 
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 50 == 0:
        print(f"Step {step}:")
        print(f"  Torques: {info.get('applied_torque', 'N/A')}")
        print(f"  Joint pos: {info['actual_position']}")  # First 4 joints
        print(f" Desired pos: {info['desired_position']}")  # First 4 joints
        print(f"  Block A height: {env.get_body_com('block_a')[2]:.3f}")
        
    if terminated:
        print(f"TERMINATED at step {step}")
        break

env.close()