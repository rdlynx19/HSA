import numpy as np
from hsa_gym.envs.hsa_position import HSAEnv

env = HSAEnv(render_mode="human", frame_skip=20, pd_pos_control=False)
obs, info = env.reset()

print("Testing actuator mapping...")

# Test: Command first 4 actuators to +2.355, last 4 to -0.785
action = np.array([3.14, 3.14, 3.14, 3.14, -3.14, -3.14, -3.14, -3.14])

for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(action)

print(f"Commanded: {info['des_pos']}")
print(f"Actual:    {info['actual_pos']}")
print("\nMapping (actuator_idx → actual_pos_idx):")
for i in range(8):
    # Find which actual position matches this desired target
    desired_val = info['des_pos'][i]
    for j in range(8):
        if abs(info['actual_pos'][j] - desired_val) < 0.01:
            print(f"  Actuator {i} (desired={desired_val:.2f}) → actual_pos[{j}]")
            break

env.close()