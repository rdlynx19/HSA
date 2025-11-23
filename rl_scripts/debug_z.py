import numpy as np
from hsa_gym.envs.hsa_constrained import HSAEnv
from gymnasium.wrappers import TimeLimit

env = HSAEnv(xml_file="hsaTerrainModel.xml", render_mode="human", actuator_group=[1], enable_terrain=True, terrain_type="flat")
env = TimeLimit(env, max_episode_steps=500)

obs, info = env.reset()

# Access the unwrapped environment
data = env.unwrapped.data
model = env.unwrapped.model

print("="*60)
print("INITIAL STATE")
print("="*60)
print(f"Block A Z: {data.qpos[2]:.3f}")
print(f"Block B Z: {data.qpos[15]:.3f}")
print(f"Contacts: {data.ncon}")
print(f"Reward: {info.get('reward_total_costs', 'N/A')}")

print("\n" + "="*60)
print("STABILITY TEST - Holding with zero action for 200 steps")
print("="*60)

# Just hold still with zero action
stable_steps = 0
for i in range(200000):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 20 == 0:  # Print every 20 steps
        print(f"\nStep {i}:")
        print(f"  Z-heights: A={data.qpos[2]:.3f}, B={data.qpos[15]:.3f}")
        print(f"  Z-velocities: A={data.qvel[2]:.3f}, B={data.qvel[14]:.3f}")
        print(f"  Contacts: {data.ncon}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
    
    if not terminated and not truncated:
        stable_steps += 1
    
    if terminated or truncated:
        print(f"\n⚠️  TERMINATED at step {i+1}")
        print(f"    Reason: {'Terminated' if terminated else 'Truncated'}")
        print(f"    Final Z-heights: A={data.qpos[2]:.3f}, B={data.qpos[15]:.3f}")
        break

if stable_steps == 20000:
    print("\n" + "="*60)
    print("✅ ROBOT IS STABLE!")
    print("="*60)
    print(f"Robot remained stable for all {stable_steps} steps")
else:
    print("\n" + "="*60)
    print("❌ ROBOT IS UNSTABLE")
    print("="*60)
    print(f"Robot was only stable for {stable_steps}/200 steps")

env.close()
