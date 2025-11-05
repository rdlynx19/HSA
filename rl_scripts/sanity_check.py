import time
import numpy as np
from hsa_gym.envs.hsa_v1 import HSAEnv

def run_sanity_test():
    env = HSAEnv(
        render_mode="human",
        randomize_goal=False, 
        use_locks=True)

    print("\n==== SANITY CHECK TEST START ====\n")

    # --- Phase 1: Zero Action Test ---
    print("Phase 1: Zero Action (20 steps)")
    obs, info = env.reset(seed=0)
    zero_action = {
        "motors": np.zeros(env.action_space["motors"].shape, dtype=np.float32),
        "locks": np.ones(env.action_space["locks"].shape, dtype=np.int8), 
    }
    for i in range(200):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        print(f"[Zero] step={i}, reward={reward:.3f}, dist={info.get('cur_distance', None):.3f}")
        if terminated or truncated:
            print("Terminated early during zero-action phase.")
            break
        time.sleep(0.05)

    # --- Phase 2: Ramp Action Test ---
    print("\nPhase 2: Ramp Action (30 steps)")
    obs, info = env.reset(seed=1)
    for i in range(300):
        # gradually increase action from low to high
        frac = i / 299
        low = env.action_space["motors"].low
        high = env.action_space["motors"].high
        motors = low + frac * (high - low)

        action = {"motors": motors, "locks": np.ones_like(env.action_space["locks"].sample())}
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"[Ramp] step={i}, reward={reward:.3f}, dist={info.get('cur_distance', None):.3f}")
        if terminated or truncated:
            print("Terminated early during ramp-action phase.")
            break
        time.sleep(0.05)

    # --- Phase 3: Sine Wave Action Test ---
    print("\nPhase 3: Sine Wave Action (40 steps)")
    obs, info = env.reset(seed=2)
    t = 0.0
    for i in range(400):
        t += 0.1
        motors = np.sin(t) * env.action_space["motors"].high * 0.5  # 50% amplitude
        action = {"motors": motors.astype(np.float32), "locks": np.ones_like(env.action_space["locks"].sample())}

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[Sine] step={i}, reward={reward:.3f}, dist={info.get('cur_distance', None):.3f}")
        if terminated or truncated:
            print("Terminated early during sine-wave phase.")
            break
        time.sleep(0.05)

    env.close()
    print("\n==== SANITY CHECK COMPLETE ====\n")


if __name__ == "__main__":
    run_sanity_test()
