import numpy as np
from hsa_gym.envs.hsa_position import HSAEnv

env = HSAEnv(render_mode="human",frame_skip=20, pd_pos_control=False)  # or True if you want position mode
obs, info = env.reset()
print(obs)

n_act = 8
baseline_pos = info['actual_pos'].copy()
mapping = {}

for i in range(n_act):
    action = np.zeros(n_act)
    action[i] = 1.57  # give a big unique command only to actuator i

    # step a bit to let dynamics respond
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
    actual = info['actual_pos']

    # find index whose change is largest
    deltas = np.abs(actual - baseline_pos)
    j = int(np.argmax(deltas))
    mapping[i] = (j, deltas[j])
    # reset baseline for next isolation test (reset env so history doesn't accumulate)
    obs, info = env.reset()
    baseline_pos = info['actual_pos'].copy()

env.close()

print("Single-actuator isolation mapping (actuator -> (joint index, delta)): ")
for a, (j, d) in mapping.items():
    print(f"  Actuator {a} -> joint {j} (delta {d:.4f})")
