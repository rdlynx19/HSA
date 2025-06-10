import mujoco
import mujoco.viewer
import numpy as np
import time
from torsional.utils import load_model_from_arg

# Load model
# model = mujoco.MjModel.from_xml_path("../../models/8Actuators/8_actuator.xml")
model = load_model_from_arg()
data = mujoco.MjData(model)

# Define actuator groups
actuator_groups = [
    ["spring1a_motor", "spring3a_motor"],
    ["spring1c_motor", "spring3c_motor"],
    ["spring2a_motor", "spring4a_motor"],
    ["spring2c_motor", "spring4c_motor"]
]

# Direction map
sign_map = {
    "spring1a_motor": 1,
    "spring3a_motor": 1,
    "spring2c_motor": 1,
    "spring4c_motor": 1,
    "spring1c_motor": 1,
    "spring3c_motor": 1,
    "spring2a_motor": 1,
    "spring4a_motor": 1,
}

# Actuator IDs
all_actuators = sum(actuator_groups, [])
actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in all_actuators}

# Parameters
dt = model.opt.timestep
steps_per_phase = int(2.0/ dt)         # Ramp duration
steps_per_pause = int(1.0 / dt)         # Pause duration
amplitude = 3.14
num_cycles = 3

def ramp_actuation(group, direction='up'):
    targets = {name: sign_map[name] * amplitude for name in group}
    for step in range(steps_per_phase):
        progress = step / steps_per_phase
        ramp = progress if direction == 'up' else (1 - progress)
        # data.ctrl[:] = 0
        for name in group:
            aid = actuator_ids[name]
            data.ctrl[aid] = ramp * targets[name]
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

def pause_simulation():
    # data.ctrl[:] = 0
    for _ in range(steps_per_pause):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

# Viewer loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    for cycle in range(num_cycles):
        for group in actuator_groups:
            ramp_actuation(group, direction='up')
            pause_simulation()
            ramp_actuation(group, direction='down')
            pause_simulation()

        for group in reversed(actuator_groups):
            ramp_actuation(group, direction='up')
            pause_simulation()
            ramp_actuation(group, direction='down')
            pause_simulation()

print("Smooth ramped wave pattern completed.")
