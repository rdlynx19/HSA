import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("../../models/8Actuators/8_actuator.xml")
data = mujoco.MjData(model)

# Actuator control direction map
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

# Actuator ID lookup
actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in sign_map.keys()}

# Groups to actuate in order
sequence = [
    ["spring1a_motor", "spring3a_motor"],
    ["spring1c_motor", "spring3c_motor"],
    ["spring2a_motor", "spring4a_motor"],
    ["spring2c_motor", "spring4c_motor"]
]

# Parameters
dt = model.opt.timestep
steps_per_ramp = int(2.0 / dt)
steps_per_pause = int(1.0 / dt)
amplitude = 3.14

def ramp(group, direction='up'):
    for step in range(steps_per_ramp):
        frac = step / steps_per_ramp
        scale = frac if direction == 'up' else (1 - frac)
        # data.ctrl[:] = 0
        for name in group:
            aid = actuator_ids[name]
            sign = sign_map[name]
            data.ctrl[aid] = sign * amplitude * scale
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

def pause():
    # data.ctrl[:] = 0
    for _ in range(steps_per_pause):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

# Run viewer with single actuation sequence
with mujoco.viewer.launch_passive(model, data) as viewer:
    for group in sequence:
        ramp(group, direction='up')
        pause()
        ramp(group, direction='down')
        pause()

print("Single actuation pattern completed.")
