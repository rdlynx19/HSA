import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("../../models/8Actuators/anisotropic8_actuator.xml")
data = mujoco.MjData(model)

# Replace with actual geom names for the two blocks
block_a_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_a")  # likely name
block_b_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_b")  # likely name

# Define actuator groups for extension and contraction
actuators_extend = ["spring1a_motor", "spring3a_motor"]
actuators_retract = ["spring1c_motor", "spring3c_motor"]
actuator_ids_extend = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuators_extend]
actuator_ids_retract = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuators_retract]

# Timing and control settings
dt = model.opt.timestep
ramp_steps = int(0.3 / dt)
amplitude = 3.0

def set_friction(geom_id, friction_type):
    if friction_type == "anisotropic":
        model.geom_friction[geom_id] = np.array([0.01, 1.0, 0.01])
    else:
        model.geom_friction[geom_id] = np.array([0.01, 0.01, 0.01])

def ramp_ctrl(act_ids):
    for step in range(ramp_steps):
        torque = amplitude * (step / ramp_steps)
        for aid in act_ids:
            data.ctrl[aid] = torque
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

def ramp_down_ctrl(act_ids):
    for step in range(ramp_steps):
        torque = amplitude * (1 - step / ramp_steps)
        for aid in act_ids:
            data.ctrl[aid] = torque
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(5):
        # Step 1: Fix block A, let block B move forward
        set_friction(block_a_geom_id, "anisotropic")
        set_friction(block_b_geom_id, "low")
        ramp_ctrl(actuator_ids_extend)
        ramp_down_ctrl(actuator_ids_extend)

        # Step 2: Fix block B, retract A
        set_friction(block_a_geom_id, "low")
        set_friction(block_b_geom_id, "anisotropic")
        ramp_ctrl(actuator_ids_retract)
        ramp_down_ctrl(actuator_ids_retract)
