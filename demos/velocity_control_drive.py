import mujoco
import mujoco.viewer
import numpy as np
import time
from torsional.utils import load_model_from_arg

def main(model_path=None):
    """
    Loads a MuJoCo model and applies position control to specified actuators."""
    if model_path is None:
        model = load_model_from_arg()
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))

    data = mujoco.MjData(model)
    VELOCITY_ACTUATORS= ["spring1a_vel", "spring2a_vel", "spring3a_vel", "spring4a_vel", "spring2c_vel", "spring1c_vel", "spring4c_vel", "spring3c_vel"]
    velocity_actuator_ids = []

    for name in VELOCITY_ACTUATORS:
        velocity_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if velocity_actuator_id < 0:
            raise ValueError(f"Actuator '{name}' not found in the model.")
        velocity_actuator_ids.append(velocity_actuator_id)
    
    JOINT_NAME = "cylinder3a_hinge"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)

    if joint_id < 0:
        raise ValueError(f"Joint '{JOINT_NAME}' not found in the model.")
        
    print(f"Total control dimensions: {model.nu}")
    print(f"Monitoring joint: {JOINT_NAME} (ID: {joint_id})")
    print("-" * 50)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        duration = 10.0 # seconds
        total_duration = 2 * duration
        # Set all controls to zero initially. (Disabling all groups?)
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        initial_qpos = data.qpos[model.jnt_qposadr[joint_id]]
 
        pos_vel = 2.0
        neg_vel = -2.0
        
        start_time = time.time()

        while viewer.is_running() and (data.time < total_duration):
            sim_time = data.time
   
            # apply differential constant velocity to two groups of actuators
            for i,act_id in enumerate(velocity_actuator_ids):
                if i%2 == 0:
                    data.ctrl[act_id] = pos_vel
                else:
                    data.ctrl[act_id] = neg_vel 

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            if data.time % 0.1 < model.opt.timestep:
                print(f"Time: {data.time:.2f}s, Target: {pos_vel:.1f} rad, Actual: {data.qvel[model.jnt_dofadr[joint_id]]:.2f} rad")
            # Switch target position every second
        data.ctrl[:] = 0.0
        time.sleep(model.opt.timestep)
        start_time = time.time()


if __name__ == "__main__":
    main()