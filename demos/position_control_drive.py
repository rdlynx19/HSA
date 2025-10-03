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
    POSITION_ACTUATORS= ["spring3a_motor", "spring3c_motor", "spring4a_motor", "spring4c_motor"]
    position_actuator_ids = []

    for name in POSITION_ACTUATORS:
        position_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if position_actuator_id < 0:
            raise ValueError(f"Actuator '{name}' not found in the model.")
        position_actuator_ids.append(position_actuator_id)
    
    JOINT_NAME = "cylinder3a_hinge"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)

    if joint_id < 0:
        raise ValueError(f"Joint '{JOINT_NAME}' not found in the model.")
        
    print(f"Total control dimensions: {model.nu}")
    print(f"Monitoring joint: {JOINT_NAME} (ID: {joint_id})")
    print("-" * 50)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        duration = 0.5 # seconds
        total_duration = 2 * duration
        # Set all controls to zero initially. (Disabling all groups?)
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        initial_qpos = data.qpos[model.jnt_qposadr[joint_id]]
 
        target_positions = [0.5, 2.5]
        
        start_time = time.time()

        while viewer.is_running() and (data.time < total_duration):
            sim_time = data.time
   
            # Determine current segment and target position
            if sim_time < duration:
                start_pos = initial_qpos
                end_pos = target_positions[0]
                current_segment_time = sim_time
            # toggle target position every 'duration' seconds
            else:
                start_pos = target_positions[0]
                end_pos = target_positions[1]
                current_segment_time = sim_time - duration

            # Linear interpolation for target position
            interp_factor = np.clip(current_segment_time / duration, 0.0, 1.0)
            target_position = start_pos + (end_pos - start_pos) * interp_factor


            for act_id in position_actuator_ids:
                data.ctrl[act_id] = target_position
          

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.tismetep)

            if data.time % 0.1 < model.opt.timestep:
                print(f"Time: {data.time:.2f}s, Target: {target_position:.1f} rad, Actual: {data.qpos[model.jnt_qposadr[joint_id]]:.2f} rad")
            # Switch target position every second
        data.ctrl[:] = 0.0
        time.sleep(model.opt.timestep)
        start_time = time.time()


if __name__ == "__main__":
    main()