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
    POSITION_ACTUATORS= ["spring1a_motor", "spring2c_motor", "spring3a_motor", "spring4c_motor"]
    position_actuator_ids = []

    # To enable an actuator group, we can use
    # model.opt.disableactuator &= ~(1 << group_index)

    # To disable an actuator group, we can use
    # model.opt.disableactuator |= (1 << group_index)

    # Checking if the group is disabled
    # ((model.opt.disableactuator >> group_index) & 1) == 1
    
    model.opt.disableactuator &= ~(1 << 1)

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
        duration = 5 # seconds
        total_duration = 2 * duration
        # Set all controls to zero initially. (Disabling all groups?)
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        viewer.sync()
        initial_qpos = data.qpos[model.jnt_qposadr[joint_id]]
 
        pos_target_positions = [0.5, 2.5]
        neg_target_positions = [-0.5, -2.5]
        
        start_time = time.time()

        while viewer.is_running() and (data.time < total_duration):
            sim_time = data.time
   
            # Determine current segment and target position
            if sim_time < duration:
                start_pos = initial_qpos
                end_pos = pos_target_positions[0]
                start_pos_n = initial_qpos
                end_pos_n = neg_target_positions[0]
                current_segment_time = sim_time
            # toggle target position every 'duration' seconds
            else:
                start_pos = pos_target_positions[0]
                end_pos = pos_target_positions[1]
                start_pos_n = neg_target_positions[0]
                end_pos_n = neg_target_positions[1]
                current_segment_time = sim_time - duration

            # Linear interpolation for target position
            interp_factor = np.clip(current_segment_time / duration, 0.0, 1.0)
            pos_target_position = start_pos + (end_pos - start_pos) * interp_factor
            neg_target_position = start_pos_n + (end_pos_n - start_pos_n) * interp_factor


            for i, act_id in enumerate(position_actuator_ids):
                # Apply target based on group index
                if i % 2 == 0:
                    data.ctrl[act_id] = pos_target_position
                else:
                    data.ctrl[act_id] = neg_target_position
          

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            if data.time % 0.1 < model.opt.timestep:
                print(f"Time: {data.time:.2f}s, Target: {pos_target_position:.1f} rad, Actual: {data.qpos[model.jnt_qposadr[joint_id]]:.2f} rad")
            # Switch target position every second
        data.ctrl[:] = 0.0
        time.sleep(model.opt.timestep)
        start_time = time.time()


if __name__ == "__main__":
    main()