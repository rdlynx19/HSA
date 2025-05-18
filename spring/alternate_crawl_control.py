import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    model = mujoco.MjModel.from_xml_path("/home/redhairedlynx/Documents/academics/hsa/spring/spring_tendon_variant.xml")
    data = mujoco.MjData(model)

    actuator_names = [
        "spring1a_motor",
        "spring2a_motor",
        "spring3a_motor",
        "spring4a_motor", 
    ]
    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in actuator_names
    ]

    for name, idx in zip(actuator_names, actuator_ids):
        if idx == -1:
            raise RuntimeError(f"Actuator '{name}' not found.")
    
    for i in range(model.njnt):
        jnt_name = model.joint(i).name
        dof_start = model.jnt_dofadr[i]  # Starting DOF index for this joint
      
        print(f"Joint '{jnt_name}'  uses DOFs {dof_start}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        duration = 2.0  # Time per phase in seconds

        while viewer.is_running():
            elapsed = time.time() - start_time
            phase = int(elapsed // duration) % 2

            # Reset all actuators
            # for i in range(4):
            #     data.ctrl[actuator_ids[i]] = 0.0

            # if phase == 0:
            #     # Activate spring 1 and 3 (diagonal control from block A)
            #     data.ctrl[actuator_ids[0]] = 10.0  # spring1a
            #     data.ctrl[actuator_ids[2]] = 10.0  # spring3a
            # else:
            #     # Activate spring 2 and 4 (diagonal control from block B)
            #     data.ctrl[actuator_ids[0]] = 10.0  # spring2a
            #     data.ctrl[actuator_ids[2]] = 10.0  # spring4a
            
            mujoco.mj_step(model, data)
            viewer.sync()
         
            time.sleep(0.001)


if __name__ == "__main__":
    main()
