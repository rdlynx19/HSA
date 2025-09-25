import mujoco
import mujoco.viewer
import time
import numpy as np
from torsional.utils import load_model_from_arg

def main(model_path=None):
    # Load model
    if model_path is None:
        model = load_model_from_arg()
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Store initial positions
    block_a_init = data.qpos[0:3].copy()
    block_b_init = data.qpos[7:10].copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        force_duration = 2.0
        force_magnitude = 10.0

        while viewer.is_running():
            elapsed = time.time() - start_time

            if elapsed < force_duration:
                # Apply equal forces to both blocks
                data.xfrc_applied[0] = [force_magnitude, 0, 0, 0, 0, 0]
                data.xfrc_applied[1] = [force_magnitude, 0, 0, 0, 0, 0]

                # Calculate and print displacements
                block_a_pos = data.qpos[0:3]
                block_b_pos = data.qpos[7:10]
                
                disp_a = np.linalg.norm(block_a_pos - block_a_init)
                disp_b = np.linalg.norm(block_b_pos - block_b_init)
                
                print(f"\rTime: {elapsed:.2f}s | Block A displacement: {disp_a:.3f}m | Block B displacement: {disp_b:.3f}m", end="")
            
            else:
                # Stop forces
                data.xfrc_applied[0] = [0, 0, 0, 0, 0, 0]
                data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print("\nSimulation ended.")

if __name__ == "__main__":
    main()