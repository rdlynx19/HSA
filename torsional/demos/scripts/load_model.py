import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from torsional.utils import load_model_from_arg

def main():
    # model = mujoco.MjModel.from_xml_path("/home/redhairedlynx/Documents/academics/hsa/torsional/models/8Actuators/8_actuator.xml")
    model = load_model_from_arg()
    data = mujoco.MjData(model)


    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        duration = 2.0  # Time per phase in seconds

        while viewer.is_running():
            elapsed = time.time() - start_time
            phase = int(elapsed // duration) % 2
            
            mujoco.mj_step(model, data)
            viewer.sync()
         
            time.sleep(0.001)


if __name__ == "__main__":
    main()
