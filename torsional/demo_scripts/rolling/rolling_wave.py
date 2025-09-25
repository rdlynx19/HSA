import mujoco
import mujoco.viewer
import numpy as np
import time
from torsional.utils import load_model_from_arg

def main(model_path=None):
    # === Load model ===
    if model_path is None:
        model = load_model_from_arg()
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))

    data = mujoco.MjData(model)

    # === Actuator names and IDs ===
    actuator_names = ["spring3a_motor", "spring4a_motor"]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    for name in actuator_names]

    # === Timing parameters ===
    dt = model.opt.timestep
    ramp_duration = 0.5  # seconds
    pause_duration = 1.0  # seconds
    steps_ramp = int(ramp_duration / dt)
    steps_pause = int(pause_duration / dt)
    amplitude = 3.0  # torque value
    num_cycles = 3

    # === Ramping functions ===
    def ramp_up(a_ids):
        for step in range(steps_ramp):
            torque = amplitude * (step / steps_ramp)
            for aid in a_ids:
                data.ctrl[aid] = torque
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

    def ramp_down(a_ids):
        for step in range(steps_ramp):
            torque = amplitude * (1 - step / steps_ramp)
            for aid in a_ids:
                data.ctrl[aid] = torque
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

    def hold(a_ids, torque_val):
        for _ in range(steps_pause):
            for aid in a_ids:
                data.ctrl[aid] = torque_val
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

    # === Main control loop ===
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(num_cycles):
            ramp_up(actuator_ids)
            hold(actuator_ids, amplitude)
            ramp_down(actuator_ids)
            hold(actuator_ids, 0.0)

    print("Finished smooth actuation cycle with ramp-down.")


if __name__ == "__main__":
    main()

