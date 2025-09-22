import mujoco
import mujoco.viewer
import time
from torsional.utils import load_model_from_arg

def main(model_path=None):
    # === Load model ===
    if model_path is None:
        model = load_model_from_arg()
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))

    data = mujoco.MjData(model)

    # === Viewer loop ===
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        duration = 2.0  # Time per phase in seconds

        while viewer.is_running():
            elapsed = time.time() - start_time
            phase = int(elapsed // duration) % 2

            # Here you could add actuation logic based on phase if needed

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print("Simulation ended.")

if __name__ == "__main__":
    main()
