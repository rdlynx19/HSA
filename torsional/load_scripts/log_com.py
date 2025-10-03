import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from torsional.utils import load_model_from_arg

def actuator_color(name):
    """Give a unique color to each actuator for plotting."""
    color_map = {
        "spring1a_motor": "red",
        "spring2a_motor": "blue",
        "spring3a_motor": "green",
        "spring4a_motor": "orange"
    }
    return color_map.get(name, "gray")


def main(model_path=None):
    # === Load model ===
    if model_path is None:
        model = load_model_from_arg()
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path))

    data = mujoco.MjData(model)

    # === Actuator control map ===
    actuator_names = ["spring1a_motor", "spring2a_motor", "spring3a_motor", "spring4a_motor"]
    actuator_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names}

    # === Block body IDs ===
    block_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_a")
    block_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_b")

    # === Data logging ===
    times = []
    com_distances = []
    actuator_log = []

    # === Viewer ===
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        print("Interact with the viewer. Start moving actuators to log data.")

        while viewer.is_running():
            elapsed = time.time() - start_time
            times.append(elapsed)

            # Compute COM distance
            com_a = data.xpos[block_a_id].copy()
            com_b = data.xpos[block_b_id].copy()
            dist = np.linalg.norm(com_a - com_b)
            com_distances.append(dist)

            # Log active actuators
            active = [name for name, idx in actuator_ids.items() if abs(data.ctrl[idx]) > 1e-3]
            actuator_log.append(active)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    # === Plot results ===
    times = np.array(times)
    com_distances = np.array(com_distances)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, com_distances, label="COM Distance", color='black')

    # Shade regions where actuators were active
    for actuator in actuator_ids.keys():
        is_active = [actuator in acts for acts in actuator_log]
        start = None
        for i, active_now in enumerate(is_active):
            if active_now and start is None:
                start = times[i]
            elif not active_now and start is not None:
                ax.axvspan(start, times[i], alpha=0.2, color=actuator_color(actuator), label=actuator)
                start = None
        if start is not None:
            ax.axvspan(start, times[-1], alpha=0.2, color=actuator_color(actuator), label=actuator)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance Between COMs of block_a and block_b Over Time")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

