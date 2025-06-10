import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from torsional.utils import load_model_from_arg

def main():
    model = load_model_from_arg()
    data = mujoco.MjData(model)

    # === Actuator info ===
    actuator_name = "spring1a_motor"
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    # === Body IDs ===
    spring1a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spring_1a")
    spring1c_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "spring_1c")

    # === Data logging ===
    times = []
    distances = []
    actuator_log = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Use the GUI to actuate the model. Close the viewer to generate the plot.")

        start_time = time.time()

        while viewer.is_running():
            elapsed = time.time() - start_time
            times.append(elapsed)

            # Log actuator usage
            actuator_log.append(abs(data.ctrl[actuator_id]) > 1e-3)

            # Measure distance between COMs of spring_1a and spring_1c
            pos_a = data.xpos[spring1a_id]
            pos_c = data.xpos[spring1c_id]
            dist = np.linalg.norm(pos_c - pos_a)
            distances.append(dist)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    # === Plotting ===
    times = np.array(times)
    distances = np.array(distances)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, distances, label="Distance between spring_1a and spring_1c", color="black")

    # Highlight when actuator was active
    start = None
    for i, active in enumerate(actuator_log):
        if active and start is None:
            start = times[i]
        elif not active and start is not None:
            ax.axvspan(start, times[i], color="red", alpha=0.2, label="Actuator Active")
            start = None
    if start is not None:
        ax.axvspan(start, times[-1], color="red", alpha=0.2, label="Actuator Active")

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance Between spring_1a and spring_1c Over Time")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
