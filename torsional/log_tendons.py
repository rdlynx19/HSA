import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    xml_path = "/home/redhairedlynx/Documents/academics/hsa/torsional/torsional_tendon_cable.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # === Actuator → All Tendons per Cable ===
    actuator_to_tendons = {
        "spring1a_motor": ["inter1a1", "inter1a2", "inter1b1", "inter1b2"],
        "spring2a_motor": ["inter2a1", "inter2a2", "inter2b1", "inter2b2"],
        "spring3a_motor": ["inter3a1", "inter3a2", "inter3b1", "inter3b2"],
        "spring4a_motor": ["inter4a1", "inter4a2", "inter4b1", "inter4b2"]
    }

    # === ID mappings ===
    actuator_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in actuator_to_tendons
    }

    tendon_ids = {}
    tendon_rest_lengths = {}
    for tendons in actuator_to_tendons.values():
        for tendon in tendons:
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon)
            tendon_ids[tendon] = tid
            tendon_rest_lengths[tendon] = model.tendon_lengthspring[tid]

    # === Logging buffers ===
    times = []
    tendon_logs = {actuator: {t: [] for t in ts} for actuator, ts in actuator_to_tendons.items()}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            elapsed = time.time() - start_time
            times.append(elapsed)

            # Determine active actuators
            active = []
            for name, idx in actuator_ids.items():
                if abs(data.ctrl[idx]) > 1e-3:
                    active.append(name)

            # Log tendon lengths for each actuator (fill with value or NaN)
            for actuator in actuator_to_tendons:
                for tendon in actuator_to_tendons[actuator]:
                    tid = tendon_ids[tendon]
                    val = data.ten_length[tid] if actuator in active else np.nan
                    tendon_logs[actuator][tendon].append(val)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)

    # === Plotting ===
    times = np.array(times)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for i, (actuator, tendons) in enumerate(tendon_logs.items()):
        ax = axs[i]
        for tendon, vals in tendons.items():
            vals = np.array(vals)
            rest = tendon_rest_lengths[tendon][0]
            ax.plot(times, vals, label=f"{tendon}", linewidth=2)
            ax.axhline(rest, linestyle='--', color='gray', linewidth=1,
                       label=f"{tendon} rest")
        ax.set_title(f"{actuator} – Tendon Lengths")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Length (m)")
        ax.grid(True)

        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
