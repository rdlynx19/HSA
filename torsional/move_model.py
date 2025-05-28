import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

def run_servo_with_bi_directional_motion():
    model = mujoco.MjModel.from_xml_path(
        "/home/redhairedlynx/Documents/academics/hsa/torsional/torsional_tendon_cable.xml"
    )
    data = mujoco.MjData(model)

    # IDs
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "spring1a_motor")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spring1a_hinge")
    joint_qpos_adr = model.jnt_qposadr[joint_id]

    tendon_names = ["inter1a1", "inter1a2", "inter1b1", "inter1b2"]
    tendon_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name) for name in tendon_names]
    rest_lengths = [model.tendon_lengthspring[tid] for tid in tendon_ids]

    # Logs
    time_log = []
    torque_log = []
    angle_log = []
    spring_extension_log = []

    # Simulation parameters
    n_steps_per_phase = 4000
    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Phase 1: Move to +45 degrees
        data.ctrl[actuator_id] = -np.pi / 4
        for i in range(n_steps_per_phase):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

            time_log.append(i * dt)
            torque_log.append(data.actuator_force[actuator_id])
            angle_log.append(data.qpos[joint_qpos_adr])
            total_extension = float(np.sum([
                data.ten_length[tid] - rest_lengths[j] for j, tid in enumerate(tendon_ids)
            ]))
            spring_extension_log.append(total_extension)

        # Phase 2: Move to 0 degrees
        data.ctrl[actuator_id] = 0
        for i in range(n_steps_per_phase):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

            t = (n_steps_per_phase + i) * dt
            time_log.append(t)
            torque_log.append(data.actuator_force[actuator_id])
            angle_log.append(data.qpos[joint_qpos_adr])
            total_extension = float(np.sum([
                data.ten_length[tid] - rest_lengths[j] for j, tid in enumerate(tendon_ids)
            ]))
            spring_extension_log.append(total_extension)

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(time_log, torque_log)
    plt.xlabel("Time (s)")
    plt.ylabel("Servo Torque (Nm)")
    plt.title("Torque Over Time")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(time_log, np.degrees(angle_log))
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (degrees)")
    plt.title("Servo Position")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(time_log, np.array(spring_extension_log) * 1000)
    plt.xlabel("Time (s)")
    plt.ylabel("Total Spring Extension (mm)")
    plt.title("Spring Displacement Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_servo_with_bi_directional_motion()
