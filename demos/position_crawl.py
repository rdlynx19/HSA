from torsional.controlAPI import MuJoCoControlInterface
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Example usage of extension using position control
    """
    model_path = "torsional/models/closer_model.xml"
    sim = MuJoCoControlInterface(model_path=model_path)

    # Enable actautor group 1 and disable group 2
    sim.disable_actuator_group(2)
    sim.enable_actuator_group(1)

    try:
        sim.position_control_crawl(position=2.14, lock=True, duration=0.5)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        plt.title("Trajectories of All Bodies")
        plt.grid(True)
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")

        # --- Simple plotting loop ---
        for body, data in sim.trajectory.items():
            times, poses = zip(*data)
            poses = np.array(poses)
            plt.plot(poses[:, 0], poses[:, 1], label=body)

        plt.legend()
        plt.axis("equal")
        plt.show()
        sim.close_simulation()

if __name__ == "__main__":
    main()