"""
Demo Script 5: Continuous Crawling Locomotion and Trajectory Analysis.

This script executes the continuous crawling motion of the HSA robot using position 
control. T

Crucially, the script records the 2D (X-Y) world-space trajectory of the robot's 
tracked bodies throughout the simulation. The resulting path is plotted using Matplotlib 
to visualize the efficiency and stability of the movement. 
"""
from torsional.controlAPI import MuJoCoControlInterface
import matplotlib.pyplot as plt
import numpy as np

def crawl():
    """
    Initializes the MuJoCo simulation, runs the continuous crawling demo in the 
    unlocked mode, and plots the X-Y trajectories of all tracked bodies.

    :returns: None
    :rtype: None
    """
    model_path = "torsional/models/closer_model.xml"
    sim = MuJoCoControlInterface(model_path=model_path)

    # Enable actuator group 1 (position control) and disable group 2 (velocity control)
    sim.disable_actuator_group(2)
    sim.enable_actuator_group(1)

    try:
        sim.start_simulation()
        # Run continuous crawling simulation in unlocked mode.
        # This records the trajectory internally.
        sim.position_control_crawl(position=2.14, lock=False, duration=0.5)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # --- Plotting Trajectories ---
        plt.title("Trajectories of All Bodies (Unlocked Crawl)")
        plt.grid(True)
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")

        # Plot the trajectory of each tracked body
        for body, data in sim.trajectory.items():
            if data:
                # Unzip (time, pos) tuples into separate lists
                times, poses = zip(*data)
                poses = np.array(poses)
                # Plot X vs Y positions
                plt.plot(poses[:, 0], poses[:, 1], label=body)

        plt.legend()
        plt.axis("equal")
        plt.show()
        sim.close_simulation()

if __name__ == "__main__":
    crawl()