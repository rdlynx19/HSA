"""
Demo Script: Crawling Trajectory Plot (Distance vs. Time).

This script demonstrates the continuous crawling locomotion behavior of the HSA robot 
using position control. It specifically tracks and plots the trajectory of the robot during this motion. 
"""
from torsional.controlAPI import MuJoCoControlInterface
import matplotlib.pyplot as plt

def main():
    """
    Initializes the MuJoCo simulation, runs the continuous crawling demo in locked 
    position control mode, and plots the resulting distance between the robot's 
    two main blocks over time.

    The crawl is executed in "locked" mode (`lock=True`), meaning the primary equality 
    constraints are maintained during the motion cycle. The simulation runs until 
    interrupted by the user.

    :returns: None
    :rtype: None
    """
    model_path = "torsional/models/closer_model.xml"
    sim = MuJoCoControlInterface(model_path=model_path)

    # Enable actuator group 1 (position control) and disable group 2 (velocity control)
    sim.disable_actuator_group(2)
    sim.enable_actuator_group(1)

    try:
        # Run continuous crawling simulation in locked mode.
        # This calls extension and contraction repeatedly until Ctrl+C.
        sim.position_control_crawl(position=2.14, lock=True, plot=True)
        
        # The commented code below shows alternative demos (bending/twisting) that are disabled in this demo.
        # sim.position_control_bend_left(position=2.8, duration=2.5, plot=True)
        # sim.position_control_contraction(duration=2.5, plot=True)
        # sim.position_control_bend_right(position=2.8, duration=2.5, plot=True)
        # sim.position_control_contraction(duration=2.5, plot=True)
        # sim.position_control_twist1(position=2.84, duration=2.5, plot=True)
        # sim.position_control_contraction(duration=2.5, plot=True)
        # sim.position_control_twist2(position=2.84, duration=2.5, plot=True)
        # sim.position_control_contraction(duration=2.5, plot=True)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # --- Plotting Distances ---
        if sim.distances:
            times, dists = zip(*sim.distances)
            plt.title("Crawling Mode: Block Distance vs. Time")
            plt.grid(True)
            plt.plot(times, dists)
            plt.xlabel("Time (s)")
            plt.ylabel("Distance between blocks (m)")
            plt.show()
        else:
            print("No distance data recorded.")
        sim.close_simulation()


if __name__ == "__main__":
    main()