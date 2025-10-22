from torsional.controlAPI import MuJoCoControlInterface
import matplotlib.pyplot as plt

def main():
    """
    Example usage of extension using position control
    """
    model_path = "torsional/models/vertical_orientation.xml"
    sim = MuJoCoControlInterface(model_path=model_path)

    # Enable actautor group 1 and disable group 2
    sim.disable_actuator_group(2)
    sim.enable_actuator_group(1)

    try:
        # sim.position_control_crawl(position=2.8, lock=False, plot=True)
        # sim.position_control_bend_left(position=2.8, duration=2.5, plot=True)
        # sim.position_control_contraction(duration=2.5, plot=True)
        # sim.position_control_bend_right(position=2.8, duration=2.5, plot=True)
        # sim.position_control_contraction(duration=2.5, plot=True)
        sim.position_control_twist1(position=2.8, duration=2.5, plot=True)
        sim.position_control_contraction(duration=2.5, plot=True)
        sim.position_control_twist2(position=2.8, duration=2.5, plot=True)
        sim.position_control_contraction(duration=2.5, plot=True)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        times, dists = zip(*sim.distances)
        plt.title("Vertical Twisting Mode: Locked")
        plt.grid(True)
        plt.plot(times, dists)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance between blocks (m)")
        plt.show()
        sim.close_simulation()


if __name__ == "__main__":
    main()