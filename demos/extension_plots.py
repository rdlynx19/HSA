from torsional.controlAPI import MuJoCoControlInterface
import matplotlib.pyplot as plt

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
        sim.position_control_crawl(position=2.14, lock=False, plot=False)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        times, dists = zip(*sim.distances)
        plt.plot(times, dists)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance between blocks (m)")
        plt.show()
        sim.close_simulation()


if __name__ == "__main__":
    main()