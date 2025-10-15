from torsional.controlAPI import MuJoCoControlInterface

def main():
    """
    Example usage of twist using position control
    """
    model_path = "torsional/models/vertical_orientation.xml"
    sim = MuJoCoControlInterface(model_path=model_path)

    # Enable actautor group 1 and disable group 2
    sim.disable_actuator_group(2)
    sim.enable_actuator_group(1)

    try:
        sim.start_simulation()
        sim.position_control_bend_right(position=2.9, duration=5)
        sim.position_control_contraction(duration=5)
        # sim.position_control_crawl(position=1.57)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sim.close_simulation()
        

if __name__ == "__main__":
    main()