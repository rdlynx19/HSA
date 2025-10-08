from torsional.controlAPI import MuJoCoControlInterface

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
        sim.position_control_crawl(position=1.57)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        sim.close_simulation()

if __name__ == "__main__":
    main()