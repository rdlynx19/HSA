from torsional.controlAPI import MuJoCoControlInterface

def main():
    """
    Example usage of twist using position control
    """
    model_path = "torsional/models/closer_model.xml"
    sim = MuJoCoControlInterface(model_path=model_path)

    # Enable actautor group 1 and disable group 2
    sim.disable_actuator_group(2)
    sim.enable_actuator_group(1)

    try:
        sim.start_simulation()
        sim.position_control_twist2(position=2.8, duration=5.00)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sim.close_simulation()
        

if __name__ == "__main__":
    main()