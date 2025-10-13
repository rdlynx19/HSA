from torsional.controlAPI import MuJoCoControlInterface

def main():
    """
    Example usage of the MuJoCoControlInterface to control a MuJoCo model.
    """
    model_path = "torsional/models/actuator_groups.xml"
    sim = MuJoCoControlInterface(model_path=model_path)
    
    # Enable actuator group 2 and disable group 1
    sim.enable_actuator_group(2)
    sim.disable_actuator_group(1)

    try: 
        sim.start_simulation()
        sim.velocity_control_drive(velocity=6.0) 
        # sim.close_simulation()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

if __name__ == "__main__":
    main()