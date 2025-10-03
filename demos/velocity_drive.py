from torsional.controlAPI import MuJoCoControlInterface, RobotState

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
        sim.velocity_control_drive(velocity=6.0)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        sim.close_simulation()

if __name__ == "__main__":
    main()