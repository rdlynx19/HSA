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
        sim.start_simulation()
        sim.modify_equality_constraints(disable=True, 
                                         constraints=["disc1b", "disc2b", "disc3b", "disc4b"])
        sim.position_control_extension(position=2.00, duration=5.0)
        sim.position_control_contraction(duration=5.0)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sim.close_simulation()
        

if __name__ == "__main__":
    main()