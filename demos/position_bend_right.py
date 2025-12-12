"""
Demo Script: Bending Locomotion (Bend Right).

This script demonstrates a sequence involving position control to achieve a 
differential "bend right" motion, followed by a recovery contraction phase. 
"""
from torsional.controlAPI import MuJoCoControlInterface

def main():
    """
    Initializes the MuJoCo simulation, performs a differential bend right maneuver, 
    and returns the robot to the idle state via contraction.

    The demonstration uses position control (Actuator Group 1) for the movement.

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
        
        # Phase 1: Bend Right
        # The robot uses differential position control and constraint locking/unlocking 
        # to achieve a lateral bend.
        sim.position_control_bend_right(position=2.9, duration=5)
        
        # Phase 2: Contraction (Recovery)
        # Returns the robot to the IDLE/zero position.
        sim.position_control_contraction(duration=5)
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Ensures the viewer and simulation data are cleaned up
        sim.close_simulation()
        

if __name__ == "__main__":
    main()