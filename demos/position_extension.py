"""
Demo Script: Extension and Contraction Cycle.

This script executes the simplest complete motion sequence of the HSA robot: 
extending the body by actuating joints toward target positions, followed by 
contracting the body back to the neutral (zero) state. 
"""
from torsional.controlAPI import MuJoCoControlInterface
import time
def extend_contract():
    """
    Initializes the MuJoCo simulation, performs a single full extension-contraction 
    cycle, and ensures clean shutdown.

    The motion uses position control (Actuator Group 1) and temporarily disables 
    all disc equality constraints to observe the full range of joint movement.

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
        
        # Phase 1: Preparation (Disable all constraints)
        sim.modify_equality_constraints(disable=True, 
                                         constraints=["disc1b", "disc2b", "disc3b", "disc4b"])
        
        # Phase 2: Extension (Moves joints to 2.14 rad)
        sim.position_control_extension(position=2.14, duration=3.0)
        
        # Phase 3: Contraction (Moves joints back to 0.0 rad)
        sim.position_control_contraction(duration=3.0)
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Ensures the viewer and simulation data are cleaned up
        sim.close_simulation()
        

if __name__ == "__main__":
    extend_contract()