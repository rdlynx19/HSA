"""
Demo Script: Differential Twisting Locomotion (Twist Type 2).

This script demonstrates a differential "twist" maneuver using position control.

NOTE: Replicating the stability of complex, friction-driven gaits 
like this twisting motion in MuJoCo is challenging to match 
the performance of the physical robot. 
"""
from torsional.controlAPI import MuJoCoControlInterface

def twist2():
    """
    Initializes the MuJoCo simulation, performs a differential twist maneuver (Twist Type 1), 
    and keeps the viewer open until interrupted.

    The demonstration uses position control (Actuator Group 1) combined with precise 
    modification of equality constraints to create the differential twisting force.

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
        
        # Phase 1: Twist Type 2
        # This motion relies on specific constraint unlocking/locking implemented inside 
        # position_control_twist2 to achieve lateral displacement.
        sim.position_control_twist2(position=2.8, duration=5.00)
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Ensures the viewer and simulation data are cleaned up
        sim.close_simulation()
        

if __name__ == "__main__":
    twist2()