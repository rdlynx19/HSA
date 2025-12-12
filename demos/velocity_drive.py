"""
Demo Script: Velocity Control Driving (PID-Driven Locomotion).

This script demonstrates the use of the `velocity_control_drive` method, which 
attempts to use PID controllers to maintain a constant joint velocity for continuous 
locomotion. This mode is generally intended for high-speed, sustained movement.

NOTE: The velocity control feature is currently noted as not fully functional or 
stable, and may exhibit erratic behavior or fail to achieve sustained motion in 
the simulation. 
"""
from torsional.controlAPI import MuJoCoControlInterface

def main():
    """
    Initializes the MuJoCo simulation, attempts to execute a velocity-controlled 
    driving maneuver, and keeps the viewer open until interrupted.

    The demonstration enables Actuator Group 2 (velocity control) and disables 
    Actuator Group 1 (position control). It also disables all equality constraints 
    prior to starting the velocity drive.

    :returns: None
    :rtype: None
    """
    model_path = "torsional/models/closer_model.xml"
    sim = MuJoCoControlInterface(model_path=model_path)
    
    # Enable actuator group 2 (velocity control) and disable group 1 (position control)
    sim.enable_actuator_group(2)
    sim.disable_actuator_group(1)

    try: 
        sim.start_simulation()
        
        # Phase 1: Preparation (Disable all equality constraints)
        sim.modify_equality_constraints(disable=True, 
                                         constraints=["disc1b", "disc2b", "disc3b", "disc4b"])
        
        # Phase 2: Velocity Control Drive (PID attempts to maintain velocity=6.0 rad/s)
        sim.velocity_control_drive(velocity=6.0) 
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # Ensures the viewer and simulation data are cleaned up
        sim.close_simulation()

if __name__ == "__main__":
    main()