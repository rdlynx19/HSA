"""
Demo Script: MuJoCo Model Loading and Viewer Initialization.

This script provides a minimal example of how to initialize the `MuJoCoControlInterface`, 
load a specified MuJoCo XML model, configure actuator groups, and launch the passive 
viewer using the `view_model` method. It is intended for quick verification of 
model loading and simulation initialization.
"""
from torsional.controlAPI import MuJoCoControlInterface, RobotState

def load_model():
    """
    Initializes the MuJoCo simulation interface with the specified XML model, 
    enables the position control actuator group (Group 1), and launches the 
    passive viewer.

    The simulation continues running until the viewer window is manually closed 
    or interrupted by the user.

    :returns: None
    :rtype: None
    """
    model_path = "torsional/models/closer_model.xml"
    sim = MuJoCoControlInterface(model_path=model_path)
    
    # Enable actuator group 1 (position control) and disable group 2 (velocity control)
    sim.enable_actuator_group(1)
    sim.disable_actuator_group(2)

    try:
        # Runs simulation steps and syncs viewer until viewer is closed.
        sim.view_model()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        sim.close_simulation()

if __name__ == "__main__":
    load_model()