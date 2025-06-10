# torsional/utils.py
import os
import sys
import mujoco

def load_model_from_arg():
    if len(sys.argv) != 2:
        print("Usage: python3 script.py <relative_model_path>")
        print("Example: python3 staggered_wave.py 8Actuators/8_actuator.xml")
        sys.exit(1)

    model_rel_path = sys.argv[1]

    # Find the root of the torsional folder (two levels above this file)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    torsional_root = utils_dir  # already in torsional

    # Path to models directory
    models_dir = os.path.join(torsional_root, "models")
    model_path = os.path.join(models_dir, model_rel_path)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    return mujoco.MjModel.from_xml_path(model_path)
