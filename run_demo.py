#!/usr/bin/env python3
import argparse, sys, importlib
from pathlib import Path

# Mapping demo names to their corresponding paths
DEMO_MAP = {
    "bending": "torsional.demo_scripts.bending.bending_sequence",
    "staggered_wave": "torsional.demo_scripts.extend_contract.staggered_wave",
    "staggered_wave_repeat": "torsional.demo_scripts.extend_contract.staggered_waveRepeat",
    "inchworm": "torsional.demo_scripts.inchworm.inchworm_dynamic",
    "rolling_wave" : "torsional.demo_scripts.rolling.rolling_wave",
    "log_com": "torsional.demo_scripts.load_scripts.log_com",
    "log_single_actuator": "torsional.demo_scripts.load_scripts.log_single_actuator",
    "load_model": "torsional.demo_scripts.load_scripts.load_model",
    "test_friction": "torsional.demo_scripts.load_scripts.test_friction",
    "position_control": "torsional.demo_scripts.control_interface.position_control",
}

# Map model names to their actual XML paths
MODEL_MAP = {
    "single_actuator": "torsional/models/singleActuator/single_actuator.xml",
    "eight_actuators": "torsional/models/8Actuators/8_actuator.xml",
    "friction": "torsional/models/8Actuators/anisotropic8_actuator.xml",
    "rolling_8": "torsional/models/8Actuators/rolling8actuator.xml",
    "no_collision_10": "torsional/models/noCollision/no_collision10.xml",
    "no_collision_14": "torsional/models/noCollision/no_collision14.xml",
    "no_collision_17": "torsional/models/noCollision/no_collision17.xml",
    "no_collision_7": "torsional/models/noCollision/no_collision7.xml",
    "end_contact_17": "torsional/models/pointOfContact/17mm.xml",
    "actuator_groups": "torsional/models/testing/actuator_groups.xml",
}

def main():
    parser = argparse.ArgumentParser(
        description="Run torsional soft actuator demo_scripts and scripts.\n\n"
                    "Examples:\n"
                    "  python run_demo.py bending singleActuator\n"
                    "  python run_demo.py rolling_wave 8Actuators",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "demo",
        choices=DEMO_MAP.keys(),
        metavar="DEMO",
        help="Demo to run\n - " + "\n - ".join(DEMO_MAP.keys()),
    )

    parser.add_argument(
        "model",
        choices=MODEL_MAP.keys(),
        metavar="\nMODEL",
        help="Model to use\n - " + "\n - ".join(MODEL_MAP.keys())
    )
    args = parser.parse_args()
    model_path = Path(MODEL_MAP[args.model])
    module_name = DEMO_MAP[args.demo]
    try:
        demo_module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"could not import demo '{args.demo}' ({module_name}): {e}")
        sys.exit(1)

    # If the demo script has 'main()' function, call it
    if hasattr(demo_module, "main"):
        demo_module.main(str(model_path))
    else:
        print(f"Demo '{args.demo}' does not define a main(model_path) function.")

if __name__ == "__main__":
        main()