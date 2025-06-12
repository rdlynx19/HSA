# HSA Spring Actuation Locomotion in MuJoCo

This repository explores locomotion using torsional spring actuation. It provides MuJoCo simulation models and Python scripts to prototype and evaluate various gait strategies.

## Folder Structure
```
hsa/
├── torsional/ # Torsional spring locomotion models and demos
│ ├── models/ # MuJoCo XML models
│ ├── scripts/ # Helpers for model loading, logging, etc.
│ └── demos/ # Predefined gait sequences and simulations
│
└── terrain/ # (WIP) Terrain interaction and rough surface locomotion
```

## QuickStart

### Clone the repository
```bash
git clone https://github.com/rdlynx19/hsa.git
cd hsa
```

### Create and activate virtual environment
```
python3 -m venv mujoco_venv
source mujoco_venv/bin/activate
pip install -r requirements.txt
```

### Running a Demo
```
cd torsional
python3 demos/<script_directory>/<script_name> <model_directory>/<model_name>

Eg:
python3 demos/extend_contract/staggered_wave.py 8Actuators/8_actuator.xml
```

## Customization
- Edit XML models in `torsional/models/` to modify spring placements or parameters
- Write control scripts using MuJoCo's python bindings and helper utilities in `torsional/scripts/`

## Demo

More demos can be found in `demos/` directory.


## Credits 
This work was developed as part of my final project for the MS in Robotics program at Northwestern University, under the guidance of Prof. Matt Elwin and Prof. Ryan Truby.
