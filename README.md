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
python3 -m venv mujoco_venv
source mujoco_venv/bin/activate
