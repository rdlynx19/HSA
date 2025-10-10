# HSA Spring Actuation Locomotion in MuJoCo

This repository explores locomotion for an HSA model created with an arrangement of linear springs. It provides few varians of the simulation models and Python scripts to prototype and evaluate various gait strategies using MuJoCo.

## Folder Structure
```
.
├── torsional/                 # Torsional spring locomotion models and API
│   ├── models/                # MuJoCo XML models
│   ├── demos/                 # Predefined gait sequences and simulations
│   └── controlAPI.py          # Helper API for controlling models
├── demos_vids/                # Demo videos of different locomotion modes (outdated)
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── activate.sh                # Activate virtual environment
├── deactivate.sh              # Deactivate virtual environment
├── DevLog.md                  # Development notes and logs
└── README.md

```

## QuickStart

### Clone the repository
```bash
git clone https://github.com/rdlynx19/hsa.git
cd hsa
```

### Create and activate virtual environment
```
# Install and set Python version with pyenv (if not already done)
pyenv install 3.10.4      
pyenv virtualenv 3.10.4 hsa_venv
pyenv activate hsa_venv

# Install dependencies
pip install -r requirements.txt
```

### Running a Demo
```
cd torsional
python3 demos/<demo_name>

Eg:
python3 demos/velocity_drive.py
```

## Customization
- Edit XML models in `torsional/models/` to modify spring placements or parameters
- Write demo scripts using the controlAPI in `torsional/controlAPI.py`


## Credits 
This work was developed as part of my final project for the MS in Robotics program at Northwestern University, under the guidance of Prof. Matt Elwin and Prof. Ryan Truby.
