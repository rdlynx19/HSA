# Emergent Locomotion in Handed Shearing Auxetic (HSA) Actuated Robot

This project focused on modeling, simulating, and developing locomotion gaits for a soft-robotic module based on Handed Shearing Auxetic (HSA) actuators. The primary goal was to reproduce the physical module’s behavior in simulation, and then apply learning-based methods to discover novel and unintuitive gaits. This documentation contains the definition of the HSA Gym environment, the curriculum learning framework, terrain generation utilities, and reinforcement learning training and evaluation scripts. Additionally, it includes an interface using the MuJoCo API used for preliminary testing and demonstration of standard gaits.

[Portfolio Post](https://pushkardave.com/hsa-rl)

[Sphinx Documentation](https://pushkardave.com/HSA)

## QuickStart

1. Clone the repository:
```bash
git clone https://github.com/rdlynx19/hsa.git
cd hsa
```

2. Create and activate a virtual environment:
```bash
# Python ≥ 3.12
python3 -m venv venv

# Install dependencies
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$PWD
```

3. Run the trained model
```bash
python rl_scripts/action_eval.py [-h] [--demo {corridor, flat}] [--episodes EPISODES]

Evaluate trained PPO models for HSA Robot Locomotion.

options:
  -h, --help            show this help message and exit
  --demo {corridor,flat}
                        Select which demo model to evaluate: corridor or flat (default: flat)
  --episodes EPISODES   Number of episodes to run for evaluation (default: 2)
```

## Project Structure
```
HSA/
├── hsa_gym/                  # Custom Gymnasium environment for HSA robot
│   └── envs/                 # Environment implementations
├── models/                   # Pre-trained PPO models
│   ├── ppo_curriculum_corridor/    # Corridor navigation (29M steps)
│   └── ppo_curriculum_flat_small/  # Flat terrain (100M steps)
├── rl_scripts/               # Training and evaluation scripts
│   ├── action_eval.py        # Evaluate trained models
│   └── ppo_curriculum.py     # Train with curriculum learning
├── control_api/              # Low-level control interface
├── control_demos/            # Control demonstrations
├── util_scripts/             # Utility scripts
├── tensorboard_logs/         # Training logs
├── requirements.txt          # Python dependencies
└── README.md
```

## Credits 
This work was developed as part of my final project for the MS in Robotics program at Northwestern University, under the guidance of Prof. Matt Elwin, Prof. Ryan Truby and Dr. Taekyoung Kim.
