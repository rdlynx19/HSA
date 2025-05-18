# Handed Shearing Auxetics
### Simulation (Week 1 4/24 - 4/30)
- Elastica: https://github.com/GazzolaLab/PyElastica?tab=readme-ov-file
	- Requires Python version (>= 3.10 and <3.12)
- Mujoco: https://github.com/google-deepmind/mujoco
	- Python Docs: https://mujoco.readthedocs.io/en/stable/python.html
	- pip installation, so activate virtual env
	- `python -m mujoco.viewer` to launch the simulation
### Mujoco (Week 2 4/30 - 5/7)
- Simulated 2 blocks attached with springs in Mujoco
- Mujoco docs has lots of detailed information on modeling, forces and other kinds of setup. Things that should be thought about a bit more might be 
	- Impedance
	- Stiffness and Damping
	- Friction: Can have anistropic friction between a contact pair, but not for an individual geometry
	- Muscles: Is there a muscle which can act like hsa?
	- https://github.com/google-deepmind/mujoco/tree/main/model/plugin/elasticity

### Mujoco (Week 3 5/7 - 5/14)
- Implemented hybrid spring cable model which is close to what we want
- Just cable or just spring models won't work well
- Springs only have linear actuators which change their lengths, to actuate in a twisting manner, you need a cable
