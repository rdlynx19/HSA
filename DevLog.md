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

### Mujoco (Week 4 5/14 - 5/21)
- Torsional spring structure using 3 cylindric cables (rigid bodies) and 4 tendons (springs)
- Must have Y offset and zig zag connection to produce extension
- Mass calculations for cables:
	- Each individual cable cylinder has volume 0.141372 L
	- If I want their mass to be 20g (so that entire HSA structure can be around 60g)
	- Density parameter should be set to **150**
- Mass calculations for blocks:
	- Based on experimental data, the ratio of block mass to HSA mass should be 8:1
	- So given the volume of each block is 0.5 L
	- If the mass is to be 480g
	- Density parameter should be set to **960**

### Mujoco (Week 5 5/21 - 5/28)
- Derive equations of relationship between torque applied and the spring extension
- Find what parameters should be adjusted to produce more extension in the springs
- Some Results for the Single Actuator Model

| Extension | Stiffness | Structure  | Joints | Result                 |
| --------- | --------- | ---------- | ------ | ---------------------- |
| ~ 40 mm   | 3000      | Asymmetric | HHS    | ![[40mmExtension.png]] |
| ~ 50mm    | 5000      | Asymmetric | HHS    | ![[50mmExtension.png]] |
| ~ 70mm    | 5000      | Asymmetric | HHS    | ![[70mmExtension.png]] |
- Results on the Two Block Model

| Extension | Stiffness | Structure         | Joints | Result                 |
| --------- | --------- | ----------------- | ------ | ---------------------- |
| ~ 25mm    | 3000      | Asymmetric Blocks | HHF    | ![[25mmExtension.png]] |
| ~ 20mm    | 300       | Asymmetric Blocks | HHF    | ![[20mmExtension.png]] |

### Mujoco (Week 6 5/28 - 6/4)
- Recorded some plots and organized results in repository
- Tried chaining the actuators together - not helpful, losing out on extension
- Remove collision between the central cylindrical plates and ground
- Tested out demos and recorded videos for
	- Extension & Contraction
	- Bending
	- Rolling
- Wrote up README and documentation

### Mujoco (Week 7 9/18 - 9/25)
- Reorganized repository with common launch script for all demos
- Trying to figure out anistropic friction in Mujoco
	- [Ice Skating](https://github.com/google-deepmind/mujoco/issues/67)
	- [Same Axis](https://github.com/google-deepmind/mujoco/issues/514)
- Changed contact points to end springs