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

### Mujoco (Week 8 9/25 - 10/2)
- Restructured model to incorporate three different types of actuators
	- Torque control
	- Position control
	- Velocity control
- Enabled toggling for enabling and disabling groups of actuators i.e. switching between position control, torque control, and velocity control
- Corresponding Motor IDs: 0(1a) 1(4c) 2(3a) 5(2c) 4(1c) 3(4a) 6(3c) 7(2a)
- LookUp Table for actuated motors in different locomotion modes:

	| Locomotion Mode | Motor IDs | Sim Motor IDs | Notes | Range|
	|----------|----------|----------|----------|------|
	| Drive (Velocity Control)  | 0, 1, 2, 3, 4, 5, 6, 7 | 1a, 2c, 3a, 4c, 1c, 2a, 3c, 4a  | 0,1,2,5 forward 3,4,6,7 backward|scaling_factor * 440|
	| Rotation in Drive |  |   | ||
	| Crawl (Position Control)  | 0, 1, 2, 5 | 1a, 4c, 3a, 2c | Extend & Retract repeatedly on either side||
	| Rotation in Crawl||||
	| Paddling (Position Control) |0, 2, 3, 7| 1a, 3a, 4a, 2a|0,2 +ext & 3,7 -ext & 3,7 0|+4000 and -4000|
	| Extension (Position Control) |0, 1, 2, 3, 4, 5, 6, 7| 1a, 2c, 3a, 4c, 1c, 2a, 3c, 4a|0,2,4,6 +extension 1,5,3,7 -extension|scaling_factor * 4000|
	| Twist Direction Zero (Position Control) | 0, 2 or 0,6| 1a, 3a or 1a,3c |0,2 +ext| +5200 (or arbitrary value) from current position|
	| Twist Direction One (Position Control) | 1, 5 or 1,7| 4c, 2c or 4c, 2a |1,5 -ext| -5200 (or arbitrary value) from current position|
	| Bending (Position Control) | 0, 1, 2, 5| 1a, 4c, 3a, 2c |0+,5- or 2+,5- or !(5+,0-) or 2+,1-| scaling_factor * 4000 (signs mentioned)|
	| Turn (Velocity Control) | 0, 2, 4, 6| 1a, 3a, 1c, 3c|4,6+ then 0,6- & 2,4+| 20 then 80|

- Redistributed inertia to make model more stable and accurate to real world
- Added anistropic friction using `contact` in MuJoCo
- Writing a wrapper around the MuJoCo API to streamline the simulation and have beter control over each locomotion mode of the robot
		
### Mujoco (Week 9 10/2 - 10/9)
- More progress on the wrapper, now we have driving, crawling, extension, contraction, bending and twisting functions
- Improvements are needed in making the motions more realistic to the real robots
- Added state machine, with different states - need to encode valid transitions

### Mujoco (Week 10 10/9 - 10/16)
- Found a good trick to add more stability to the structure - we can lock the intermediate discs when they are not being actuated using equality constraints
- Added function to dynamically manipulate the equality constraints during runtime so we can quickly switch from one mode to another
- Look Up Table for knowing which constraints are active/inactive during a particular mode

| Locomotion Mode | Constraints Active | Constraints Inactive | Notes |
|-----------------|-------------------- |---------------------| ------|
| Bend Right|1 and 4|  2 and 3| |
| Bend Left |2 and 3| 1 and 4| |
| Twist 1|1 and 3 | 2 and 4|
|Twist 2| 2 and 4| 1 and 3| 

