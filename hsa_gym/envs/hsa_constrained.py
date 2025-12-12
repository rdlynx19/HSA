"""
HSAEnv Module: Custom MuJoCo Environment for Handed Shearing Auxetic (HSA) Robot Locomotion.

This module defines the `HSAEnv` class, a specialized child environment inherited 
from `CustomMujocoEnv`. It implements a 2D locomotion task where a robot composed 
of two blocks connected by HSA actuators must move towards a goal marker.

Key features implemented in this file include:

* **Reward Function:** A complex dense reward structure incorporating forward progress, 
    control smoothness, joint constraints, acceleration limits, and contact penalties.
* **Curriculum/Goals:** Logic for generating goals for progressive 
    task difficulty (when enabled). 
* **Terrain Interaction:** Logic for spawning the robot safely above procedurally 
    generated terrain (heightfield) and managing vertical termination limits relative 
    to the terrain bounds.
* **Observation Space:** Construction of a detailed observation vector using 
    actuated joint states, COM kinetics, and goal-relative vectors.
"""

import numpy as np
from numpy.typing import NDArray

from gymnasium import utils
from .mujoco_env_constrained import CustomMujocoEnv
from gymnasium.spaces import Box

class HSAEnv(CustomMujocoEnv):
    """
    HSA Environment Class for MuJoCo-based simulation.
    In this environment, a robot must learn to move towards a direction
    in the XY plane by coordinating its two blocks. The faster it moves, the more reward it gets. We are trying to train a locomotion policy here.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, 
                 xml_file: str = "hsaLooseModel.xml",
                 frame_skip: int = 4,
                 default_camera_config: dict[str, float | int] = {},
                 forward_reward_weight: float = 10.0,
                 ctrl_cost_weight: float = 1e-3,
                 actuator_group: list[int] = [1],
                 action_group: list[int] = [1],
                 smooth_positions: bool = True,
                 clip_actions: float = 1.0,
                 contact_cost_weight: float = 1e-4,
                 yvel_cost_weight: float = 1.0,
                 constraint_cost_weight: float = 1e-2,
                 acc_cost_weight: float = 1e-2,
                 distance_reward_weight: float = 1.0,
                 early_termination_penalty: float = 50.0,
                 joint_vel_cost_weight: float = 1e-3,
                 stagnation_penalty_weight: float = 0.15,
                 alive_bonus: float = 0.1,
                 max_increment: float = 3.14,
                 enable_terrain: bool = False,
                 terrain_type: str = "craters",
                 goal_position: list[float] = [3.0, 1.0, 0.1],
                 num_checkpoints: int = 20,
                 checkpoint_reward: float = 40.0,
                 checkpoint_radius: float = 0.4,
                 start_radius: float = 0.5,
                 end_radius: float = 4.0,
                 num_turns: int = 2.0,
                 max_episode_steps: int = 4000,
                 ensure_flat_spawn: bool = True,
                 **kwargs):
        """
        Initialize the HSA Environment, setting reward weights and simulation parameters.

        This calls the base class :py:class:`CustomMujocoEnv` constructor after 
        setting up all task-specific parameters, reward weights, and internal state 
        variables.

        :param xml_file: MuJoCo model file to load.
        :type xml_file: str
        :param frame_skip: Number of simulation steps per environment step.
        :type frame_skip: int
        :param default_camera_config: Configuration dictionary for rendering camera.
        :type default_camera_config: dict[str, float or int]
        :param forward_reward_weight: Weight for the reward component based on forward velocity (towards goal).
        :type forward_reward_weight: float
        :param ctrl_cost_weight: Weight for the control cost (penalizing change in action).
        :type ctrl_cost_weight: float
        :param actuator_group: List of actuator group IDs to enable.
        :type actuator_group: list[int]
        :param action_group: List of actuator group IDs defining the action space.
        :type action_group: list[int]
        :param smooth_positions: Whether to interpolate control targets over frames.
        :type smooth_positions: bool
        :param clip_actions: Value to clip the action space to (redundant if using Gym Box).
        :type clip_actions: float
        :param contact_cost_weight: Weight for the penalty on excessive foot contact forces.
        :type contact_cost_weight: float
        :param yvel_cost_weight: Weight for the penalty on lateral velocity (y-axis movement).
        :type yvel_cost_weight: float
        :param constraint_cost_weight: Weight for the penalty on violating angular joint difference constraints.
        :type constraint_cost_weight: float
        :param acc_cost_weight: Weight for the penalty on high joint accelerations.
        :type acc_cost_weight: float
        :param distance_reward_weight: Weight for the reward component based on progress towards the goal.
        :type distance_reward_weight: float
        :param early_termination_penalty: Fixed or dynamic penalty subtracted from the total reward upon premature termination.
        :type early_termination_penalty: float
        :param joint_vel_cost_weight: Weight for the penalty on high joint velocities.
        :type joint_vel_cost_weight: float
        :param stagnation_penalty_weight: Weight for the penalty applied when the robot makes little progress or idles.
        :type stagnation_penalty_weight: float
        :param alive_bonus: Reward added at every step the environment is not terminated.
        :type alive_bonus: float
        :param max_increment: Maximum angle (in radians) that the control target can be incremented by per step.
        :type max_increment: float
        :param enable_terrain: Whether to enable procedural terrain generation.
        :type enable_terrain: bool
        :param terrain_type: Type of terrain to generate (e.g., ``"craters"``, ``"spiral"``).
        :type terrain_type: str
        :param goal_position: Initial $XYZ$ position of the goal marker.
        :type goal_position: list[float]
        :param num_checkpoints: Number of checkpoints to generate along a spiral path (if ``terrain_type='spiral'``).
        :type num_checkpoints: int
        :param checkpoint_reward: Bonus reward given upon reaching a checkpoint.
        :type checkpoint_reward: float
        :param checkpoint_radius: Distance threshold to collect a checkpoint.
        :type checkpoint_radius: float
        :param start_radius: Inner radius for the spiral path generation.
        :type start_radius: float
        :param end_radius: Outer radius for the spiral path generation.
        :type end_radius: float
        :param num_turns: Number of turns in the spiral path.
        :type num_turns: int
        :param max_episode_steps: Maximum steps before environment truncation (handled by wrapper).
        :type max_episode_steps: int
        :param ensure_flat_spawn: Whether to flatten the terrain around the robot's spawn area.
        :type ensure_flat_spawn: bool
        :param kwargs: Additional keyword arguments passed to the base class.
        :type kwargs: dict
        """

        # --- Reweight Parameters ---
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.yvel_cost_weight = yvel_cost_weight
        self.constraint_cost_weight = constraint_cost_weight
        self.acc_cost_weight = acc_cost_weight
        self.early_termination_penalty = early_termination_penalty
        self.distance_reward_weight = distance_reward_weight
        self.alive_bonus = alive_bonus 
        self.joint_vel_cost_weight = joint_vel_cost_weight
        self.stagnation_penalty_weight = stagnation_penalty_weight
        
        # --- Config Parameters ---
        self.actuator_group_ids = actuator_group
        self.clip_actions = clip_actions

        self.goal_position_xyz = np.array(goal_position, dtype=np.float64)

        self.max_increment = max_increment
        self.enable_terrain_flag = enable_terrain
        self.terrain_type_str = terrain_type

        # --- Checkpoint Parameters ---
        self.num_checkpoints = num_checkpoints
        self.checkpoint_reward = checkpoint_reward
        self.checkpoint_radius = checkpoint_radius
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.num_turns = num_turns

        self.max_episode_steps = max_episode_steps

        self.qvel_limit = 2000.0 # Max joint velocity for termination condition
        self.qacc_limit = 15000.0 # Max joint acceleration for termination

        # --- Runtime State ---
        self.step_count = 0
        self.terrain_z_min = 0.0
        self.terrain_z_max = 0.0
        self.position_history = []
        self.progress_window = 50

        # --- Base Class Call ---
        CustomMujocoEnv.__init__(self,
                                 xml_file,
                                 frame_skip,
                                 observation_space=None,
                                 default_camera_config=default_camera_config,
                                 actuator_group=actuator_group,
                                 action_group=action_group,
                                 smooth_positions=smooth_positions,
                                 enable_terrain=enable_terrain,
                                 terrain_type=terrain_type,
                                 goal_position=goal_position,
                                 ensure_flat_spawn=ensure_flat_spawn,
                                 **kwargs)

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.dt))
        }

        # --- Observation Indices ---
        self.actuated_qpos_indices = [7, 22, 10, 24, 20, 9, 21, 12]
        self.actuated_qvel_indices = [6, 20, 9, 22, 18, 8, 19, 11]

        # Observation Size
        observation_size = (
            len(self.actuated_qpos_indices)
            + len(self.actuated_qvel_indices)
            + 2  # XY com position of robot
            + 3  # XYZ velocity of robot
            + 1 # Distance to goal
            + 2 # Vector to goal position
        )

        # Observation Space
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float64
        )

        # Observation Structure
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "com_position": 2,
            "com_velocity": 3,
            "distance_to_goal": 1,
            "vec_to_goal": 2
        }
  
        # Previous action for smoothing reward calculation
        self.prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        self.prev_scaled_action = np.zeros(self.action_space.shape[0], 
                                          dtype=np.float32)
        
        # Previous velocity for acceleration cost
        self.prev_joint_velocities = np.zeros(
            len(self.actuated_qvel_indices), 
            dtype=np.float32
        )

        # Corrected bug: use stored goal_position_xyz, not local parameter
        self.prev_distance_to_goal = np.linalg.norm(
            self.compute_COM() - self.goal_position_xyz[:2]
        )
        
        # Get checkpoint positions
        if self.terrain_type_str == "spiral":
            self.checkpoint_positions = self.calculate_spiral_checkpoints()
            self.current_checkpoint_index = 0
        else:
            self.checkpoint_positions = []
            self.current_checkpoint_index = 0


    def calculate_spiral_checkpoints(self) -> list[NDArray[np.float64]]:
        """
        Calculate evenly spaced checkpoints along a clockwise spiral path.

        It applies a critical starting angle offset (currently $\pi$ radians) 
        to align the path with terrain features like valleys. 

        :returns: A list of 2D NumPy arrays, where each array is an $[x, y]$ world-coordinate position of a checkpoint.
        :rtype: list[NDArray[np.float64]]
        """
        checkpoints = []
        total_angle = 2 * np.pi * self.num_turns
        
        # CRITICAL: Starting angle offset to align with valley
        starting_angle_offset = np.pi   # Adjust as needed
        
        for i in range(self.num_checkpoints):
            progress = (i + 1) / self.num_checkpoints
            
            # Clockwise with offset
            angle = -(progress * total_angle) + starting_angle_offset
            
            radius = self.start_radius + (self.end_radius - self.start_radius) * progress
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            checkpoints.append(np.array([x, y], dtype=np.float64))
    
        return checkpoints

    def check_checkpoints(self) -> float: 
        """
        Check if the robot has reached the current checkpoint, and advance the goal if so.

        This function is only active when :py:attr:`self.terrain_type_str` is ``"spiral"``.

        :returns: The checkpoint bonus reward if a checkpoint was collected, otherwise $0.0$.
        :rtype: float
        """
        if not self.checkpoint_positions:
            return 0.0
    
        # Check if current checkpoint (which is goal) is reached
        if self.current_checkpoint_index >= len(self.checkpoint_positions):
            return 0.0  # All checkpoints collected
        
        current_pos = self.compute_COM()
        checkpoint_pos = self.checkpoint_positions[self.current_checkpoint_index]
        distance = np.linalg.norm(current_pos - checkpoint_pos)

        if distance < self.checkpoint_radius:
            self.checkpoints_collected.add(self.current_checkpoint_index)
            bonus = self.checkpoint_reward

            # Advance to next checkpoint
            self.current_checkpoint_index += 1
            if self.current_checkpoint_index < len(self.checkpoint_positions):
                next_checkpoint = self.checkpoint_positions[self.current_checkpoint_index]
                self.goal_position_xyz = np.array([next_checkpoint[0], next_checkpoint[1], 0.1], dtype=np.float64)
                self.update_goal_marker(goal_position=self.goal_position_xyz)
                self.prev_distance_to_goal = np.linalg.norm(current_pos - self.goal_position_xyz[:2])
            else:
                pass  # All checkpoints collected
            
            return bonus

        return 0.0


    def set_curriculum_manager(self, curriculum_manager) -> None: 
        """
        Set the curriculum manager instance for the environment.

        The curriculum manager controls task difficulty, such as defining 
        the sampling distribution for goal positions during 
        :py:meth:`~HSAEnv.reset_model`.

        :param curriculum_manager: An instance of a curriculum manager class (e.g., ``CurriculumManager``).
        :type curriculum_manager: object
        :returns: None
        :rtype: None
        """
        self.curriculum_manager = curriculum_manager

    def compute_terrain_bounds(self) -> None: 
        """
        Compute and store the actual world height bounds (min and max Z-coordinates) for the generated terrain.

        This function must be called after the model has been initialized with the terrain data.

        :returns: None
        :rtype: None
        """
        if not self.enable_terrain_flag:
            self.terrain_z_min = 0.0
            self.terrain_z_max = 0.0
            return

        hfield_size = self.model.hfield_size[0]
        x_half, y_half, z_max, base_height = hfield_size

        nrow = self.model.hfield_nrow[0]
        ncol = self.model.hfield_ncol[0]
        terrain_data = self.model.hfield_data.reshape(nrow, ncol)

        store_min = terrain_data.min()
        store_max = terrain_data.max()

        self.terrain_z_min = base_height + store_min * z_max
        self.terrain_z_max = base_height + store_max * z_max
        # print(f"[Terrain] Height bounds: min={self.terrain_z_min:.3f}m, max={self.terrain_z_max:.3f}m")

    def get_spawn_height(self, x, y) -> float:
        """
        Get the actual world Z-coordinate height of the terrain at the given $(x, y)$ world coordinates. 

        :param x: World $X$-coordinate of the desired spawn location.
        :type x: float
        :param y: World $Y$-coordinate of the desired spawn location.
        :type y: float
        :returns: The terrain height in meters at the given $(x, y)$ position.
        :rtype: float
        """
        hfield_size = self.model.hfield_size[0]
        x_half, y_half, z_max, base_height = hfield_size

        nrow = self.model.hfield_nrow[0]
        ncol = self.model.hfield_ncol[0]
        # World coordinates range from [-x_half, x_half] and [-y_half, y_half]
        grid_i = int((x + x_half) / (2 * x_half) * (nrow - 1))
        grid_j = int((y + y_half) / (2 * y_half) * (ncol - 1))
        # Clamp to valid range
        grid_i = np.clip(grid_i, 0, nrow - 1)
        grid_j = np.clip(grid_j, 0, ncol - 1)

       # Get normalized height (0 to 1)
        terrain_data = self.model.hfield_data.reshape(nrow, ncol)
        stored_height = terrain_data[grid_i, grid_j]

           # Convert to actual height
        actual_height = base_height + stored_height * z_max
        
        return actual_height

    def distance_cost(self, goal_position: NDArray[np.float64]) -> float:
        """
        Compute the progress reward based on the change in distance to the goal.

        :param goal_position: The 3D position vector ($[x, y, z]$) of the current goal target.
        :type goal_position: NDArray[np.float64]
        :returns: The distance progress reward (scalar float).
        :rtype: float
        """
        current_position = self.compute_COM()
        distance = np.linalg.norm(current_position - goal_position[:2])

        # Progress reward: positive if getting closer, negative if moving away
        distance_progress = self.prev_distance_to_goal - distance

        self.prev_distance_to_goal = distance
        return distance_progress

    def acceleration_cost(self) -> float:
        """
        Compute the cost based on high joint accelerations (changes in velocity).
        Acceleration is calculated using a finite difference approximation over the environment timestep 
        
        :returns: The weighted acceleration cost.
        :rtype: float
        """
        joint_velocities = self.data.qvel[self.actuated_qvel_indices]
    
        # Compute acceleration as change in velocity
        joint_acc = (joint_velocities - self.prev_joint_velocities) / self.dt

        # Clip to prevent explosion
        joint_acc = np.clip(joint_acc, -50.0, 50.0)
        # Acceleration cost
        acc_cost = self.acc_cost_weight * np.sum(np.square(joint_acc))
        # Cap cost
   
        # Update previous velocities
        self.prev_joint_velocities = joint_velocities.copy()

        return acc_cost

    def constraint_cost(self, diff_margin: float = 0.01, 
                        penalty_factor: float = 1.0, bonus_factor: float = 0.025) -> tuple[float, float]:
        """
        Compute penalty and bonus for violating or satisfying angular difference constraints between paired joints.

        This mechanism ensures the robot's paired joints (e.g., 1A and 1C) maintain a coupled angular difference.
        * **Penalty:** Applied when the absolute angular difference $|A - C|$ exceeds a threshold, $\pi - \text{diff\_margin}$. The penalty grows quadratically with proximity to $\pi$.
        * **Bonus:** A small bonus is awarded for each pair where the difference is maintained below the threshold. 

        :param diff_margin: The angular margin before $\pi$ where the quadratic penalty begins to apply.
        :type diff_margin: float
        :param penalty_factor: Scaling factor for the quadratic penalty on constraint violation.
        :type penalty_factor: float
        :param bonus_factor: Constant bonus awarded per constraint pair that is satisfied (difference is small).
        :type bonus_factor: float
        :returns: A tuple containing the total constraint penalty (cost) and the total constraint satisfaction bonus.
        :rtype: tuple[float, float]
        """
        # Order of indices is 1A, 2A, 3A, 4A, 1C, 2C, 3C, 4C
        joint_positions = {
            "1A": self.data.qpos[7],
            "2A": self.data.qpos[22],
            "3A": self.data.qpos[10],
            "4A": self.data.qpos[24],
            "1C": self.data.qpos[20],
            "2C": self.data.qpos[9],
            "3C": self.data.qpos[21],
            "4C": self.data.qpos[12],
        }
        control_targets = {
            "1A": self.data.ctrl[8],
            "2A": self.data.ctrl[9],
            "3A": self.data.ctrl[10],
            "4A": self.data.ctrl[11],
            "1C": self.data.ctrl[12],
            "2C": self.data.ctrl[13],
            "3C": self.data.ctrl[14],
            "4C": self.data.ctrl[15],
        }

        constraint_cost = 0.0
        constraint_bonus = 0.0
        diff_threshold = np.pi - diff_margin

        pairs = [
            ("1A", "1C", "Pair1"),
            ("2A", "2C", "Pair2"),
            ("3A", "3C", "Pair3"),
            ("4A", "4C", "Pair4"),
        ]

        for a_name, c_name, pair_name in pairs:
            pos_a = joint_positions[a_name]
            pos_c = joint_positions[c_name]

            # # Get absolute difference
            diff = abs(pos_a - pos_c)
            # Only penalize if difference exceeds threshold
            if diff > diff_threshold:
                # Quadratic penalty that grows as we approach pi
                proximity = (diff - diff_threshold) / diff_margin
                proximity = np.clip(proximity, 0.0, 1.0)

                # Quadratic growth: penalty = k * proximity^2
                pair_penalty = penalty_factor * (proximity ** 2)
                constraint_cost += pair_penalty
            else:
                constraint_bonus += bonus_factor 
            
        # Handedness constraint can be assumed to be enforced by the threshold constraint
        return constraint_cost, constraint_bonus

    def contact_cost(self) -> float:
        """
        Compute the cost based on excessive contact forces experienced by the robot's foot geometries.

        :returns: The weighted total contact cost.
        :rtype: float
        """
        foot_geom_names = ["cylinder3a_con", "cylinder3c_con",
                           "cylinder4a_con", "cylinder4c_con"]
        total_excessive_force = 0.0
        force_threshold = 50.0  # N Threshold for excessive force
        for geom_name in foot_geom_names:
            try:
                excessive_force = self.get_contact_force(geom_name, 
                                                         max_normal_force=force_threshold)
                total_excessive_force += excessive_force ** 2 # Square the excessive force
            except:
                pass
        
        contact_cost = self.contact_cost_weight * total_excessive_force
        
        return contact_cost


    # Control cost to penalize large actions
    def control_cost(self, action: NDArray[np.float32]) -> float:
        """
        Compute the control cost based on the change in the action vector (motor command smoothness).
 
        This cost encourages smooth, low-frequency control signals. 

        :param action: The current normalized action vector supplied by the agent.
        :type action: NDArray[np.float32]
        :returns: The weighted control cost.
        :rtype: float
        """
        # Compute the difference between current and previous actions
        action_diff = action - self.prev_action
        control_cost = self.ctrl_cost_weight * np.sum(np.square(action_diff))
        self.prev_action = action.copy()

        return control_cost
    
    def joint_velocity_cost(self) -> float:
        """
        Compute the cost associated with high joint velocities.
        This penalizes excessive rotational speeds in the robot's joints.

        :returns: The weighted joint velocity cost.
        :rtype: float
        """
        joint_velocities = self.data.qvel[self.actuated_qvel_indices]
        joint_velocities = np.clip(joint_velocities, -50.0, 50.0)
        joint_vel_cost = self.joint_vel_cost_weight * np.sum(np.square(joint_velocities))
        return joint_vel_cost
    
    def vec_to_goal(self) -> NDArray[np.float64]:
        """
        Compute the 2D unit vector pointing from the robot's Center of Mass (COM) to the current goal position.

        :returns: The 2D unit vector (length 1) to the goal position.
        :rtype: NDArray[np.float64]
        """
        current_position = self.compute_COM()
        goal_vector = self.goal_position_xyz[:2] - current_position
        distance_to_goal = np.linalg.norm(goal_vector)
        goal_unit_vector = goal_vector / (distance_to_goal + 1e-8)  # Avoid division by zero
        return goal_unit_vector

    def check_termination(self, observation: NDArray[np.float64]) -> tuple[bool, list[str]]:
        """
        Check for conditions that cause the episode to terminate prematurely.

        Termination occurs if any dynamic state (qpos, qvel, qacc) contains NaNs/Infs, 
        if the robot's blocks exceed vertical $Z$-limits (relative to terrain if enabled), 
        or if the maximum joint velocity (:py:attr:`self.qvel_limit`) or acceleration 
        (:py:attr:`self.qacc_limit`) are surpassed.

        :param observation: The current observation vector (checked for NaNs/Infs).
        :type observation: NDArray[np.float64]
        :returns: A tuple containing a boolean flag (True if termination conditions are met) 
            and a list of strings detailing the reason(s) for termination.
        :rtype: tuple[bool, list[str]]
        """
        reasons = []

        if np.isnan(self.data.qpos).any(): reasons.append("qpos_nan")
        if np.isinf(self.data.qpos).any(): reasons.append("qpos_inf")

        if np.isnan(self.data.qvel).any(): reasons.append("qvel_nan")
        if np.isinf(self.data.qvel).any(): reasons.append("qvel_inf")

        if np.isnan(self.data.qacc).any(): reasons.append("qacc_nan")
        if np.isinf(self.data.qacc).any(): reasons.append("qacc_inf")

        # Body out of range
        z_a = self.get_body_com("block_a")[2]
        z_b = self.get_body_com("block_b")[2]
        
        if self.enable_terrain_flag:
            # Safe margin above terrain
            safe_margin = 0.5
            upper_limit = self.terrain_z_max + safe_margin
            lower_limit = self.terrain_z_min - safe_margin

            if z_a > (upper_limit):
                reasons.append("block_a_too_high")
            if z_b > (upper_limit):
                reasons.append("block_b_too_high")
            if z_a < (lower_limit):
                reasons.append("block_a_too_low")
            if z_b < (lower_limit):
                reasons.append("block_b_too_low")
        else:
            if z_a > 0.5: reasons.append("block_a_too_high")
            if z_b > 0.5: reasons.append("block_b_too_high")
            if z_a < -0.5: reasons.append("block_a_too_low")
            if z_b < -0.5: reasons.append("block_b_too_low")

        # Observation issues
        if np.isnan(observation).any(): reasons.append("obs_nan")
        if np.isinf(observation).any(): reasons.append("obs_inf")

        # Dynamic limits
        if np.max(np.abs(self.data.qvel)) > self.qvel_limit: reasons.append("qvel_limit")
        if np.max(np.abs(self.data.qacc)) > self.qacc_limit: reasons.append("qacc_limit")

        terminated = len(reasons) > 0
        return terminated, reasons

    def scale_action(self, norm_action: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Scale the normalized action vector to the desired control increment range.
        
        The scaled action is then used by :py:meth:`~CustomMujocoEnv.do_simulation` 
        as the increment to the current actuator control targets.

        :param norm_action: Normalized action vector in the range $[-1, 1]$.
        :type norm_action: NDArray[np.float32]
        :returns: Scaled action vector, representing the desired control target increment.
        :rtype: NDArray[np.float32]
        """
        scaled_action = norm_action * self.max_increment
        
        return scaled_action

    def step(self, action: NDArray[np.float32]
             ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        """
        Advance the environment by one timestep, running the physics simulation and computing all metrics.
        This method implements the core Gymnasium ``step`` logic.

        :param action: The normalized action vector (motor command) supplied by the agent.
        :type action: NDArray[np.float32]
        :returns: A tuple containing the next observation, the step reward, 
            a terminated flag, a truncated flag, and an info dictionary detailing reward components and state.
        :rtype: tuple[NDArray[np.float64], float, bool, bool, dict[str, np.float64]]
        """
        self.step_count += 1
        # Scale action
        scaled_action = self.scale_action(action)

        previous_position = self.compute_COM()
        self.do_simulation(scaled_action,
                           self.frame_skip, 
                           self.actuator_group_ids)
        current_position = self.compute_COM()

        # Calculate velocity
        xy_velocity = (current_position - previous_position) / self.dt
        x_velocity, y_velocity = xy_velocity
        x_velocity = np.clip(x_velocity, -5.0, 5.0)
        y_velocity = np.clip(y_velocity, -5.0, 5.0)

        # Check for checkpoint collection
        checkpoint_bonus = 0.0
        if self.terrain_type_str == "spiral" and self.checkpoint_positions:
            checkpoint_bonus = self.check_checkpoints()
        
        observation = self.get_obs()
        reward, reward_info = self.get_reward(action, x_velocity, y_velocity)
        reward += checkpoint_bonus
        
        terminated, term_reasons = self.check_termination(observation)
        truncated = False
        success = False

        if self.checkpoint_positions and self.terrain_type_str == "spiral":
            if self.current_checkpoint_index >= len(self.checkpoint_positions):
                reward += 100.0
                early_term_pen = 0.0  # No penalty if goal is reached
                terminated = True
                term_reasons.append("goal_reached")
                success = True

        # For other terrains: check if close to goal
        if self.terrain_type_str != "spiral":
            current_distance = np.linalg.norm(
                self.compute_COM() - self.goal_position_xyz[:2]
            )
            if current_distance < 0.20:
                    reward += 40.0
                    early_term_pen = 0.0  # No penalty if goal is reached
                    terminated = True
                    term_reasons.append("goal_reached")
                    success = True

        # Scale early termination penalty
        if terminated and not truncated and not success:
            steps_lost = self.max_episode_steps - self.step_count
            early_term_pen = (self.early_termination_penalty) * steps_lost
            early_term_pen = 100.0
        else: 
            early_term_pen = 0.0
       
        alive_bonus = (
            self.alive_bonus if not (terminated or truncated) else 0.0
        )

        reward = reward + alive_bonus - early_term_pen

        actual_position = self.data.qpos[self.actuated_qpos_indices].copy()

        info = {
            "previous_position": previous_position,
            "current_position": current_position,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "desired_position": scaled_action,
            "actual_position": actual_position,
            "tracking_error": np.mean(np.abs(scaled_action - actual_position)),
            "reward_termination": early_term_pen,
            "reward_alive": alive_bonus,
            "reward_checkpoint_bonus": checkpoint_bonus,   
            "termination_reasons": term_reasons,
            **reward_info
        }

        if self.terrain_type_str == "spiral":
            info.update({
                "checkpoints_collected": len(self.checkpoints_collected),
                "current_checkpoint_index": self.current_checkpoint_index,
                "total_checkpoints": self.num_checkpoints,
                "checkpoint_progress": (len(self.checkpoints_collected) / self.num_checkpoints)
            })

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, truncated, info

    def get_reward(self, 
                    action: NDArray[np.float32],
                    x_velocity: float = 0.0,
                    y_velocity: float = 0.0,
                    ) -> tuple[float, dict[str, float]]:
        """
        Compute the total reward and detailed breakdown for the current step.

        The total reward is calculated by summing various components:
        
        * **Positive Components:** Forward velocity (projection onto goal vector), Lateral velocity (cost, weighted positively here), Distance progress, and Constraint satisfaction bonus.
        * **Negative Components (Costs):** Control cost, contact cost, constraint violation cost, acceleration cost, joint velocity cost, and stagnation penalty.
        
        The reward is constrained by caps placed on individual cost components.

        :param action: The normalized action vector used in the current step.
        :type action: NDArray[np.float32]
        :param x_velocity: The average velocity of the robot's COM along the X-axis.
        :type x_velocity: float
        :param y_velocity: The average velocity of the robot's COM along the Y-axis (lateral).
        :type y_velocity: float
        :returns: A tuple containing the total scalar reward and a dictionary of all reward/cost components.
        :rtype: tuple[float, dict[str, float]]
        """
        com_velocity = np.array([x_velocity, y_velocity])
        projected_velocity = np.dot(com_velocity, self.vec_to_goal())
        # Reward is based on velocity towards goal
        forward_reward = (self.forward_reward_weight) * max(0.0, projected_velocity)

        # Y velocity reward
        lateral_reward = self.yvel_cost_weight * abs(y_velocity)

        # Control cost penalty
        ctrl_cost = self.control_cost(action)
        ctrl_cost = min(ctrl_cost, 20.0)
        # Contact cost penalty
        contact_cost = self.contact_cost()
        contact_cost = min(contact_cost, 30.0)
        # Constraint cost penalty
        constraint_violation, constraint_bonus = self.constraint_cost()
        constraint_cost = self.constraint_cost_weight * constraint_violation
        constraint_cost = min(constraint_cost, 50.0)
        # Acceleration cost penalty
        acc_cost = self.acceleration_cost()
        acc_cost = min(acc_cost, 20.0)
        # Joint velocity cost penalty
        joint_vel_cost = self.joint_velocity_cost()
        joint_vel_cost = min(joint_vel_cost, 50.0)
        # Distance reward
        distance_reward = (
            self.distance_reward_weight * self.distance_cost(self.goal_position_xyz)
        )

        stagnation_penalty = 0.0
        # # Progress Check within Window
        self.position_history.append(self.compute_COM().copy())
        if len(self.position_history) > self.progress_window:
            self.position_history.pop(0)
        if len(self.position_history) >= self.progress_window:
            start_pos = self.position_history[0]
            end_pos = self.position_history[-1]
            progress = np.linalg.norm(end_pos - start_pos)
            if progress < 0.05:
                # Penalize lack of progress
                stagnation_penalty = self.stagnation_penalty_weight * (1.0 - (progress / 0.05))

        # # Reward shaping to encourage getting closer to goal 
        # current_distance = np.linalg.norm(
        #     self.compute_COM() - self.goal_position_xyz[:2]
        # )
        # reward_shaping = 0.1 / (current_distance + 0.5)
        # distance_reward += reward_shaping

        costs = (
            ctrl_cost + contact_cost + constraint_cost + acc_cost + joint_vel_cost
            )
        reward = forward_reward + lateral_reward - costs + distance_reward + constraint_bonus - stagnation_penalty
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl_cost": -ctrl_cost,
            "reward_contact_cost": -contact_cost,
            "reward_lateral": lateral_reward,
            "reward_constraint_cost": -constraint_cost,
            "reward_constraint_bonus": constraint_bonus,
            "reward_acc_cost": -acc_cost,
            "reward_joint_vel_cost": -joint_vel_cost,
            "reward_distance": distance_reward,
            "reward_stagnation_penalty": -stagnation_penalty,
            "reward_total_costs": -costs
        }
    
        return reward, reward_info

    def get_obs(self) -> NDArray[np.float64]:
        """
        Generate the full observation vector for the agent.

        The observation vector is a concatenation of local robot state and goal-relative information, 
        adhering to the structure defined in :py:attr:`self.observation_structure`. The components, 
        in order, are:
        
        1. Actuated joint positions ($qpos$).
        2. Actuated joint velocities ($qvel$).
        3. Robot center of mass (COM) position ($XY$ plane).
        4. Robot base velocity (average of block velocities, $XYZ$).
        5. Distance to goal (scalar).
        6. Unit vector to goal ($XY$).

        :returns: The concatenated observation vector.
        :rtype: NDArray[np.float64]
        """
        pos = self.data.qpos[self.actuated_qpos_indices].copy()
        vel = self.data.qvel[self.actuated_qvel_indices].copy()

        # Current position of the robot's COM
        current_position = self.compute_COM().flatten()

        # Get base velocity (average of both blocks)
        blocka_vel = self.data.qvel[0:3].copy()
        blockb_vel = self.data.qvel[13:16].copy()
        avg_velocity = 0.5 * (blocka_vel + blockb_vel)

        # Get distance to goal
        distance_to_goal = np.linalg.norm(
            self.compute_COM() - self.goal_position_xyz[:2]
        ).reshape(1)
        vec_to_goal = self.goal_position_xyz[:2] - self.compute_COM() 
        norm_vec = vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-8)

        observation = np.concatenate([pos, 
                                      vel, 
                                      current_position, 
                                      avg_velocity, 
                                      distance_to_goal,
                                      norm_vec]).ravel()
        return observation

    def reset_model(self) -> NDArray[np.float64]:
        """
        Set the task-specific initial state (qpos, qvel) of the robot and return the initial observation.
        Clears step count, previous actions, joint velocities, and checkpoint tracking.

        :returns: The initial observation state of the environment.
        :rtype: NDArray[np.float64]
        """
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if self.enable_terrain_flag:
            # Get spawn positions for both blocks
            blocka_x, blocka_y = -0.2, 0.0
            blockb_x, blockb_y = 0.2, 0.0
            # Get terrain height at those positions
            terrain_a_z = self.get_spawn_height(blocka_x, blocka_y)
            terrain_b_z = self.get_spawn_height(blockb_x, blockb_y)

            # Use the higher terrain height to ensure both blocks are above ground
            terrain_height = max(terrain_a_z, terrain_b_z)

            robot_clearance = 0.1  # Meters above terrain
            spawn_z = terrain_height + robot_clearance

            qpos[2] = spawn_z
            qpos[15] = spawn_z

        self.set_state(qpos, qvel)
        
        self.step_count = 0
        # Initialize previous action at reset
        self.prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        
        self.prev_scaled_action = np.zeros(self.action_space.shape[0], 
                                          dtype=np.float32)
        
        self.prev_joint_velocities = np.zeros(
            len(self.actuated_qvel_indices), 
            dtype=np.float32
        )

        self.checkpoints_collected = set()
        self.current_checkpoint_index = 0

        if self.terrain_type_str == "spiral":
            marker_x, marker_y = self.checkpoint_positions[0]
            marker_z = 0.1
            self.goal_position_xyz = np.array([marker_x, marker_y, marker_z], dtype=np.float64)
            self.update_goal_marker(goal_position=[marker_x, marker_y, marker_z])
            print(f"[Spiral] Goal set to checkpoint 1/{len(self.checkpoint_positions)}")
        else:
            # Curriculum based goal sampling
            if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
                # Use curriculum manager to sample goal
                goal_pos = self.curriculum_manager.sample_goal_position()
                marker_x, marker_y, marker_z = goal_pos

            else:
                ranges = [(-2.0, -1.5), (1.5, 2.0)]
                low, high = ranges[np.random.choice([0, 1])]
                marker_x = np.random.uniform(low, high)
                marker_y = np.random.uniform(-0.0, 0.0)
                marker_z = 0.1

            self.update_goal_marker(goal_position=[marker_x, marker_y, marker_z])
            self.goal_position_xyz = np.array([marker_x, marker_y, marker_z], dtype=np.float64)

        self.prev_distance_to_goal = np.linalg.norm(
            self.compute_COM() - self.goal_position_xyz[:2]
        )

        observation = self.get_obs()
        return observation
    
    def compute_COM(self) -> NDArray[np.float64]:
        """
        Compute the projected Center of Mass (COM) position of the robot in the $XY$ plane.

        :returns: The 2D $XY$ position vector of the robot's approximate COM.
        :rtype: NDArray[np.float64] 
        """
        blocka_pos = self.get_body_com("block_a").copy()
        blockb_pos = self.get_body_com("block_b").copy()

        # Center of Mass position
        return 0.5 * (blocka_pos[:2] + blockb_pos[:2])
    
    def get_reset_info(self) -> dict[str, np.float64]:
        """
        Generate the initial ``info`` dictionary returned during a :py:meth:`~CustomMujocoEnv.reset`.

        :returns: A dictionary containing initial state and performance information.
        :rtype: dict[str, np.float64]
        """
        previous_position = self.compute_COM()
        current_position = self.compute_COM()

        xy_velocity = (current_position - previous_position) / self.dt
        x_velocity, y_velocity = xy_velocity

        scaled_action = self.prev_scaled_action.copy()
        actual_position = self.data.qpos[self.actuated_qpos_indices].copy()

        info = {
            "previous_position": previous_position,
            "current_position": current_position,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "desired_position": scaled_action,
            "actual_position": actual_position,
            "tracking_error": np.mean(np.abs(scaled_action - actual_position)),
        }
        return info