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
                 xml_file: str = "hsaModel.xml",
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
                 alive_bonus: float = 0.1,
                 max_increment: float = 3.14,
                 enable_terrain: bool = False,
                 terrain_type: str = "craters",
                 goal_position: list[float] = [3.0, 1.0, 0.1],
                 **kwargs):

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._yvel_cost_weight = yvel_cost_weight
        self._constraint_cost_weight = constraint_cost_weight
        self._acc_cost_weight = acc_cost_weight
        self._early_termination_penalty = early_termination_penalty
        self._distance_reward_weight = distance_reward_weight
        self._alive_bonus = alive_bonus 
        self._joint_vel_cost_weight = joint_vel_cost_weight

        self._actuator_group = actuator_group
        self._clip_actions = clip_actions

        self._goal_position = np.array(goal_position, dtype=np.float64)

        self._max_increment = max_increment
        self._enable_terrain = enable_terrain
        self._terrain_type = terrain_type

        self._qvel_limit = 1000.0 # Max joint velocity for termination condition
        self._qacc_limit = 15000.0 # Max joint acceleration for termination

        # Step Count
        self._step_count = 0

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
                                 goal_position=self._goal_position,   
                                 **kwargs)

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.dt))
        }

        # Order of indices
        # self.actuated_qpos_indices = {
        #     "1A": 7,
        #     "2A": 22,
        #     "3A": 10,
        #     "4A": 24,
        #     "1C": 20,
        #     "2C": 9,
        #     "3C": 21,
        #     "4C": 12
        #}
        self._actuated_qpos_indices = [7, 22, 10, 24, 20, 9, 21, 12]
        # self.actuated_qvel_indices = {
            # "1A": 6,
            # "2A": 20,
            # "3A": 9,
            # "4A": 22,
            # "1C": 18,
            # "2C": 8,
            # "3C": 19,
            # "4C": 11
        #}
        self._actuated_qvel_indices = [6, 20, 9, 22, 18, 8, 19, 11]

        # Observation Size
        observation_size = (
            len(self._actuated_qpos_indices)
            + len(self._actuated_qvel_indices)
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
        self._prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        self._prev_scaled_action = np.zeros(self.action_space.shape[0], 
                                          dtype=np.float32)
        
        # Previous velocity for acceleration cost
        self._prev_joint_velocities = np.zeros(
            len(self._actuated_qvel_indices), 
            dtype=np.float32
        )

        self._prev_distance_to_goal = np.linalg.norm(
            self._compute_COM() - np.array(goal_position[:2])
            )
        
    def set_curriculum_manager(self, curriculum_manager):
        """
        Set the curriculum manager for the environment.
        :param curriculum_manager: An instance of CurriculumManager
        """
        self.curriculum_manager = curriculum_manager

    def _get_spawn_height(self, x, y) -> float:
        """
        Get the spawn height for the terrain at the given (x, y) coordinates.
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
        normalized_height = terrain_data[grid_i, grid_j]

           # Convert to actual height
        actual_height = base_height + normalized_height * z_max
        
        return actual_height

    def distance_cost(self, 
                         goal_position: NDArray[np.float64]) -> float:
        """
        Compute the distance reward based on the current position and the goal position.
        """
        current_position = self._compute_COM()
        distance = np.linalg.norm(current_position - goal_position[:2])

        # Progress reward: positive if getting closer, negative if moving away
        distance_progress = self._prev_distance_to_goal - distance

        self._prev_distance_to_goal = distance

        return distance_progress

    def _scale_action(self, 
                      norm_action: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Scale the normalized action the actuator control range.
        Default position is always 0 for all actuators.

        :param norm_action: Normalized action in [-1, 1]
        :return: Scaled action within actuator control range
        """
        scaled_action = norm_action * self._max_increment
        
        return scaled_action
    
    def acceleration_cost(self) -> float:
        """
        Penalize large accelerations (changes in velocity).
        """
        joint_velocities = self.data.qvel[self._actuated_qvel_indices]
    
        # Compute acceleration as change in velocity
        joint_acc = (joint_velocities - self._prev_joint_velocities) / self.dt

        # Clip to prevent explosion
        joint_acc = np.clip(joint_acc, -50.0, 50.0)
        # Acceleration cost
        acc_cost = self._acc_cost_weight * np.sum(np.square(joint_acc))
        # Cap cost
   
        # Update previous velocities
        self._prev_joint_velocities = joint_velocities.copy()

        return acc_cost

    def constraint_cost(self,
                        diff_margin: float = 0.01,
                        penalty_factor: float = 1.0,
                        bonus_factor: float = 0.025) -> float:
        """
        Compute penalty for violating angular difference constraints.

        Penalizes two types of violations:
        1. Joint difference violations: |A - C| > pi
        2. Handedness violations during starting positions?

        :param diff_margin: Start penalizing when withing this margin of pi
        :param handedness_margin: Start penalizing when within this margin of 0
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

            # # Compute angle difference with proper wrapping
            # # this ensures the difference is within [-pi, pi]
            # raw_diff = pos_a - pos_c
            # wrapped_diff = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))

            # # Get absolute difference
            diff = abs(pos_a - pos_c)
            # Now diff is guaranteed to be in [0, pi]
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
            
        # Handedness constraint is missing for now
        # Hoping model learns it by using the difference constraint alone
        return constraint_cost, constraint_bonus

    def contact_cost(self) -> float:
        """
        Penalize excessive contact forces.
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
        
        contact_cost = self._contact_cost_weight * total_excessive_force
        
        return contact_cost


    # Control cost to penalize large actions
    def control_cost(self, 
                     action: NDArray[np.float32]
                     ) -> float:
        """
        Compute the control cost based on the action taken.

        :param action: Action dictionary containing motor commands
        :return: Control cost as a float
        """
        # Compute the difference between current and previous actions
        action_diff = action - self._prev_action
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action_diff))
        self._prev_action = action.copy()

        
        return control_cost
    
    def joint_velocity_cost(self) -> float:
        """
        Penalize high joint velocities.
        """
        joint_velocities = self.data.qvel[self._actuated_qvel_indices]
        joint_velocities = np.clip(joint_velocities, -50.0, 50.0)
        joint_vel_cost = self._joint_vel_cost_weight * np.sum(np.square(joint_velocities))
        return joint_vel_cost
    
    def vec_to_goal(self) -> NDArray[np.float64]:
        """
        Compute the vector from the robot's current position to the goal position.
        """
        current_position = self._compute_COM()
        goal_vector = self._goal_position[:2] - current_position
        distance_to_goal = np.linalg.norm(goal_vector)
        goal_unit_vector = goal_vector / (distance_to_goal + 1e-8)  # Avoid division by zero
        return goal_unit_vector

    def _check_termination(self,
                           observation: NDArray[np.float64]
                           ) -> bool:
        """
        Check if the episode should be terminated and store the reason.
        :return: True if the episode should be terminated, False otherwise
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
        
        if self._enable_terrain:
            blocka_xy = self.get_body_com("block_a")[:2]
            blockb_xy = self.get_body_com("block_b")[:2]

            terrain_a_z = self._get_spawn_height(blocka_xy[0], blocka_xy[1])
            terrain_b_z = self._get_spawn_height(blockb_xy[0], blockb_xy[1])
            # Safe margin above terrain
            safe_margin = 0.5
            if z_a > (terrain_a_z + safe_margin):
                reasons.append("block_a_too_high")
            if z_b > (terrain_b_z + safe_margin):
                reasons.append("block_b_too_high")
            if z_a < (terrain_a_z - safe_margin):
                reasons.append("block_a_too_low")
            if z_b < (terrain_b_z - safe_margin):
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
        if np.max(np.abs(self.data.qvel)) > self._qvel_limit: reasons.append("qvel_limit")
        if np.max(np.abs(self.data.qacc)) > self._qacc_limit: reasons.append("qacc_limit")

        terminated = len(reasons) > 0
        return terminated, reasons

    def step(self, 
             action: NDArray[np.float32]
             ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        """
        Take a step in the environment using the provided action.

        :param action: Action dictionary containing motor commands 
        :return: A tuple containing the observation, reward, termination status, truncation status, and info dictionary
        """
        self._step_count += 1
        # Scale action
        scaled_action = self._scale_action(action)

        previous_position = self._compute_COM()
        self.do_simulation(scaled_action, 
                           self._prev_scaled_action, 
                           self.frame_skip, 
                           self._actuator_group)
        current_position = self._compute_COM()

        # Calculate velocity
        xy_velocity = (current_position - previous_position) / self.dt
        x_velocity, y_velocity = xy_velocity
        x_velocity = np.clip(x_velocity, -5.0, 5.0)
        y_velocity = np.clip(y_velocity, -5.0, 5.0)
        
        observation = self._get_obs()
        reward, reward_info = self._get_reward(action, x_velocity, y_velocity)
        terminated, term_reasons = self._check_termination(observation)

        truncated = False
        # Scale early termination penalty
        if terminated and not truncated:
            steps_lost = 3000 - self._step_count
            early_term_pen = (self._early_termination_penalty) * steps_lost
        else: 
            early_term_pen = 0.0
       
        alive_bonus = (
            self._alive_bonus if not (terminated or truncated) else 0.0
        )

        # Bonus for reaching close to goal
        current_distance = np.linalg.norm(
            self._compute_COM() - self._goal_position[:2]
        )
        if current_distance < 0.20:
            reward += 40.0
            early_term_pen = 0.0  # No penalty if goal is reached
            terminated = True
            term_reasons.append("goal_reached")

        reward = reward + alive_bonus - early_term_pen

        actual_position = self.data.qpos[self._actuated_qpos_indices].copy()

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
            "termination_reasons": term_reasons,
            **reward_info
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, truncated, info

    def _get_reward(self, 
                    action: NDArray[np.float32],
                    x_velocity: float = 0.0,
                    y_velocity: float = 0.0,
                    ) -> tuple[float, dict[str, float]]:
        """
        Compute the reward for the current step.

        :param action: Action dictionary containing motor commands 
        :return: A tuple containing the reward and a dictionary of reward components
        """
        com_velocity = np.array([x_velocity, y_velocity])
        projected_velocity = np.dot(com_velocity, self.vec_to_goal())
        # Reward is based on velocity towards goal
        forward_reward = (self._forward_reward_weight) * max(0.0, projected_velocity)

        # Y velocity reward
        lateral_reward = self._yvel_cost_weight * abs(y_velocity)

        # Control cost penalty
        ctrl_cost = self.control_cost(action)
        ctrl_cost = min(ctrl_cost, 20.0)
        # Contact cost penalty
        contact_cost = self.contact_cost()
        contact_cost = min(contact_cost, 30.0)
        # Constraint cost penalty
        constraint_violation, constraint_bonus = self.constraint_cost()
        constraint_cost = self._constraint_cost_weight * constraint_violation
        constraint_cost = min(constraint_cost, 50.0)
        # Acceleration cost penalty
        acc_cost = self.acceleration_cost()
        acc_cost = min(acc_cost, 20.0)
        # Joint velocity cost penalty
        joint_vel_cost = self.joint_velocity_cost()
        joint_vel_cost = min(joint_vel_cost, 50.0)
        # Distance reward
        distance_reward = (
            self._distance_reward_weight * 
            self.distance_cost(self._goal_position)
        )
        # Reward shaping to encourage getting closer to goal
        distance = np.linalg.norm(
            self._compute_COM() - self._goal_position[:2]
        )
        reward_shaping = 0.5 / (distance + 0.2)
        distance_reward += reward_shaping

        costs = (
            ctrl_cost + contact_cost + constraint_cost + acc_cost + joint_vel_cost
            )
        reward = forward_reward + lateral_reward - costs + distance_reward + constraint_bonus
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
            "reward_total_costs": -costs
        }
    
        return reward, reward_info

    def _get_obs(self) -> NDArray[np.float64]:
        """
        Get the current observation from the environment.

        :return: Observation as a numpy array
        """
        pos = self.data.qpos[self._actuated_qpos_indices].copy()
        vel = self.data.qvel[self._actuated_qvel_indices].copy()

        # Current position of the robot's COM
        current_position = self._compute_COM().flatten()

        # Get base velocity (average of both blocks)
        blocka_vel = self.data.qvel[0:3].copy()
        blockb_vel = self.data.qvel[13:16].copy()
        avg_velocity = 0.5 * (blocka_vel + blockb_vel)

        # Get distance to goal
        distance_to_goal = np.linalg.norm(
            self._compute_COM() - self._goal_position[:2]
        ).reshape(1)
        vec_to_goal = self._goal_position[:2] - self._compute_COM() 
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
        Reset the model to its initial state.

        :return: Initial observation after reset
        """
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        if self._enable_terrain:
            # Get spawn positions for both blocks
            blocka_x, blocka_y = -0.2, 0.0
            blockb_x, blockb_y = 0.2, 0.0
            # Get terrain height at those positions
            terrain_a_z = self._get_spawn_height(blocka_x, blocka_y)
            terrain_b_z = self._get_spawn_height(blockb_x, blockb_y)

            # Use the higher terrain height to ensure both blocks are above ground
            terrain_height = max(terrain_a_z, terrain_b_z)

            robot_clearance = 0.1  # Meters above terrain
            spawn_z = terrain_height + robot_clearance

            qpos[2] = spawn_z
            qpos[15] = spawn_z

        self.set_state(qpos, qvel)
        
        self._step_count = 0
        # Initialize previous action at reset
        self._prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        
        self._prev_scaled_action = np.zeros(self.action_space.shape[0], 
                                          dtype=np.float32)
        
        self._prev_joint_velocities = np.zeros(
            len(self._actuated_qvel_indices), 
            dtype=np.float32
        )
         
        # Curriculum based goal sampling
        if hasattr(self, 'curriculum_manager') and self.curriculum_manager is not None:
            # Use curriculum manager to sample goal
            goal_pos = self.curriculum_manager.sample_goal_position()
            marker_x, marker_y, marker_z = goal_pos

        else:
            ranges = [(-3.0, -1.5), (1.5, 3.0)]
            low, high = ranges[np.random.choice([0, 1])]
            marker_x = np.random.uniform(low, high)
            marker_y = np.random.uniform(-1.0, 1.0)
            marker_z = 0.1

        self._update_goal_marker(goal_position=[marker_x, marker_y, marker_z])
        self._goal_position = np.array([marker_x, marker_y, marker_z], dtype=np.float64)

        self._prev_distance_to_goal = np.linalg.norm(
            self._compute_COM() - self._goal_position[:2]
        )

        observation = self._get_obs()
        return observation
    
    def _compute_COM(self) -> NDArray[np.float64]:
        """
        Compute the center of mass (COM) of the robot in the XY plane.

        :return: Center of mass position as a numpy array
        """

        blocka_pos = self.get_body_com("block_a").copy()
        blockb_pos = self.get_body_com("block_b").copy()

        # Center of Mass position
        return 0.5 * (blocka_pos[:2] + blockb_pos[:2])
    
    def _get_reset_info(self) -> dict[str, np.float64]:
        """
        Get additional info upon environment reset.

        :return: Info dictionary containing initial COM position
        """
        previous_position = self._compute_COM()
        current_position = self._compute_COM()

        xy_velocity = (current_position - previous_position) / self.dt
        x_velocity, y_velocity = xy_velocity

        scaled_action = self._prev_scaled_action.copy()
        actual_position = self.data.qpos[self._actuated_qpos_indices].copy()

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