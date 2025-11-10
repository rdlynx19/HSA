"""
HSA Env Class for Torque Control. Action space is positions for the actuators.
"""
import numpy as np
from numpy.typing import NDArray

from .mujoco_env_v3 import CustomMujocoEnv
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
                 forward_reward_weight: float = 1.0,
                 ctrl_cost_weight: float = 1e-3,
                 contact_force_range: tuple[float, float] = (-1.0, 1.0),
                 contact_cost_weight: float = 1e-5,
                 yvel_cost_weight: float = 0.5e-3,
                 actuator_groups: list[int] = [0],
                 action_group: list[int] = [1],
                 clip_actions: float = 1.0,
                 normalize_actions: bool = True,
                 use_pd_control: bool = True,
                 pos_gain: float = 40.0,
                 vel_gain: float = 0.05,
                 max_velocity: float = 35.0,
                 **kwargs):

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_force_range = contact_force_range
        self._contact_cost_weight = contact_cost_weight
        self._yvel_cost_weight = yvel_cost_weight
        self._actuator_groups = actuator_groups
        self._clip_actions = clip_actions

        CustomMujocoEnv.__init__(self,
                            xml_file,
                            frame_skip,
                            observation_space=None,
                            default_camera_config=default_camera_config,
                            actuator_groups=actuator_groups,
                            action_group=action_group,
                            normalize_actions=normalize_actions,
                            use_pd_control=use_pd_control,
                            pos_gain=pos_gain,
                            vel_gain=vel_gain,
                            max_velocity=max_velocity,
                           **kwargs)
        
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.dt))
        }

        # Store the joint qpos indices
        self.actuated_qpos_indices = [7, 22, 9, 23, 20, 8, 21, 10]

        # Observation Size
        observation_size = (
            len(self.actuated_qpos_indices) # qpos
            + len(self.actuated_qpos_indices)  # qvel
            + 2  # XY com position of robot
            + 3  # XYZ velocity of robot
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
            "base_velocity": 3,
        }
  
        # Previous action for smoothing reward calculation
        self.prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        self.prev_scaled_action = np.zeros(self.action_space.shape[0],
                                          dtype=np.float32)
      


    def _scale_action(self, 
                      norm_action: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Scale the normalized action the actuator control range

        :param norm_action: Normalized action in [-1, 1]
        :return: Scaled action within actuator control range
        """
        # Clip to ensure action is within bounds
        clipped_action = np.clip(norm_action, 
                                 -self._clip_actions, 
                                 self._clip_actions)
        
        action_min = self.unnorm_action_space_bounds[:, 0]
        action_max = self.unnorm_action_space_bounds[:, 1]

        # Map [-1, 1] to [action_min, action_max]
        # Formula: scaled = action_min + ( (norm_action + 1) / 2 ) * (action_max - action_min)
        # this maps action = -1 -> action_min, action = 1 -> action_max

        # Scale normalized action to unnormalized action space
        scaled_action = action_min + ((clipped_action + 1.0) / 2.0) * (action_max - action_min)

        # Final clipping to action limits
        scaled_action = np.clip(scaled_action, action_min, action_max)
        return scaled_action


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
        action_diff = action - self.prev_action
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action_diff))
        self.prev_action = action.copy()

        return control_cost
    
    def contact_forces(self) -> NDArray[np.float64]:
        """
        Get the contact forces in the environment.

        :return: Clipped contact forces
        """
        raw_contact_forces = self.data.cfrc_ext.copy()
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces
    
    def contact_cost(self) -> float:
        """
        Compute the contact cost based on contact forces.

        :return: Contact cost
        """
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces())
        )
        return contact_cost

    def step(self, 
             action: NDArray[np.float32]
             ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        """
        Take a step in the environment using the provided action.

        :param action: Action dictionary containing motor commands, normalized in [-1, 1] 
        :return: A tuple containing the observation, reward, termination status, truncation status, and info dictionary
        """
        # Scale action to unnormalized action space
        scaled_action = self._scale_action(action)

        previous_position = self._compute_COM()
        self.do_simulation(scaled_action, 
                           self.prev_scaled_action, 
                           self.frame_skip, 
                           self.actuator_groups)
        self.prev_scaled_action = scaled_action.copy()
        current_position = self._compute_COM()

        # Calculate velocity
        xy_velocity = (current_position - previous_position) / self.dt
        x_velocity, y_velocity = xy_velocity
        
        observation = self._get_obs()
        reward, reward_info = self._get_reward(action, x_velocity, y_velocity)
        terminated = ((self.get_body_com("block_a")[2] > 0.4) or 
                      (self.get_body_com("block_b")[2] > 0.4) or
                      (np.isnan(observation).any()) or
                       (np.isinf(observation).any()))

        actual_pos = self.data.qpos[self.actuated_qpos_indices].copy()
        # Actual torque is for first 8 actuators only
        actual_trq = self.data.ctrl[0:8].copy()

        truncated = False
        info = {
            "previous_position": previous_position,
            "current_position": current_position,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "desired_position": scaled_action,
            "actual_position": actual_pos,
            "tracking_error": np.mean(np.abs(scaled_action - actual_pos)),
            "applied_torque": actual_trq,
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

        :param action: Action dictionary containing motor commands in [-1, 1]
        :return: A tuple containing the reward and a dictionary of reward components
        """
        # Reward is based on velocity in x direction
        forward_reward = self._forward_reward_weight * x_velocity

        # Control cost penalty
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost()
        yvel_cost = self._yvel_cost_weight * np.square(y_velocity)

        costs = ctrl_cost + contact_cost + yvel_cost
        reward = forward_reward - costs
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl_cost": -ctrl_cost,
            "reward_contact_cost": -contact_cost,
            "reward_yvel_cost": -yvel_cost,
        }
    
        return reward, reward_info

    def _get_obs(self) -> NDArray[np.float64]:
        """
        Get the current observation from the environment.

        :return: Observation as a numpy array
        """
        pos = self.data.qpos[self.actuated_qpos_indices].copy()
        vel = self.data.qvel[self.actuated_qpos_indices].copy()

        # Current position of the robot's COM
        current_position = self._compute_COM().flatten()

        # Get base velocity (average of both blocks)
        blocka_vel = self.data.qvel[0:3].copy()
        blockb_vel = self.data.qvel[13:16].copy()
        avg_vel = 0.5 * (blocka_vel + blockb_vel)

        observation = np.concatenate([pos, # 8 values
                                      vel, # 8 values
                                      current_position, # 2 values
                                      avg_vel]).ravel()
        return observation

    def reset_model(self) -> tuple[NDArray[np.float64], dict[str, np.float64]]:
        """
        Reset the model to its initial state.

        :return: Initial observation after reset
        """
        self.set_state(self.init_qpos, self.init_qvel)
        
        # Initialize previous action at reset
        self.prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        self.prev_scaled_action = np.zeros(self.action_space.shape[0],
                                          dtype=np.float32)

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

        scaled_action = self.prev_scaled_action.copy()
        
        actual_pos = self.data.qpos[self.actuated_qpos_indices].copy()
        # Actual torque is for first 8 actuators only
        actual_trq = self.data.ctrl[0:8].copy()
        info = {
            "previous_position": previous_position,
            "current_position": current_position,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "desired_position": scaled_action,
            "actual_position": actual_pos,
            "tracking_error": np.mean(np.abs(scaled_action - actual_pos)),
            "applied_torque": actual_trq, 
        }
        return info