import numpy as np
from numpy.typing import NDArray

from gymnasium import utils
from .mujoco_env_position import CustomMujocoEnv
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
                 actuator_group: list[int] = [1],
                 action_group: list[int] = [1],
                 smooth_positions: bool = True,
                 clip_actions: float = 1.0,
                 contact_cost_weight: float = 1e-4,
                 **kwargs):

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._actuator_group = actuator_group
        self._clip_actions = clip_actions

        CustomMujocoEnv.__init__(self,
                                 xml_file,
                                 frame_skip,
                                 observation_space=None,
                                 default_camera_config=default_camera_config,
                                 actuator_group=actuator_group,
                                 action_group=action_group,
                                 smooth_positions=smooth_positions,
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
            "com_velocity": 3
        }
  
        # Previous action for smoothing reward calculation
        self._prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        self._prev_scaled_action = np.zeros(self.action_space.shape[0], 
                                          dtype=np.float32)

    def _scale_action(self, 
                      norm_action: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Scale the normalized action the actuator control range.
        Default position is always 0 for all actuators.

        :param norm_action: Normalized action in [0, 1]
        :return: Scaled action within actuator control range
        """
        # Clip to ensure action is within bounds
        clipped_action = np.clip(norm_action, 0.0, self._clip_actions)

        action_min = self.unnorm_action_space_bounds[:, 0]
        action_max = self.unnorm_action_space_bounds[:, 1]

        scaled_action = np.zeros_like(clipped_action)

        # Scale normalized action to unnormalized action space
        for i in range(len(clipped_action)):
            if action_max[i] > 0:
                # Range[0, 3.14]: scale action [0, 1] to [0, 3.14]
                scaled_action[i] = (clipped_action[i]) * action_max[i]
            else:
                # Range[-3.14, 0]: scale action [0, 1] to [0, -3.14]
                scaled_action[i] = clipped_action[i] * action_min[i]

        return scaled_action

    def contact_cost(self) -> float:
        """
        Penalize excessive contact forces.
        """
        foot_geom_names = ["cylinder3a_con", "cylinder3c_con",
                           "cylinder4a_con", "cylinder4c_con"]
        total_excessive_force = 0.0
        force_threshold = 30.0  # N Threshold for excessive force
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

    def step(self, 
             action: NDArray[np.float32]
             ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        """
        Take a step in the environment using the provided action.

        :param action: Action dictionary containing motor commands 
        :return: A tuple containing the observation, reward, termination status, truncation status, and info dictionary
        """
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
        
        observation = self._get_obs()
        reward, reward_info = self._get_reward(action, x_velocity)
        terminated = ((self.get_body_com("block_a")[2] > 0.4) or 
                      (self.get_body_com("block_b")[2] > 0.4) or
                      (np.isnan(observation).any()) or
                       (np.isinf(observation).any()))
        truncated = False

        actual_position = self.data.qpos[self._actuated_qpos_indices].copy()

        info = {
            "previous_position": previous_position,
            "current_position": current_position,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "desired_position": scaled_action,
            "actual_position": actual_position,
            "tracking_error": np.mean(np.abs(scaled_action - actual_position)),
            **reward_info
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, truncated, info

    def _get_reward(self, 
                    action: NDArray[np.float32],
                    x_velocity: float = 0.0,
                    ) -> tuple[float, dict[str, float]]:
        """
        Compute the reward for the current step.

        :param action: Action dictionary containing motor commands 
        :return: A tuple containing the reward and a dictionary of reward components
        """
        # Reward is based on velocity in x direction
        forward_reward = self._forward_reward_weight * x_velocity

        # Control cost penalty
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl_cost": -ctrl_cost,
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

        observation = np.concatenate([pos, 
                                      vel, 
                                      current_position, 
                                      avg_velocity]).ravel()
        return observation

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the model to its initial state.

        :return: Initial observation after reset
        """
        self.set_state(self.init_qpos, self.init_qvel)
        
        # Initialize previous action at reset
        self._prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)
        
        self._prev_scaled_action = np.zeros(self.action_space.shape[0], 
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