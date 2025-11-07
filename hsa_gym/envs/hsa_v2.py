import numpy as np
from numpy.typing import NDArray

from gymnasium import utils
from .mujoco_env_v2 import CustomMujocoEnv
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
                 actuator_groups: list[int] = [1],
                 **kwargs):

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self.actuator_groups = actuator_groups

        CustomMujocoEnv.__init__(self,
                           xml_file,
                           frame_skip,
                           observation_space=None,
                           default_camera_config=default_camera_config,
                           actuator_groups=self.actuator_groups,
                           **kwargs)
        
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.dt))
        }

        # Observation Size
        observation_size = (
            self.data.qpos.size
            + self.data.qvel.size
            + 2  # XY com position of robot
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
        }
  
        # Previous action for smoothing reward calculation
        self.prev_action = np.zeros(self.action_space.shape[0], 
                                    dtype=np.float32)

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

    def step(self, 
             action: NDArray[np.float32]
             ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        """
        Take a step in the environment using the provided action.

        :param action: Action dictionary containing motor commands 
        :return: A tuple containing the observation, reward, termination status, truncation status, and info dictionary
        """
        previous_position = self._compute_COM()
        self.do_simulation(action, self.prev_action, self.frame_skip, self.actuator_groups)
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
        info = {
            "prev_position": previous_position,
            "cur_position": current_position,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
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
        pos = self.data.qpos.flatten()
        vel = self.data.qvel.flatten()

        # Current position of the robot's COM
        current_position = self._compute_COM().flatten()

        observation = np.concatenate([pos, vel, current_position]).ravel()
        return observation

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the model to its initial state.

        :return: Initial observation after reset
        """
        self.set_state(self.init_qpos, self.init_qvel)
        
        # Initialize previous action at reset
        self.prev_action = np.zeros(self.action_space.shape[0], 
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
    
