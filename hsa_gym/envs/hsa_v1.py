import numpy as np
from numpy.typing import NDArray

from gymnasium import utils
from .mujoco_env import CustomMujocoEnv
from gymnasium.spaces import Box, Space

class HSAEnv(CustomMujocoEnv, utils.EzPickle):
    """
    HSA Environment Class for MuJoCo-based simulation.
    In this environment, a robot must learn to move towards a goal position
    in the XY plane by coordinating its two blocks.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, 
                 xml_file: str = "hsaModel.xml",
                 frame_skip: int = 4,
                 default_camera_config: dict[str, float | int] = {},
                 forward_reward_weight: float = 1.0,
                 ctrl_cost_weight: float = 1e-4,
                 randomize_goal: bool = False,
                 actuator_groups: list[int] = [1],
                 use_locks: bool = False,
                 **kwargs):
        
        utils.EzPickle.__init__(self,
                                xml_file,
                                frame_skip,
                                default_camera_config,
                                forward_reward_weight,
                                ctrl_cost_weight,
                                **kwargs)
        
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._randomize_goal = randomize_goal
        # Goal bounds in the XY plane
        self.goal_bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])
        # Initializing goal position
        self.goal = np.zeros(2)

        self.actuator_groups = actuator_groups

        CustomMujocoEnv.__init__(self,
                           xml_file,
                           frame_skip,
                           observation_space=None,
                           default_camera_config=default_camera_config,
                           actuator_groups=actuator_groups,
                           **kwargs)
        
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.dt))
        }

        # Observation Size
        observation_size = (
            self.data.qpos.size
            + self.data.qacc.size
            + 1  # for distance from goal position
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
            "qvel": self.data.qacc.size,
            "goal_distance": 1
        }
  

    # Control cost to penalize large actions
    def control_cost(self, 
                     action: dict[str, NDArray[np.float32 | np.uint8]]
                     ) -> float:
        """
        Compute the control cost based on the action taken.

        :param action: Action dictionary containing motor commands
        :return: Control cost as a float
        """
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action["motors"]))
        return control_cost

    def step(self, action: dict[str, NDArray[np.float32 | np.uint8]]) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        """
        Take a step in the environment using the provided action.

        :param action: Action dictionary containing motor commands and disc lock/unlock signals
        :return: A tuple containing the observation, reward, termination status, truncation status, and info dictionary
        """
        print(action)
        prv_dist = self._get_distance_to_goal()
        self.do_simulation(action, self.frame_skip, self.actuator_groups)
        observation = self._get_obs()
        reward, reward_info = self._get_reward(action)
        terminated = (self._get_distance_to_goal() < 0.05) or (not np.isfinite(observation).all() or np.isnan(observation).any())
        truncated = False
        info = {
            "prev_distance": prv_dist,
            "cur_distance": self._get_distance_to_goal(),
            **reward_info
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, truncated, info

    def _get_reward(self, action: dict[str, NDArray[np.float32 | np.uint8]]) -> tuple[float, dict[str, float]]:
        """
        Compute the reward for the current step.

        :param action: Action dictionary containing motor commands and disc lock/unlock signals
        :return: A tuple containing the reward and a dictionary of reward components
        """
        # Compute current distance to goal
        cur_dist = self._get_distance_to_goal() 
        
        # Reward is based on reduction in distance to goal
        progress_reward = self._forward_reward_weight * (self.prev_dist - cur_dist)

        # Control cost penalty
        ctrl_cost = self.control_cost(action)
        reward = progress_reward - ctrl_cost
        reward_info = {
            "reward_progress": progress_reward,
            "reward_ctrl_cost": -ctrl_cost,
        }

        # Bonmus for reaching close to the goal
        if cur_dist < 0.05:
            reward += 1.0

        # Update previous distance
        self.prev_dist = cur_dist
        return reward, reward_info

    def _get_obs(self) -> NDArray[np.float64]:
        """
        Get the current observation from the environment.

        :return: Observation as a numpy array
        """
        position = self.data.qpos.flatten()
        acc = self.data.qacc.flatten()

        # Distance from goal position
        goal_distance = self._get_distance_to_goal()

        observation = np.concatenate([position, acc, [goal_distance]]).ravel()
        return observation

    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the model to its initial state.

        :return: Initial observation after reset
        """
        self.set_state(self.init_qpos, self.init_qacc)
        
        # Assign a new goal at reset
        self.goal = self._sample_goal()
        self.prev_dist = self._get_distance_to_goal()
        observation = self._get_obs()
        return observation
    
    def _get_distance_to_goal(self) -> float:
        """
        Compute the distance from the robot's center of mass to the goal position.

        :return: Distance to the goal as a float
        """
        robot_COM = self._compute_COM()
        goal_distance = np.linalg.norm(robot_COM - self.goal).astype(np.float64)
        return goal_distance

    def _sample_goal(self) -> NDArray[np.float64]:
        """
        Sample a new goal position.

        :return: New goal position as a numpy array
        """
        if self._randomize_goal:
            low, high = self.goal_bounds[:, 0], self.goal_bounds[:, 1]
            return np.random.uniform(low=low, high=high)
        else:
            return np.array([2.0, 0.0])

    def _compute_COM(self) -> NDArray[np.float64]:
        """
        Compute the center of mass (COM) of the robot in the XY plane.

        :return: Center of mass position as a numpy array
        """

        blocka_pos = self.get_body_com("block_a").copy()
        blockb_pos = self.get_body_com("block_b").copy()

        # Center of Mass position
        return 0.5 * (blocka_pos[:2] + blockb_pos[:2])
    
