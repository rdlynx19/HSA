from os import path

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space

from ..utils import terrain_generators as utils

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e

DEFAULT_SIZE = 1920 # Default rendering size

def expand_model_path(model_path: str) -> str:
    """Expand the `model path` to a full path if it starts with '~' or '.' or '/'."""
    if model_path.startswith(".") or model_path.startswith("/"):
        fullpath = model_path
    elif model_path.startswith("~"):
        fullpath = path.expanduser(model_path)
    else:
        fullpath = path.join(path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise OSError(f"File {fullpath} does not exist")
    return fullpath

class CustomMujocoEnv(gym.Env):
    """
    Custom MuJoCo Environment base class.
    
    All member variables defined in this class should start with an underscore
    """

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space | None = None,
        render_mode: str | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: int | None = None,
        camera_name: str | None = None,
        default_camera_config: dict[str, float | int] | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] = {},
        actuator_group: list[int] = [1],
        action_group: list[int] = [1],
        smooth_positions: bool = True,
        enable_terrain: bool = False,
        terrain_type: str = "craters",
        goal_position: list[float] = [1.5, 0.0, 0.1]
    ):
        """
        Base abstract class for MuJoCo based environments.

        :param model_path: Path to the MuJoCo model XML file.
        :param frame_skip: Number of simulation steps per gym `step()`
        :param observation_space: Observation space of the environment
        :param render_mode: The mode to render with. Supported modes are: "human", "rgb_array"
        :param width: Width of the render window
        :param height: Height of the render window
        :param camera_id: Camera ID to use for rendering
        :param camera_name: Camera name to use for rendering (cannot be used in conjunction with `camera_id`)
        :param default_camera_config: Configuration for rendering camera
        :param max_geom: Maximum number of rendered geometries
        :param visual_options: render flag options
        :param actuator_group: List of actuator group to enable
        :param action_group: List of actuator group to include in action space
        :param smooth_positions: Whether to smooth actuator position changes over frames
        :param enable_terrain: Whether to enable terrain generation
        :param terrain_type: Type of terrain to generate
        """

        self.fullpath = expand_model_path(model_path)

        self.width = width
        self.height = height

        # May use width and height
        self.model, self.data = self._initialize_simulation(actuator_group)
        self._update_goal_marker(goal_position=goal_position)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        if observation_space is not None:
            self.observation_space = observation_space
        # Initialize action space
        self._set_action_space(action_group=action_group,
                               actuator_group=actuator_group)

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            camera_name,
            visual_options,
        )

        self._smooth_positions = smooth_positions
        self.enable_terrain = enable_terrain
        self._terrain_type = terrain_type

        self.unnorm_action_space_bounds = np.column_stack([
            self._action_unnorm_low.copy(),
            self._action_unnorm_high.copy(),
        ]).astype(np.float32)

        self.actuator_ctrlrange = np.column_stack([
            self._actuator_low.copy(),
            self._actuator_high.copy(),
        ]).astype(np.float32)

    def _get_range_bounds(self, 
                          bound_group: list[int] = [1]
                          ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get the range bounds for the specified action/actuator group.

        :param bound_group: List of actuator/action group to get bounds for
        :return: A tuple containing the lower and upper bounds
        """
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)

        group_low = []
        group_high = []
        for group in bound_group:
            start = group * 8
            end = start + 8
            group_low.append(bounds[start:end, 0])
            group_high.append(bounds[start:end, 1])

        bound_low = np.concatenate(group_low).astype(np.float32)
        bound_high = np.concatenate(group_high).astype(np.float32)

        return bound_low, bound_high


    def _set_action_space(self, 
                          action_group: list[int] = [1],
                          actuator_group: list[int] = [1]) -> spaces.Box:
        """
        Set the action space of the environment.
        """
        self._action_unnorm_low, self._action_unnorm_high = self._get_range_bounds(action_group)

        # Action space for actuators
        self.action_space = spaces.Box(low=-1.0, 
                                       high=1.0,
                                       shape=(self._action_unnorm_low.shape[0],),
                                       dtype=np.float32)
        
        self._actuator_low, self._actuator_high = self._get_range_bounds(actuator_group)

        return self.action_space        
    

    def _initialize_simulation(self, 
                               actuator_group: list[int] = [1],
                               ) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`. 

        :return: A tuple containing the MuJoCo model and data
        """
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        if self._enable_terrain:
            terrain_data = utils.generate_terrain(
                terrain_type=self._terrain_type,
                width=model.hfield_nrow[0],
                height=model.hfield_ncol[0],
                ensure_flat_spawn=False)
            # Add terrain to the model
            model.hfield_data[:] = terrain_data.flatten()
        
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
      
        # Enable only specified actuator groups
        for i in range(len(actuator_group)):
            model.opt.disableactuator &= ~(1 << actuator_group[i])
        return model, data

    def _update_goal_marker(self, 
                            goal_position: list[float] = [1.5, 0.0, 0.1],
                            marker_name: str = "goal"
                            ) -> None:
        """
        Update the position of the goal marker in the simulation.

        :param goal_position: Desired position of the goal marker
        :param marker_name: Name of the marker in the MuJoCo model
        """
        marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, marker_name)
        if marker_id == -1:
            raise ValueError(f"Marker '{marker_name}' not found in the model.")
        
        self.model.body_pos[marker_id] = np.array(goal_position, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)


    def set_state(self, qpos: NDArray[np.float64], qvel: NDArray[np.float64]):
        """
        Set the joints position qpos and velocity qvel of the model.

        :param qpos: Joint position states
        :param qvel: Joint velocity states
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    
    def _step_mujoco_simulation(self, 
                                action: NDArray[np.float32],
                                prev_action: NDArray[np.float32],
                                frame_skip: int = 4,
                                actuator_group: list[int] = [1]) -> None:
        """
        Step the MuJoCo simulation forward by `n_frames` steps using the provided control inputs and constraints.

        :param action: Control inputs for the actuators
        :param prev_action: Previous control inputs for smoothing (does not seem to be used)
        :param frame_skip: Number of simulation frames to step
        :param actuator_group: List of actuator group to control
        """
        # Compute indices of active actuators
        actuator = []
        for group in actuator_group:
            start = group * 8
            actuator.extend(range(start, start + 8))

        # Get current targets for active actuators
        current_targets = self.data.ctrl[actuator].copy()

        # Action represents the increment, so add it to current targets
        new_targets = current_targets + action

        if self._smooth_positions:
            for i in range(frame_skip):
                alpha = (i + 1) / frame_skip
                inter_targets = current_targets + alpha * (new_targets - current_targets)
                self.data.ctrl[actuator] = inter_targets
                mujoco.mj_step(self.model, self.data)
        else:
            self.data.ctrl[actuator] = new_targets
            for _ in range(frame_skip):
                mujoco.mj_step(self.model, self.data)

        mujoco.mj_rnePostConstraint(self.model, self.data)
        
    def render(self) -> NDArray[np.uint8] | None:
        """
        Render a frame from the MuJoCo simulation as specified by the render mode.

        :return: Rendered frame
        """
        return self.mujoco_renderer.render(self.render_mode)

    def close(self) -> None:
        """
        Close rendering contexts processes.
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
    
    def get_body_com(self, body_name: str) -> NDArray[np.float64]:
        """
        Get the cartesian position of a body frame.

        :param body_name: Name of the body
        :return: Cartesian position of the body's COM
        """
        return self.data.body(body_name).xpos

    def reset(self, *, 
              seed: int | None = None, 
              options: dict | None = None) -> tuple[NDArray[np.float64], dict]:
        """
        Reset the environment to an initial state and return an initial observation.

        :param seed: Optional seed for the environment's random number generator
        :param options: Optional dictionary of additional options for resetting the environment
        :return: A tuple containing the initial observation and an info dictionary
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    @property
    def dt(self) -> float:
        """
        Return the time step of the simulation.

        :return: Time step of the simulation
        """
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, 
                      scaled_action: NDArray[np.float32], 
                      prev_scaled_action: NDArray[np.float32],
                      frame_skip: int = 4,
                      actuator_group: list[int] = [1]) -> None:
        """
        Step the MuJoCo simulation forward by `frame_skip` steps using the provided control inputs.

        :param scaled_action: Control inputs for the actuators (scaled)
        :param prev_scaled_action: Previous control inputs for smoothing
        :param frame_skip: Number of simulation frames to step
        :param actuator_group: List of actuator group to control
        """
        # alpha = 0.2
        # smoothed_action = prev_action + alpha * (action - prev_action)
        # smoothed_action = np.clip(smoothed_action, 
        #                               self.action_space.low, 
        #                               self.action_space.high)
        self._step_mujoco_simulation(scaled_action, 
                                     prev_scaled_action, 
                                     frame_skip, 
                                     actuator_group)

    def state_vector(self) -> NDArray[np.float64]:
        """
        Return the position and velocity joint states of the model. 

        :return: Concatenated position and velocity states
        """
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    
    def get_contact_force(self,
                          geom_name: str = "cylinder3a_con",
                          max_normal_force: float = 30.0
                        ) -> float:
        """
        Get the contact force on a specified geometry.
        :param geom_name: Name of the geometry
        :return: Excessive contact force
        """
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        # Get contact force magnitudes
        force = np.linalg.norm(self.data.cfrc_ext[geom_id, :3])

        excessive = max(0.0, force - max_normal_force)
        return excessive

    # methods to override:
    def step(
            self,
            action: NDArray[np.float32]
            ) -> tuple[NDArray[np.float64], np.float64, bool, bool,
                       dict[str, np.float64]]:
        raise NotImplementedError
    
    def reset_model(self) -> NDArray[np.float64]:
        """
        Reset the robot degrees of freedom (qpos and qvel)
        Implement this in each environment subclass
        """
        raise NotImplementedError
    
    def _get_reset_info(self) -> dict[str, np.float64]:
        """
        Function that generates the `info` that is return during a 
        `reset()`
        """
        return {}