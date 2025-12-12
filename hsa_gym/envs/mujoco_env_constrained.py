"""
Base Module for custom Reinforcement Learning environment built with MuJoCo and Gymnasium.

This module provides the core abstract class `CustomMujocoEnv` which handles 
MuJoCo simulation setup, rendering, action space definition, and utility methods 
for handling state, contact forces, and body positions. It is designed to be 
subclassed for specific robotic tasks.
"""
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

# Helper function to expand model path
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
        terrain_type: str = "flat",
        goal_position: list[float] = [1.5, 0.0, 0.1],
        ensure_flat_spawn: bool = True,
    ):
        """
        Initialize the custom MuJoCo environment and its core simulation components.

        This method sets up the MuJoCo model, initializes data structures, defines the 
        action space, and configures rendering and optional features like terrain 
        generation and control smoothing.

        :param model_path: Path to the MuJoCo model XML file.
        :type model_path: str
        :param frame_skip: Number of simulation steps the MuJoCo physics engine advances 
            for every single call to the Gymnasium ``step()`` function.
        :type frame_skip: int
        :param observation_space: The observation space of the environment. If ``None``, 
            it must be set by the subclass after calling ``super().__init__()``.
        :type observation_space: gymnasium.spaces.Space or None
        :param render_mode: The mode to render with. Supported modes are: ``"human"`` (for viewer) 
            or ``"rgb_array"`` (for NumPy array output).
        :type render_mode: str or None
        :param width: Width of the render window/array in pixels. Defaults to :py:obj:`DEFAULT_SIZE`.
        :type width: int
        :param height: Height of the render window/array in pixels.
        :type height: int
        :param camera_id: Camera ID to use for rendering (cannot be used with ``camera_name``).
        :type camera_id: int or None
        :param camera_name: Camera name to use for rendering.
        :type camera_name: str or None
        :param default_camera_config: Configuration dictionary for the rendering camera 
            (e.g., initial position, distance).
        :type default_camera_config: dict[str, float or int] or None
        :param max_geom: Maximum number of geometries to render.
        :type max_geom: int
        :param visual_options: Dictionary of MuJoCo render flag options.
        :type visual_options: dict[int, bool]
        :param actuator_group: List of MuJoCo actuator group IDs to explicitly enable 
            control for in the simulation.
        :type actuator_group: list[int]
        :param action_group: List of actuator group IDs whose control ranges define 
            the dimensionality and limits of the environment's action space.
        :type action_group: list[int]
        :param smooth_positions: Whether to linearly interpolate the control targets 
            over the ``frame_skip`` steps (True) or apply the target instantly (False).
        :type smooth_positions: bool
        :param enable_terrain: Whether to enable procedural terrain generation upon model initialization.
        :type enable_terrain: bool
        :param terrain_type: The type of procedural terrain to generate (e.g., ``"craters"``, ``"spiral"``).
        :type terrain_type: str
        :param goal_position: The fixed 3D Cartesian position $[x, y, z]$ of the goal site/marker.
        :type goal_position: list[float]
        :param ensure_flat_spawn: Whether to flatten the terrain grid around the robot's initial spawn point.
        :type ensure_flat_spawn: bool
        """

        self.fullpath = expand_model_path(model_path)

        self.width = width
        self.height = height

        # May use width and height
        self.model, self.data = self.initialize_simulation(actuator_group)
        self.update_goal_marker(goal_position=goal_position)

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
        self.set_action_space(action_group=action_group,
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

        self.smooth_positions = smooth_positions
        self.enable_terrain = enable_terrain
        self.terrain_type = terrain_type

        self.unnorm_action_space_bounds = np.column_stack([
            self.action_unnorm_low.copy(),
            self.action_unnorm_high.copy(),
        ]).astype(np.float32)

        self.actuator_ctrlrange = np.column_stack([
            self.actuator_low.copy(),
            self.actuator_high.copy(),
        ]).astype(np.float32)

        self.ensure_flat_spawn = ensure_flat_spawn

    def get_range_bounds(self, bound_group: list[int] = [1]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Extract the control range bounds from the MuJoCo model for specified actuator groups. It assumes 
        a consistent grouping/indexing structure (8 actuators per group).

        :param bound_group: List of actuator/action group to get bounds for
        :type bound_group: list[int]
        :return: A tuple containing the lower and upper bounds for the specified groups.
        :rtype: tuple[NDArray[np.float32], NDArray[np.float32]]
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


    def set_action_space(self, action_group: list[int] = [1], actuator_group: list[int] = [1]) -> spaces.Box:
        """
        Define the normalized Gymnasium action space and store unnormalized actuator bounds.

        :param action_group: List of actuator group IDs whose control ranges define the 
            dimensionality and limits of the unnormalized action space.
        :type action_group: list[int]
        :param actuator_group: List of actuator group IDs that are enabled and whose control 
            limits should be stored internally for simulation control.
        :type actuator_group: list[int]
        :returns: The defined Gymnasium action space object (normalized Box).
        :rtype: spaces.Box
        """
        self.action_unnorm_low, self.action_unnorm_high = self.get_range_bounds(action_group)

        # Action space for actuators
        self.action_space = spaces.Box(low=-1.0, 
                                       high=1.0,
                                       shape=(self.action_unnorm_low.shape[0],),
                                       dtype=np.float32)
        
        self.actuator_low, self.actuator_high = self.get_range_bounds(actuator_group)

        return self.action_space        
    

    def initialize_simulation(self, actuator_group: list[int] = [1]) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """
        Initialize the MuJoCo simulation model and data structures.

        This method handles several core setup tasks:
        
        1. Loads the MuJoCo model from the internal path (``self.fullpath``).
        2. Configures the default off-screen rendering dimensions (width/height).
        3. **Terrain Generation:** Optionally generates and applies procedural terrain 
           data to the model's heightfield if ``self.enable_terrain`` is true.
        4. **Actuator Enabling:** Iterates over ``actuator_group`` to explicitly enable 
           control for the specified groups by clearing the MuJoCo disable flag.
        5. Computes the initial forward dynamics via ``mujoco.mj_forward``.

        :param actuator_group: List of MuJoCo actuator group IDs to enable for control.
        :type actuator_group: list[int]
        :returns: A tuple containing the initialized MuJoCo model (``MjModel``) and 
            data (``MjData``) objects.
        :rtype: tuple[mujoco.MjModel, mujoco.MjData]
        """
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        if self.enable_terrain:
            terrain_data = utils.generate_terrain(
                terrain_type=self.terrain_type,
                width=model.hfield_nrow[0],
                height=model.hfield_ncol[0],
                ensure_flat_spawn=self.ensure_flat_spawn)
            # Add terrain to the model
            model.hfield_data[:] = terrain_data.flatten()
        
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
      
        # Enable only specified actuator groups by clearing the disable bitmask
        for i in range(len(actuator_group)):
            model.opt.disableactuator &= ~(1 << actuator_group[i])
        return model, data

    def update_goal_marker(self, goal_position: list[float] = [1.5, 0.0, 0.1], marker_name: str = "goal") -> None:
        """
        Update the Cartesian position of a specified MuJoCo site/marker in the model.

        :param goal_position: The desired $[x, y, z]$ coordinates (in meters) for the marker.
        :type goal_position: list[float]
        :param marker_name: The name of the MuJoCo site (marker) to update, as defined in the XML model.
        :type marker_name: str
        :returns: None
        :rtype: None
        """
        marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, marker_name)
        if marker_id == -1:
            raise ValueError(f"Marker '{marker_name}' not found in the model.")
        
        self.model.site_pos[marker_id] = np.array(goal_position, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)


    def set_state(self, qpos: NDArray[np.float64], qvel: NDArray[np.float64]):
        """
        Set the joint positions (qpos) and joint velocities (qvel) of the MuJoCo model's state.

        :param qpos: The full position state vector (dimension :py:attr:`self.model.nq`).
        :type qpos: NDArray[np.float64]
        :param qvel: The full velocity state vector (dimension :py:attr:`self.model.nv`).
        :type qvel: NDArray[np.float64]
        :returns: None
        :rtype: None
        """
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)


    def step_mujoco_simulation(self, action: NDArray[np.float32], frame_skip: int = 4, 
                               actuator_group: list[int] = [1]) -> None:
        """
        Advance the MuJoCo simulation by ``frame_skip`` steps with control inputs.

        Calculates the new control target (``new_targets``) by adding 
        the input ``action`` (which is interpreted as an **increment** $\Delta u$) to 
        the current control target (``self.data.ctrl``). If :py:attr:`self._smooth_positions` is True, the control target is linearly interpolated over the ``frame_skip`` simulation steps.
        
        :param action: The control input **increment** ($\Delta u$) for the active actuators.
        :type action: NDArray[np.float32]
        :param frame_skip: Number of simulation steps (frames) to advance per call.
        :type frame_skip: int
        :param actuator_group: List of actuator group IDs whose control targets will be updated.
        :type actuator_group: list[int]
        :returns: None
        :rtype: None
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

        if self.smooth_positions:
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

        :returns: The rendered frame as a NumPy array (if ``render_mode='human'`` or ``render_mode='rgb_array'``) or 
            ``None`` (if rendering is disabled).
        :rtype: NDArray[np.uint8] or None
        """
        return self.mujoco_renderer.render(self.render_mode)

    def close(self) -> None:
        """
        Close rendering contexts and release associated resources.

        :returns: None
        :rtype: None
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
    
    def get_body_com(self, body_name: str) -> NDArray[np.float64]:
        """
        Get the Cartesian position (Center of Mass) of a specified body frame.

        The position is returned in world coordinates and is accessed via the 
        :py:attr:`self.data.body` accessor.

        :param body_name: Name of the body (e.g., 'torso') as defined in the XML model.
        :type body_name: str
        :returns: The 3D Cartesian position $[x, y, z]$ of the body's CoM in the world frame.
        :rtype: NDArray[np.float64]
        """
        return self.data.body(body_name).xpos

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[NDArray[np.float64], dict]:
        """
        Reset the environment to an initial state and return an initial observation.
        This method follows the standard Gymnasium ``reset`` signature. 
        
        :param seed: Optional seed for the environment's random number generator.
        :type seed: int or None
        :param options: Optional dictionary of additional options for resetting the environment.
        :type options: dict or None
        :returns: A tuple containing the initial observation (state after reset) and an info dictionary.
        :rtype: tuple[NDArray[np.float64], dict]
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = self.get_reset_info()

        if self.render_mode == "human":
            self.render()
        return ob, info

    @property
    def dt(self) -> float:
        """
        Return the effective time step (duration) of one environment step.

        :returns: The effective simulation time step in seconds.
        :rtype: float
        """
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, scaled_action: NDArray[np.float32], frame_skip: int = 4, 
                      actuator_group: list[int] = [1]) -> None:
        """
        Advance the MuJoCo simulation using an  unnormalized control input.

        :param scaled_action: The scaled control input (increment) for the active actuators.
        :type scaled_action: NDArray[np.float32]
        :param frame_skip: Number of simulation frames to step..
        :type frame_skip: int
        :param actuator_group: List of actuator group IDs to apply control to.
        :type actuator_group: list[int]
        :returns: None
        :rtype: None
        """
        self.step_mujoco_simulation(scaled_action, frame_skip, actuator_group)

    def state_vector(self) -> NDArray[np.float64]:
        """
        Return the full state vector of the MuJoCo model.
        
        :returns: The concatenated position and velocity state vector.
        :rtype: NDArray[np.float64]
        """
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    
    def get_contact_force(self,
                          geom_name: str = "cylinder3a_con",
                          max_normal_force: float = 30.0
                        ) -> float:
        """
        Calculate the excessive contact force experienced by a specified geometry.

        :param geom_name: Name of the MuJoCo geometry (``geom``) to check for contact force.
        :type geom_name: str
        :param max_normal_force: The maximum allowable contact force magnitude before it is 
            considered "excessive" 
        :type max_normal_force: float
        :returns: The excessive force magnitude, defined as $\max(0.0, ||\mathbf{F}|| - \text{max\_normal\_force})$.
        :rtype: float
        """
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        # Get contact force magnitudes
        force = np.linalg.norm(self.data.cfrc_ext[geom_id, :3])

        excessive = max(0.0, force - max_normal_force)
        return excessive

    # methods to override:
    def step(self,action: NDArray[np.float32]
             ) -> tuple[NDArray[np.float64], np.float64, bool, bool,
                       dict[str, np.float64]]:
        """
        Advance the environment by one timestep using the provided action.

        :param action: The normalized action vector supplied by the agent, typically in the range $[-1.0, 1.0]$.
        :type action: NDArray[np.float32]
        :returns: A tuple containing the next observation, the step reward, 
            a terminated flag, a truncated flag, and an info dictionary.
        :rtype: tuple[NDArray[np.float64], float, bool, bool, dict[str, np.float64]]
        """
        raise NotImplementedError
    
    def reset_model(self) -> NDArray[np.float64]:
        """
        Set the task-specific initial state of the robot and return the initial observation.

        :returns: The initial observation state of the environment.
        :rtype: NDArray[np.float64]
        """
        raise NotImplementedError
    
    def get_reset_info(self) -> dict[str, np.float64]:
        """
        Generate the initial ``info`` dictionary returned during a :py:meth:`~CustomMujocoEnv.reset`.

        This helper method must be implemented by child environments to provide 
        any task-specific information that is available immediately after the 
        environment is reset (e.g., initial goal distance, initial velocity).

        :returns: A dictionary containing initial state information.
        :rtype: dict[str, np.float64]
        """
        return {}