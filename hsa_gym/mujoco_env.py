from os import path

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        'MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`'
    ) from e

DEFAULT_SIZE = 480 # Default rendering size

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
    """Custom MuJoCo Environment base class."""

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
        """

        self.fullpath = expand_model_path(model_path)

        self.width = width
        self.height = height

        # May use width and height
        self.model, self.data = self._initialize_simulation()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.frame_skip = frame_skip

        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'
        if observation_space is not None:
            self.observation_space = observation_space
        self._set_action_space()

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

        def _set_action_space(self):
            """
            Set the action space of the environment.
            """
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            low, high = bounds.T
            # Action space for actuators
            continous_space = spaces.Box(low=low, high=high, dtype=np.float32)
            # Action space for disc unlocking/locking
            discrete_space = spaces.MultiBinary(4)

            self.action_space = spaces.Dict({
                "motors": continous_space,
                "constraints": discrete_space
            })
            return self.action_space
        
        def _initialize_simulation(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
            """
            Initialize MuJoCo simulation data structures `mjModel` and `mjData`. 

            :return: A tuple containing the MuJoCo model and data
            """
            model = mujoco.MjModel.from_xml_path(self.fullpath)
            model.vis.global_.offwidth = self.width
            model.vis.global_.offheight = self.height
            data = mujoco.MjData(model)
            return model, data
        
        def _step_mujoco_simulation(self, ctrl: NDArray[np.float64], constraint: NDArray[np.int8], n_frames: int) -> None:
            """
            Step the MuJoCo simulation forward by `n_frames` steps using the provided control inputs and constraints.

            :param ctrl: Control inputs for the actuators
            :param constraint: Constraint activations for the discs
            :param n_frames: Number of simulation frames to step
            """
            self.data.ctrl[:] = ctrl
            self.data.eq_active[:] = constraint
            
            mujoco.mj_step(self.model, self.data, nstep=n_frames)

            mujoco.mj_rnePostConstraint(self.model, self.data)
            
        def render(self) -> MujocoRenderer.render:
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
        
        def reset(self, 
                  *, 
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

        def do_simulation(self, ctrl: NDArray[np.float64], constraint: NDArray[np.int8], n_frames: int) -> None:
            """
            Step the MuJoCo simulation forward by `n_frames` steps using the provided control inputs.

            :param ctrl: Control inputs for the actuators
            :param constraint: Constraint activations for the discs
            :param n_frames: Number of simulation frames to step
            """
            if np.array(ctrl).shape != (self.model.nu,):
                raise ValueError(
                    f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")
            self._step_mujoco_simulation(ctrl, constraint, n_frames)

        def state_vector(self) -> NDArray[np.float64]:
            """
            Return the position and velocity joint states of the model. 

            :return: Concatenated position and velocity states
            """
            return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        
        # methods to override:
        def step(
                self, 
                action: NDArray[np.float64],
            ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
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