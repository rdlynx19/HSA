import mujoco
import mujoco.viewer
import numpy as np
from .states import RobotState
import time

class MuJoCoControlInterface:
    """
    Control interface for HSA MuJoco simulations
    """
    def __init__(self, model_path: str = None):
        """
        Initialize the control interface with a MuJoCo model and data.
        """
        if model_path is None:
            raise ValueError("Model and data must be provided.")
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.state = RobotState.IDLE
        self.viewer = None

    def enable_actuator_group(self, group_index: int) -> None:
        """
        Enable a specific actuator group by its index
        :param group_index: Index of the actuator group to enable 
        """
        self.model.opt.disableactuator &= ~(1 << group_index)
       
    def disable_actuator_group(self, group_index: int) -> None:
        """
        Disable a specific actuator group by its index
        :param group_index: Index of the actuator group to disable
        """
        self.model.opt.disableactuator |= (1 << group_index)
    
    def get_joint_positions(self, joint_names: list[str]) -> list[tuple[str, float]]:
        """
        Get the positions of specified joint by their names
        :param joint_names: List of joint names to retrieve positions for

        :return: list of joint names and their positions
        """
        joint_positions = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in the model.")
            position = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            joint_positions.append((name, position))
        return joint_positions
    
    def get_joint_velocities(self, joint_names: list[str]) -> list[tuple[str, float]]:
        """
        Get the velocities of specified actuators by their names
        :param joint_names: List of actuator names to retrieve velocities for

        :return: list of actuator names and their velocities
        """
        joint_velocities = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in the model.")
            velocity = self.data.qvel[self.model.jnt_dofadr[joint_id]]
            joint_velocities.append((name, velocity))
        return joint_velocities

    def get_robot_state(self) -> RobotState:
        """
        Get the current state of the robot
        :return: Current RobotState
        """
        return self.state
    
    def set_robot_state(self, new_state: RobotState) -> None:
        """
        Set the current state of the robot
        :param new_state: New RobotState to set
        """
        self.state = new_state

    def step_simulation(self) -> None: 
        """
        Step the MuJoCo simulation forward by one timestep
        """
        mujoco.mj_step(self.model, self.data)

    def reset_simulation(self) -> None:
        """
        Reset the MuJoCo simulation to its initial state
        """
        mujoco.mj_resetData(self.model, self.data)  

    def launch_viewer(self) -> None:
        """
        Launch a passive viewer for the MuJoCo simulation
        """
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def sync_viewer(self) -> None:
        """
        Sync the viewer with the current simulation state
        """
        if self.viewer is not None:
            self.viewer.sync()

    def velocity_control_drive(self, 
                               actuator_names: list[str] = 
                               ["spring3c_vel", "spring2a_vel",            "spring3a_vel", "spring2c_vel", 
                                "spring4a_vel", "spring1c_vel", "spring4c_vel", "spring1a_vel"], 
                               joint_name: str = "cw_cont_3a_btm", 
                               velocity: float = 3.0) -> None:
        """
        Apply velocity control to specified actuators for a given duration
        """
        self.disable_actuator_group(1)
        self.enable_actuator_group(2)

        actuator_ids = []
        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in the model.")
            actuator_ids.append(actuator_id)

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in the model.")
        
        self.launch_viewer()
        try:
            #  What is a better way to do this?
            self.data.ctrl[:] = 0.0
            
            self.step_simulation()
            self.sync_viewer()
            
            while self.viewer.is_running():
                for i, act_id in enumerate(actuator_ids):
                    if i % 2 == 0:
                        self.data.ctrl[act_id] = velocity
                    

                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.model.opt.timestep)
        finally:
            self.close_simulation()

    def position_control_extension(self,
                                   actuator_names: list[str] = 
                                   ["spring1a_motor", "spring1c_motor",
                                    "spring3a_motor", "spring3c_motor",
                                    "spring2a_motor", "spring2c_motor",
                                    "spring4a_motor", "spring4c_motor"],
                                    joint_name: str = "cw_cont_3a_btm",
                                    position: float = 2.84) -> None:
        """
        Apply position control to obtain an extension motion
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = []
        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in the model.")
            actuator_ids.append(actuator_id)

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in the model.")

        self.launch_viewer()
        try:
            self.data.ctrl[:] = 0.0

            self.step_simulation()
            self.sync_viewer()


            while self.viewer.is_running():                    
                    for i, act_id in enumerate(actuator_ids):
                        if i % 2 == 0:
                            self.data.ctrl[act_id] =  position
                        else:
                            self.data.ctrl[act_id] = -position
                
                    self.step_simulation()
                    self.sync_viewer()
                    time.sleep(self.model.opt.timestep)
        finally:
            self.close_simulation()

        
    def position_control_crawl(self,
                               actuator_names: list[str] = 
                                ["spring1a_motor", "spring1c_motor",
                                "spring3a_motor", "spring3c_motor",
                                "spring2a_motor", "spring2c_motor",
                                "spring4a_motor", "spring4c_motor"],
                                position: float = 2.84,
                                tolerance: float = 0.05) -> None:
        """
        Perform crawling action by repeated contraction and extension
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = []
        actuator_to_joint_ids = {}
        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in the model.")
            actuator_ids.append(actuator_id)
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            print(f"Joint ID: {joint_id}")
            actuator_to_joint_ids[actuator_id] = joint_id

        self.launch_viewer()
        try:
            self.data.ctrl[:] = 0.0
            self.step_simulation()
            self.sync_viewer()

            total_steps = 10
            # Define extension and contraction targets
            start_ctrl = np.zeros(len(actuator_ids))
            target_ctrl = np.array([position if i % 2 == 0 else -position for i in range(len(actuator_ids))])


            while self.viewer.is_running():                    
                    # Extension phase
                    trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration=15, timestep=self.model.opt.timestep, method="quadratic")       
                    for step_values in trajectory:
                        self.data.ctrl[actuator_ids] = step_values
                        self.step_simulation()            
                        self.sync_viewer()    

                    # Hold for a short duration
                    for _ in range(int(0.5 / self.model.opt.timestep)):
                        self.step_simulation()
                        self.sync_viewer()

                    # Contraction phase
                    trajectory = self.interpolate_values(target_ctrl, start_ctrl, 15, self.model.opt.timestep, "quadratic")
                    for step_values in trajectory:
                        self.data.ctrl[actuator_ids] = step_values
                        self.step_simulation()
                        self.sync_viewer()

                    # Hold for short duration
                    for _ in range(int(0.5 / self.model.opt.timestep)):
                        self.step_simulation()
                        self.sync_viewer()
                    # time.sleep(self.model.opt.timestep)
        finally:
            self.close_simulation()

    def interpolate_values(self, start: np.ndarray = None,
                           goal: np.ndarray = None,
                           duration: float = None,
                           timestep: float = None,
                           method: str = "linear") -> np.ndarray:
        """
        Generate interpolated values between start and goal over a given duration

        :param start: Starting value or interpolation lower limit
        :param goal: Goal value or interpolation upper limit
        :param duration: Total transition time
        :param timestep: Simulation time step
        :param method: Interpolation type - linear, cosine, smoothstep, quadratic

        :return: np.ndarray of interpolated values for each time step
        """
        steps = max(2, int(duration/timestep))
        alphas = np.linspace(0, 1, steps) # progress fraction

        if method == "cosine":
            alphas = 0.5 * (1 - np.cos(np.pi * alphas))
        elif method == "smoothstep":
            alphas = alphas * alphas * (3 - 2 * alphas)
        elif method == "quadratic":
            alphas = alphas ** 2
        elif method == "linear":
            pass # Leave alphas unchanged / Use original definition of alphas
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        # Converting inputs to numpy arrays
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)

        interpolated = np.outer(1 - alphas, start) + np.outer(alphas, goal)

        return interpolated


    def view_model(self) -> None:
        """
        View the MuJoCo model in a passive viewer
        """
        self.launch_viewer()
        try:
            self.step_simulation()
            self.sync_viewer()
            
            while self.viewer.is_running():
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.model.opt.timestep)
        finally:
            self.close_simulation()

    def close_simulation(self) -> None:
        """
        Close the MuJoCo simulation cleanly
        """
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None 

        if  hasattr(self, "data"):
            self.data.ctrl[:] = 0.0

        
         
