import mujoco
import mujoco.viewer
import numpy as np
from enum import Enum, auto
import time

class RobotState(Enum):
    """
    Enum for robot states
    """
    IDLE = auto()
    EXTENDED = auto()



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

    def velocity_control_drive(self, actuator_names: list[str], joint_name: str, velocity: float, duration: float) -> None:
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
                    else:
                        self.data.ctrl[act_id] = -velocity 

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
         
