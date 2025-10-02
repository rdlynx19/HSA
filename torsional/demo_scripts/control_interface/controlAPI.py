import mujoco
import mujoco.viewer
import numpy as np
from enum import Enum, auto

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
      
    def check_actuator_group_status(self, group_index: int) -> bool:
        """
        Check if a specific actuator group is disabled
        :param group_index: Index of the actuator group to check

        :return: True if the group is disabled, False otherwise
        """
        return ((self.model.opt.disableactuator >> group_index) & 1) == 1
    
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

    def launch_viewer(self) -> mujoco.viewer.MjViewer:
        """
        Launch a passive viewer for the MuJoCo simulation
        """
        self.viewer =  mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer
    
    def sync_viewer(self) -> None:
        """
        Sync the viewer with the current simulation state
        """
        if self.viewer is not None:
            self.viewer.sync()

    def position_control_drive(self, position_actuators: list[str], joint_name: str, target_positions: list[float], duration: float) -> None:
        """
        Apply position control to specified actuators and monitor a joint's position
        :param position_actuators: List of actuator names to apply position control to
        :param joint_name: Name of the joint to monitor
        :param target_positions: List of target positions to alternate between
        :param duration: Duration in seconds for each target position
        """
        # Get robot state
        robot_state = self.get_robot_state()
        # Change actuator group state
        

        actuator_ids = []
        # Save actuator IDs to send position commands
        for name in position_actuators:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in the model.")
            actuator_ids.append(actuator_id)

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in the model.")
        
