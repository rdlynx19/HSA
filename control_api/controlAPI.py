"""
MuJoCo Control Interface for the Handed Shearing Auxetic (HSA) Robot.

This module provides the `MuJoCoControlInterface` class, a specialized interface 
for managing and controlling the HSA robot model within a MuJoCo simulation. 
It includes methods for state management, physics stepping, viewer synchronization, 
actuator group manipulation, and implementing basic robot behaviors (extension, 
contraction, bending, twisting, crawling) and trajectory 
interpolation.

The execution of movement methods is governed by the `require_state` decorator, 
enforcing a defined state machine for robot actions.
"""
import mujoco
import mujoco.viewer
import numpy as np
from .states import RobotState 
import time
from numpy.typing import NDArray 

def require_state(*valid_states: 'RobotState'):
    """
    Decorator to ensure a method is only called when the robot is in one of the specified states.

    If the current robot state is not in `valid_states`, the method call is skipped 
    and a warning is printed.

    :param valid_states: List of :py:class:`~RobotState` values in which the decorated method is valid.
    :type valid_states: tuple
    """
    def decorator(func):
        def wrapper(self: 'MuJoCoControlInterface', *args, **kwargs):
            """
            Wrapped version of the method which runs instead of the original. It decides whether we make a call to the original method or not. 
            """
            if self.state not in valid_states:
                print(f"Skipping {func.__name__}(): invalid robot state {self.state}")
                return None
            return func(self, *args, **kwargs)
        # Preserve original function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


class MuJoCoControlInterface:
    """
    Low-level control interface for HSA MuJoCo simulations.

    This class manages the MuJoCo model and data, handles state transitions, 
    and provides utility methods for kinematics, dynamics, and basic movement control.
    """
    def __init__(self, model_path: str = None):
        """
        Initialize the control interface by loading a MuJoCo model and data structures.

        :param model_path: Path to the MuJoCo model XML file.
        :type model_path: str
        :raises ValueError: If `model_path` is None.
        :returns: None
        :rtype: None
        """
        if model_path is None:
            raise ValueError("Model and data must be provided.")
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.state = RobotState.IDLE
        self.viewer = None
        self.distances = []
        self.dt = self.model.opt.timestep
        self.trajectory = {}
        self.body_ids = {}

    def start_simulation(self) -> None:
        """
        Initialize and reset the simulation, then launch the passive viewer if not already running.

        :returns: None
        :rtype: None
        """
        self._reset_simulation()
        self.launch_viewer()

    def step_simulation(self) -> None: 
        """
        Step the MuJoCo simulation forward by one timestep (:py:attr:`self.dt`).

        :returns: None
        :rtype: None
        """
        mujoco.mj_step(self.model, self.data)

    def _reset_simulation(self) -> None:
        """
        Reset the MuJoCo simulation to its initial state (qpos, qvel, ctrl=0).

        :returns: None
        :rtype: None
        """
        mujoco.mj_resetData(self.model, self.data)  

    def launch_viewer(self) -> None:
        """
        Launch a passive MuJoCo viewer for visualization.

        This sets up the viewer, locks the camera, and disables shadows/reflections for performance.

        :returns: None
        :rtype: None
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            with self.viewer.lock():
                scn = self.viewer.user_scn
                scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        else:
            print(f"Viewer is already running!")

    def sync_viewer(self) -> None:
        """
        Synchronize the viewer with the current simulation state (:py:attr:`self.data`).

        :returns: None
        :rtype: None
        """
        if self.viewer is not None:
            self.viewer.sync()

    def close_simulation(self) -> None:
        """
        Close the passive viewer and reset the MuJoCo data cleanly.

        :returns: None
        :rtype: None
        """
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None 

        self._reset_simulation()
   
    def enable_actuator_group(self, group_index: int) -> None:
        """
        Enable a specific actuator group by clearing its corresponding disable bitmask.

        :param group_index: Index (ID) of the actuator group to enable.
        :type group_index: int
        :returns: None
        :rtype: None
        """
        self.model.opt.disableactuator &= ~(1 << group_index)
       
    def disable_actuator_group(self, group_index: int) -> None:
        """
        Disable a specific actuator group by setting its corresponding disable bitmask.

        :param group_index: Index (ID) of the actuator group to disable.
        :type group_index: int
        :returns: None
        :rtype: None
        """
        self.model.opt.disableactuator |= (1 << group_index)
    
    def get_joint_positions(self, joint_names: list[str]) -> dict[str, float]:
        """
        Get the current positions of specified joints.

        :param joint_names: List of joint names to retrieve positions for.
        :type joint_names: list[str]
        :returns: Dictionary mapping joint names to their current position (float).
        :rtype: dict[str, float]
        :raises ValueError: If a joint name is not found in the model.
        """
        joint_positions = {}
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in the model.")
            position = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            joint_positions[name] = position
        return joint_positions
    
    def get_joint_velocities(self, joint_names: list[str]) -> dict[str, float]:
        """
        Get the current velocities of specified joints.

        :param joint_names: List of joint names to retrieve velocities for.
        :type joint_names: list[str]
        :returns: Dictionary mapping joint names to their current velocity (float).
        :rtype: dict[str, float]
        :raises ValueError: If a joint name is not found in the model.
        """
        joint_velocities = {}
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in the model.")
            velocity = self.data.qvel[self.model.jnt_dofadr[joint_id]]
            joint_velocities[name] = velocity
        return joint_velocities

    def get_actuator_ids(self, actuator_names: list[str]) -> list[int]:
        """
        Get the internal MuJoCo IDs of specified actuators.

        :param actuator_names: List of actuator names to retrieve IDs for.
        :type actuator_names: list[str]
        :returns: List of actuator IDs.
        :rtype: list[int]
        :raises ValueError: If an actuator name is not found in the model.
        """
        actuator_ids = []
        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in the model.")
            actuator_ids.append(actuator_id)
        return actuator_ids

    def get_actuator_to_joint_ids(self, actuator_names: list[str]) -> dict[int, int]:
        """
        Get a mapping of actuator IDs to their corresponding joint IDs.

        This uses the `actuator_trnid` field which maps the actuator to the DOF (degree of freedom) it controls, 
        and subsequently the joint it is associated with.

        :param actuator_names: List of actuator names to retrieve mappings for.
        :type actuator_names: list[str]
        :returns: Dictionary mapping actuator IDs (int) to joint IDs (int).
        :rtype: dict[int, int]
        :raises ValueError: If an actuator name is not found in the model.
        """
        actuator_to_joint_ids = {}
        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in the model.")
            # actuator_trnid[i, 0] gives the joint/site ID it's targeting
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            actuator_to_joint_ids[actuator_id] = joint_id
        return actuator_to_joint_ids

    def get_robot_state(self) -> 'RobotState':
        """
        Get the current state of the robot as defined by the internal state machine.

        :returns: Current :py:class:`~RobotState`.
        :rtype: RobotState
        """
        return self.state

    def state_transition(self, new_state: 'RobotState') -> None:
        """
        Transition the robot to a new operational state.

        :param new_state: Desired new :py:class:`~RobotState`.
        :type new_state: RobotState
        :returns: None
        :rtype: None
        """
        print(f"Transitioning from {self.state} to {new_state}")
        self.state = new_state

    def modify_equality_constraints(self, disable: bool = True, 
                                    constraints: list[str] = 
                                    ["disc1b", "disc2b", 
                                     "disc3b", "disc4b"], 
                                     all_constraints: list[str] = 
                                     ["disc1b", "disc2b", 
                                      "disc3b", "disc4b"]) -> None:
        """
        Enable or disable specific equality constraints in the simulation using their indices.

        This is typically used to selectively lock/unlock pairs of discs in the HSA robot.

        :param disable: True to disable (set state to 0), False to enable (set state to 1).
        :type disable: bool
        :param constraints: List of constraint names (as defined in XML) to modify.
        :type constraints: list[str]
        :param all_constraints: List of all possible constraint names (used for indexing).
        :type all_constraints: list[str]
        :returns: None
        :rtype: None
        """
        # NOTE: This assumes constraints are indexed sequentially based on all_constraints list order.
        for i, name in enumerate(all_constraints):
            if name in constraints:
                self.data.eq_active[i] = 0 if disable else 1

        print(f"Equality constraints updated: {self.data.eq_active}")

    def get_friction_parameters(self, geom1_name: str, geom2_name: str) -> dict[str, float] | None:
        """
        Get the friction parameters of an active contact pair between two specified geometries.

        :param geom1_name: Name of the first geometry.
        :type geom1_name: str
        :param geom2_name: Name of the second geometry.
        :type geom2_name: str
        :returns: Dictionary of friction parameters if contact exists and is active, otherwise None.
        :rtype: dict[str, float] or None
        """
        geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
        geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)
        
        # NOTE: This only checks active contacts (after mj_step).
        # To get default XML friction, one would check model.geom_friction.
        
        # Check active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == geom1_id and contact.geom2 == geom2_id) or (contact.geom1 == geom2_id and contact.geom2 == geom1_id):
                friction_params = {
                    "contact_id": i,
                    "tangential_x": contact.friction[0],
                    "tangential_y": contact.friction[1],
                    "rolling_x": contact.friction[3],
                    "rolling_y": contact.friction[4],
                }
                return friction_params

        return None
    
    def set_friction_parameters(self, geom1_name: str, 
                                geom2_name: str,
                                tangential_x: float = None,
                                tangential_y: float = None,
                                rolling_x: float = None,
                                rolling_y: float = None) -> None:
        """
        Set the friction parameters of any active contact pair between two specified geometries.

        This function modifies the friction parameters in the :py:attr:`self.data.contact` array.

        :param geom1_name: Name of the first geometry.
        :type geom1_name: str
        :param geom2_name: Name of the second geometry.
        :type geom2_name: str
        :param tangential_x: New tangential friction coefficient in x direction.
        :type tangential_x: float or None
        :param tangential_y: New tangential friction coefficient in y direction.
        :type tangential_y: float or None
        :param rolling_x: New rolling friction coefficient in x direction.
        :type rolling_x: float or None
        :param rolling_y: New rolling friction coefficient in y direction.
        :type rolling_y: float or None
        :returns: None
        :rtype: None
        """
        geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
        geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == geom1_id and contact.geom2 == geom2_id) or (contact.geom1 == geom2_id and contact.geom2 == geom1_id):
                if tangential_x is not None:
                    contact.friction[0] = tangential_x
                if tangential_y is not None:
                    contact.friction[1] = tangential_y
                if rolling_x is not None:
                    contact.friction[3] = rolling_x
                if rolling_y is not None:
                    contact.friction[4] = rolling_y

    def update_friction_side(self,
                             side: str = "block_a", 
                             mode: str = "extension") -> None:
        """
        Update the tangential friction parameters for specific foot geometries 
        based on the robot's intended side and mode of operation (extension/contraction).

        This is used to implement differential friction for anisotropic locomotion.

        :param side: Side of the robot being analyzed ("block_a" or "block_b"). (Currently unused in logic)
        :type side: str
        :param mode: Mode of operation ("extension" or "contraction") which defines the new friction coefficients.
        :type mode: str
        :returns: None
        :rtype: None
        """
        if self.data.ncon == 0:
            print(f"No contacts to update friction for.")
            return
        
        
        # Define geom pairs for both sides (assuming contact with 'floor')
        block_a_geoms = [("cylinder3a_con", "floor"), ("cylinder4c_con", "floor")]
        block_b_geoms = [("cylinder3c_con", "floor"), ("cylinder4a_con", "floor")]

        if mode == "extension":
            block_a_val = 0.0001
            block_b_val = 0.5
        elif mode == "contraction":
            # NOTE: The provided logic uses the same values for contraction as extension. 
            # This may be intentional or a place to tune for propulsion.
            block_a_val = 0.0001
            block_b_val = 0.5
        else:
            print(f"Invalid mode: {mode}")
            return
        
        # Apply friction updates based on the side
        for g1, g2 in block_a_geoms:
            self.set_friction_parameters(g1, g2, tangential_x=block_a_val)
        for g1, g2 in block_b_geoms:
            self.set_friction_parameters(g1, g2, tangential_x=block_b_val)

    @require_state(RobotState.IDLE, RobotState.EXTENDED)
    def velocity_control_drive(self, 
                               actuator_names: list[str] = 
                               ["spring1a_vel", "spring1c_vel",
                                "spring3a_vel", "spring3c_vel",
                                "spring2a_vel", "spring2c_vel",
                                "spring4a_vel", "spring4c_vel"],
                               joint_names: list[str] = 
                                ["cw_cont_1a_top", "cw_ext_1c_top",
                                 "cw_cont_3a_btm", "cw_ext_3c_btm",
                                 "cw_cont_2a_top", "cw_ext_2c_top",
                                 "cw_cont_4a_btm", "cw_ext_4c_btm"],
                               duration: float = 5.0,
                               velocity: float = 3.0, 
                               max_velocity: float = 6.28) -> None:
        """
        Apply velocity control (via PID loop) to specified actuators to achieve a continuous driving motion.

        This motion typically transitions the robot into the DRIVING state. Actuator group 1 (position) is disabled, 
        and group 2 (velocity) is enabled.

        :param actuator_names: List of velocity actuator names to control (Group 2).
        :type actuator_names: list[str]
        :param joint_names: List of corresponding joint names for reading feedback velocity.
        :type joint_names: list[str]
        :param duration: Duration (in seconds) of the initial trajectory interpolation phase.
        :type duration: float
        :param velocity: Target joint velocity setpoint for all joints (rad/s).
        :type velocity: float
        :param max_velocity: Maximum output clamp for the velocity control signal.
        :type max_velocity: float
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(1)
        self.enable_actuator_group(2)

        pid_state = [{"error_prev": 0.0, "error_int": 0.0} for _ in actuator_names]     
        list_Kp = [0.5, 0.7, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5]   
        list_Kd = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
        list_Ki = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        if self.viewer is None:
            self.start_simulation()
        try:
            self.step_simulation()
            self.sync_viewer()
            
            start_ctrl = np.zeros(len(actuator_names))
            target_ctrl = np.array([velocity if i % 2 == 0 else velocity for i in range(len(actuator_names))])
            trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration, self.dt, "smoothstep")

            for step_values in trajectory:
                pid_state = self.apply_pid_ctrl(actuator_names=actuator_names,
                                               joint_names=joint_names,
                                               target_setpoints=step_values,
                                               pid_state=pid_state,
                                               dt=self.dt,
                                               list_Kp=list_Kp, list_Kd=list_Kd, list_Ki=list_Ki,
                                               position=False, max_velocity=max_velocity)
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

            self.state_transition(RobotState.DRIVING)

            final_ctrl = trajectory[-1]
            while self.viewer.is_running():
                pid_state = self.apply_pid_ctrl(actuator_names=actuator_names,
                                               joint_names=joint_names,
                                               target_setpoints=final_ctrl,
                                               pid_state=pid_state,
                                               dt=self.dt,
                                               list_Kp=list_Kp, list_Kd=list_Kd, list_Ki=list_Ki,
                                               position=False,
                                               max_velocity=max_velocity)
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)


            self.step_simulation()
            self.sync_viewer()
            time.sleep(self.dt)
        

        except Exception as e:
            print(f"Unexpected error: {e}")

    @require_state(RobotState.IDLE, RobotState.BENDING)
    def position_control_extension(self,
                                   actuator_names: list[str] = 
                                   ["spring1a_motor", "spring1c_motor",
                                    "spring3a_motor", "spring3c_motor",
                                    "spring2a_motor", "spring2c_motor",
                                    "spring4a_motor", "spring4c_motor"],
                                    duration: float = 15.0,
                                    position: float = 2.84,
                                    plot: bool = False) -> None:
        """
        Apply position control (via setting `data.ctrl`) to obtain an extension (pushing forward) motion.

        This motion transitions the robot into the EXTENDED state and sets differential friction.

        :param actuator_names: List of position actuator names to control (Group 1).
        :type actuator_names: list[str]
        :param duration: Duration (in seconds) over which the position target is interpolated.
        :type duration: float
        :param position: Target absolute position (in radians) for the controlled joints.
        :type position: float
        :param plot: If True, records the Euclidean distance between blocks for trajectory analysis.
        :type plot: bool
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = self.get_actuator_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:
            self.modify_equality_constraints(disable=False, 
                                            constraints=["disc1b", "disc2b", "disc3b", "disc4b"])
            self.step_simulation()
            self.sync_viewer()
            
            # Wait for contact stabilization before friction update
            while(self.data.ncon == 0):
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

            self.update_friction_side(side="block_a", mode="extension")
            self.update_friction_side(side="block_b", mode="extension")

            start_ctrl = np.copy(self.data.ctrl[actuator_ids])
            # Target is alternating +position and -position
            target_ctrl = np.array([position if i % 2 == 0 else -position for i in range(len(actuator_ids))])

            # Generate interpolated trajectory
            trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration, self.dt, "linear")

            for step_values in trajectory:
                self.data.ctrl[actuator_ids] = step_values
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

            self.state_transition(RobotState.EXTENDED)
            if plot:
               distance = self.euclidean_distance("block_a", "block_b")
               self.distances.append((self.data.time, distance))

            self.step_simulation()
            self.sync_viewer()
            time.sleep(self.dt)
        except Exception as e:
            print(f"Unknown exception: {e}")
   
    @require_state(RobotState.BENDING, RobotState.TWISTING, RobotState.EXTENDED)
    def position_control_contraction(self, 
                                    actuator_names: list[str] = 
                                   ["spring1a_motor", "spring1c_motor",
                                    "spring3a_motor", "spring3c_motor",
                                    "spring2a_motor", "spring2c_motor",
                                    "spring4a_motor", "spring4c_motor"],
                                    duration: float = 0.5,
                                    plot: bool = False) -> None:
        """
        Apply position control to contract the robot back to its original (zero) state.

        This motion typically transitions the robot into the IDLE state.

        :param actuator_names: List of position actuator names to control (Group 1).
        :type actuator_names: list[str]
        :param duration: Duration (in seconds) over which the position target is interpolated to zero.
        :type duration: float
        :param plot: If True, records the Euclidean distance between blocks.
        :type plot: bool
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = self.get_actuator_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:
            self.modify_equality_constraints(disable=False, 
                                            constraints=["disc1b", "disc2b", "disc3b", "disc4b"])
            self.step_simulation()
            self.sync_viewer()

            while(self.data.ncon == 0):
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

            self.update_friction_side(side="block_a", mode="contraction")
            self.update_friction_side(side="block_b", mode="contraction")


            start_ctrl = np.copy(self.data.ctrl[actuator_ids])
            target_ctrl = np.zeros(len(actuator_ids))

            # Generate interpolated trajectory
            trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration, self.dt, "linear")

            for step_values in trajectory:
                self.data.ctrl[actuator_ids] = step_values
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

            self.state_transition(RobotState.IDLE)
            if plot:
               distance = self.euclidean_distance("block_a", "block_b")
               self.distances.append((self.data.time, distance))

            self.step_simulation()
            self.sync_viewer()
            time.sleep(self.dt)

        except Exception as e:
            print(f"Unknown exception: {e}")

    @require_state(RobotState.IDLE, RobotState.BENDING)
    def position_control_crawl(self,
                               actuator_names: list[str] = 
                                ["spring1a_motor", "spring1c_motor",
                                "spring3a_motor", "spring3c_motor",
                                "spring2a_motor", "spring2c_motor",
                                "spring4a_motor", "spring4c_motor"],
                                duration: float = 0.5,
                                position: float = 2.84,
                                lock: bool = False, 
                                plot: bool = False) -> None:
        """
        Perform a continuous crawling motion by repeatedly cycling through extension and contraction phases.

        The crawling cycle involves sequentially calling :py:meth:`~position_control_extension` 
        followed by :py:meth:`~position_control_contraction`.

        :param actuator_names: List of position actuator names (Group 1).
        :type actuator_names: list[str]
        :param duration: Duration of each sub-phase (extension and contraction).
        :type duration: float
        :param position: Target position for the extension phase.
        :type position: float
        :param lock: If True, equality constraints are kept enabled (locked). If False, they are temporarily disabled for certain constraints (currently disabled logic in this function).
        :type lock: bool
        :param plot: If True, records the trajectory/distance during sub-phases.
        :type plot: bool
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        # actuator_ids = self.get_actuator_ids(actuator_names) # Unused in this function

        if self.viewer is None:
            self.start_simulation()
        try:
            if not lock:
                # Disables equality constraints, allowing the mechanism to work
                self.modify_equality_constraints(disable=True, 
                                            constraints=["disc1b", "disc2b", "disc3b", "disc4b"])

            self.step_simulation()
            self.sync_viewer()  

            while self.viewer.is_running():
                self.position_control_extension(duration=duration, 
                                                position=position, plot=plot)   
                self.record_trajectory()
                self.position_control_contraction(duration=duration, plot=plot)
                self.record_trajectory()

        except Exception as e:
            print(f"Unknown error: {e}")

    @require_state(RobotState.IDLE)    
    def position_control_twist1(self, 
                                actuator_names: list[str] = 
                                ["spring2c_motor", "spring4c_motor","spring2a_motor", "spring4a_motor",
                                 "spring1c_motor", "spring3c_motor","spring1a_motor", "spring3a_motor"],
                                duration: float = 0.5,
                                position: float = 2.84,
                                plot: bool = False) -> None:
        """
        Perform a twisting motion sequence in one direction (Twist Type 1).

        This involves selectively disabling certain equality constraints (`disc2b`, `disc4b`)
        while keeping others enabled (`disc1b`, `disc3b`) to achieve differential movement.

        :param actuator_names: List of position actuator names (Group 1).
        :type actuator_names: list[str]
        :param duration: Duration over which the position target is interpolated.
        :type duration: float
        :param position: Target absolute position for the active joints.
        :type position: float
        :param plot: If True, records the Euclidean distance between blocks.
        :type plot: bool
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        
        actuator_ids = self.get_actuator_ids(actuator_names)
        # actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names) # Unused in this function

        if self.viewer is None:
            self.start_simulation()
        try:
            # Disable constraints 2 & 4, enable 1 & 3 (allowing differential twist)
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc2b", "disc4b"])
            self.modify_equality_constraints(disable=False,
                                             constraints=["disc1b", "disc3b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
            # Target: Activate first two actuators negatively, rest zero
            target_ctrl = np.array([-position, -position, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0])

            # Generate interpolated trajectory
            trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration, self.dt, "linear")

            for step_values in trajectory:
                self.data.ctrl[actuator_ids] = step_values
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)
              
            self.state_transition(RobotState.TWISTING)
                
            if plot:
                distance = self.euclidean_distance("block_a", "block_b")
                self.distances.append((self.data.time, distance))
            
            while self.viewer.is_running():
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

        except Exception as e:
            print(f"Unknown error: {e}")

    @require_state(RobotState.IDLE) 
    def position_control_twist2(self, 
                                actuator_names: list[str] = 
                                ["spring1c_motor", "spring1a_motor", "spring3c_motor", "spring3a_motor"],
                                duration: float = 0.5,
                                position: float = 2.84,
                                plot: bool = False) -> None:
        """
        Perform a twisting motion sequence in one direction (Twist Type 2).

        This involves selectively disabling certain equality constraints (`disc1b`, `disc3b`)
        while keeping others enabled (`disc2b`, `disc4b`) to achieve differential movement.

        :param actuator_names: List of position actuator names (Group 1).
        :type actuator_names: list[str]
        :param duration: Duration over which the position target is interpolated.
        :type duration: float
        :param position: Target absolute position for the active joints.
        :type position: float
        :param plot: If True, records the Euclidean distance between blocks.
        :type plot: bool
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control


        actuator_ids = self.get_actuator_ids(actuator_names)
        # actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names) # Unused in this function


        if self.viewer is None:
            self.start_simulation()
        try:
            # Disable constraints 1 & 3, enable 2 & 4
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc1b", "disc3b"])
            self.modify_equality_constraints(disable=False, 
                                             constraints=["disc2b", "disc4b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
            # Target: Activate alternating positions
            target_ctrl = np.array([-position if i % 2 == 0 else 0.0 for i in range(len(actuator_ids))])

            # Generate interpolated trajectory
            trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration, self.dt, "linear")

            for step_values in trajectory:
                self.data.ctrl[actuator_ids] = step_values
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

            self.state_transition(RobotState.TWISTING)

            if plot:
                distance = self.euclidean_distance("block_a", "block_b")
                self.distances.append((self.data.time, distance))

        except Exception as e:
            print(f"Unknown error: {e}")

    @require_state(RobotState.IDLE)
    def position_control_bend_left(self, 
                              actuator_names: list[str] = 
                              ["spring1a_motor", "spring4c_motor",
                                "spring2c_motor", "spring3a_motor",
                                "spring2a_motor", "spring1c_motor",
                                "spring4a_motor", "spring3c_motor"],
                              duration: float = 0.5, 
                              position: float = 2.8,
                              plot: bool = False) -> None:
        """
        Perform a bending motion sequence to the left.

        This involves selectively disabling constraints to allow bending while maintaining 
        other structural integrity.

        :param actuator_names: List of position actuator names (Group 1).
        :type actuator_names: list[str]
        :param duration: Duration over which the position target is interpolated.
        :type duration: float
        :param position: Target absolute position for the active joints.
        :type position: float
        :param plot: If True, records the Euclidean distance between blocks.
        :type plot: bool
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = self.get_actuator_ids(actuator_names)
        # actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names) # Unused in this function

        if self.viewer is None:
            self.start_simulation()
        try:
            # Disable constraints 1 & 4, enable 2 & 3
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc1b", "disc4b"])
            self.modify_equality_constraints(disable=False,
                                             constraints=["disc2b", "disc3b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
            # Target: Activate first two actuators differentially, rest zero
            target_ctrl = np.array([position, -position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Generate interpolated trajectory
            trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration, self.dt, "linear")

            for step_values in trajectory:
                self.data.ctrl[actuator_ids] = step_values
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)
              
            self.state_transition(RobotState.BENDING)  
            if plot:
               distance = self.euclidean_distance("block_a", "block_b")
               self.distances.append((self.data.time, distance))

        except Exception as e:
            print(f"Unknown error: {e}")

    @require_state(RobotState.IDLE)
    def position_control_bend_right(self, 
                              actuator_names: list[str] = 
                              ["spring3a_motor", "spring2c_motor",
                                "spring4c_motor", "spring1a_motor",
                                "spring4a_motor", "spring3c_motor",
                                "spring2a_motor", "spring1c_motor"],
                              duration: float = 0.5, 
                              position: float = 2.8,
                              plot: bool = False) -> None:
        """
        Perform a bending motion sequence to the right.

        This involves selectively disabling constraints to allow bending while maintaining 
        other structural integrity.

        :param actuator_names: List of position actuator names (Group 1).
        :type actuator_names: list[str]
        :param duration: Duration over which the position target is interpolated.
        :type duration: float
        :param position: Target absolute position for the active joints.
        :type position: float
        :param plot: If True, records the Euclidean distance between blocks.
        :type plot: bool
        :returns: None
        :rtype: None
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

       
        actuator_ids = self.get_actuator_ids(actuator_names)
        # actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names) # Unused in this function

        if self.viewer is None:
            self.start_simulation()
        try:
            # Disable constraints 2 & 3, enable 1 & 4
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc2b", "disc3b"])
            self.modify_equality_constraints(disable=False,
                                             constraints=["disc1b", "disc4b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
            # Target: Activate first two actuators differentially, rest zero
            target_ctrl = np.array([position, -position, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Generate interpolated trajectory
            trajectory = self.interpolate_values(start_ctrl, target_ctrl, duration, self.dt, "linear")

            for step_values in trajectory:
                self.data.ctrl[actuator_ids] = step_values
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)
              
            self.state_transition(RobotState.BENDING)  
            if plot:
               distance = self.euclidean_distance("block_a", "block_b")
               self.distances.append((self.data.time, distance))

        except Exception as e:
            print(f"Unknown error: {e}")

    def interpolate_values(self, start: np.ndarray | list[float],
                           goal: np.ndarray | list[float],
                           duration: float,
                           timestep: float,
                           method: str = "linear") -> np.ndarray:
        """
        Generate interpolated values between start and goal over a given duration.

        This utility function computes a sequence of control setpoints ($\mathbf{u}$) 
        smoothly transitioning from the start state to the goal state over time.

        :param start: Starting value vector (e.g., current control vector).
        :type start: np.ndarray or list[float]
        :param goal: Goal value vector (e.g., target control vector).
        :type goal: np.ndarray or list[float]
        :param duration: Total transition time in seconds.
        :type duration: float
        :param timestep: Simulation time step $\Delta t$ (e.g., :py:attr:`self.dt`).
        :type timestep: float
        :param method: Interpolation type: ``"linear"``, ``"cosine"``, ``"smoothstep"``, or ``"quadratic"``.
        :type method: str
        :returns: A 2D NumPy array where each row represents the control vector for one simulation step.
        :rtype: np.ndarray
        :raises ValueError: If an unknown interpolation method is provided.
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
        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)

        # Interpolation: (1 - alpha) * start + alpha * goal
        interpolated = np.outer(1 - alphas, start_arr) + np.outer(alphas, goal_arr)

        return interpolated

    def euclidean_distance(self, body1: str, body2: str) -> float:
        """
        Compute the Euclidean distance between the centers of mass (xpos) of two specified bodies.

        :param body1: Name of the first body.
        :type body1: str
        :param body2: Name of the second body.
        :type body2: str
        :returns: Euclidean distance (float) between the two body origins.
        :rtype: float
        :raises ValueError: If one or both body names are not found in the model.
        """
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

        if body1_id < 0 or body2_id < 0:
            raise ValueError(f"One or both bodies '{body1}', '{body2}' not found in the model.")

        pos1 = self.data.xpos[body1_id]
        pos2 = self.data.xpos[body2_id]

        distance = np.linalg.norm(pos1 - pos2)
        return distance

    def _compute_pid(self, error: float = 0.0, 
                    error_prev: float = 0.0, 
                    error_int: float = 0.0,
                    dt: float = 0.001,
                    Kp: float = 1.0, Kd: float = 0.0,
                    Ki: float = 0.0) -> tuple[float, float, float]:
        """
        Internal helper function for a single PID controller iteration calculation.

        :param error: Current error ($\text{setpoint} - \text{actual\_value}$).
        :type error: float
        :param error_prev: Error from the previous timestep.
        :type error_prev: float
        :param error_int: Accumulated integral error.
        :type error_int: float
        :param dt: Time step $\Delta t$.
        :type dt: float
        :param Kp: Proportional gain.
        :type Kp: float
        :param Kd: Derivative gain.
        :type Kd: float
        :param Ki: Integral gain.
        :type Ki: float
        :returns: A tuple: (Control signal, updated previous error, updated integral error).
        :rtype: tuple[float, float, float]
        """
        error_int += error * dt
        error_deriv = (error - error_prev) / dt 
        ctrl_signal = Kp * error + Kd * error_deriv + Ki * error_int
        return ctrl_signal, error, error_int

    def apply_pid_ctrl(self, actuator_names: list[str] = None, 
                       joint_names: list[str] = None,
                       target_setpoints: list[float] = None,
                       pid_state: list[dict[str, float]] = None,
                       dt : float = 0.001,
                       list_Kp: list[float] = [0.0], 
                       list_Kd: list[float] = [0.0],
                       list_Ki: list[float] = [0.0], 
                       position: bool = False, 
                       max_velocity: float = 6.28) -> list[dict[str, float]]:
        """
        Applies PID control across multiple actuators (parallel PID).

        This function reads current joint positions/velocities, calculates control signals 
        to track the `target_setpoints`, and writes the resulting signals to 
        :py:attr:`self.data.ctrl`.

        :param actuator_names: List of actuator names to control.
        :type actuator_names: list[str] or None
        :param joint_names: List of joint names corresponding to the actuators (for feedback).
        :type joint_names: list[str] or None
        :param target_setpoints: Desired target values (positions or velocities).
        :type target_setpoints: list[float] or None
        :param pid_state: List of dictionaries maintaining PID state (error\_prev, error\_int) for each actuator.
        :type pid_state: list[dict[str, float]] or None
        :param dt: Time step $\Delta t$ for derivative/integral calculation.
        :type dt: float
        :param list_Kp: List of proportional gains (one per actuator).
        :type list_Kp: list[float]
        :param list_Kd: List of derivative gains.
        :type list_Kd: list[float]
        :param list_Ki: List of integral gains.
        :type list_Ki: list[float]
        :param position: True for position control (feedback from $qpos$), False for velocity control (feedback from $qvel$).
        :type position: bool
        :param max_velocity: Maximum output clamp for the control signal (used only in velocity control).
        :type max_velocity: float
        :returns: The updated PID state list.
        :rtype: list[dict[str, float]]
        :raises ValueError: If required input lists are missing.
        """
        if actuator_names is None or joint_names is None or target_setpoints is None or pid_state is None:
            raise ValueError("Actuator names, joint names, target setpoints, and PID state must be provided.")
        
        actuator_ids = self.get_actuator_ids(actuator_names)
        ctrl_signals = np.zeros(len(actuator_ids))

        if not position:
            current_values = self.get_joint_velocities(joint_names)
        else:
            current_values = self.get_joint_positions(joint_names)
        
        # Compute PID control for each actuator
        for i, act_id in enumerate(actuator_ids):
            desired_value = target_setpoints[i]
            actual_value = current_values[joint_names[i]]
            error = desired_value - actual_value

            error_prev = pid_state[i].get("error_prev", 0.0)
            error_int = pid_state[i].get("error_int", 0.0)

            ctrl_signals[i], updated_error_prev, updated_error_int = self._compute_pid(
                error, error_prev, error_int, dt, list_Kp[i], list_Kd[i], list_Ki[i]
            )

            # Clamp control signals to max velocity limits
            if not position:
                ctrl_signals[i] = np.clip(ctrl_signals[i], -max_velocity, max_velocity)

            # Update PID state
            pid_state[i]["error_prev"] = updated_error_prev
            pid_state[i]["error_int"] = updated_error_int
        
        # Apply control signals to actuators
        self.data.ctrl[actuator_ids] = ctrl_signals

        return pid_state

    def record_trajectory(self, 
                          tracked_bodies: list[str] = ["disc1b"]) -> None:
        """
        Record the 3D position (xpos) of specified bodies over the simulation time.

        The trajectory data is stored in the :py:attr:`self.trajectory` dictionary.

        :param tracked_bodies: List of body names to track.
        :type tracked_bodies: list[str]
        :returns: None
        :rtype: None
        """
        if not hasattr(self, "trajectory") or not self.trajectory:
            self.trajectory = {b: [] for b in tracked_bodies}
            self.body_ids = {b: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in tracked_bodies}
        
        for body in tracked_bodies:
            if body not in self.trajectory:
                self.trajectory[body] = []
                self.body_ids[body] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)

            body_id = self.body_ids[body]
            pos = self.data.xpos[body_id].copy()
            self.trajectory[body].append((self.data.time, pos))

    def view_model(self) -> None:
        """
        Open the passive viewer and loop through simulation steps until the viewer is closed.

        :returns: None
        :rtype: None
        """
        if self.viewer is None:
            self.start_simulation()
        try:
            self.step_simulation()
            self.sync_viewer()

            while self.viewer.is_running():
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)
        except Exception as e:            
            print(f"Unknown error: {e}")