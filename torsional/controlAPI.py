import mujoco
import mujoco.viewer
import numpy as np
from .states import RobotState
import time

def require_state(*valid_states):
    """
    Decorator to ensure a method is only called on certain robot states
    :param valid_states: List of RobotState values in which the method is valid
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            """
            Wrapped version of the method which runs instead of the original. It decides whether we make a call to the original method or not. 
            """
            if self.state not in valid_states:
                print(f"Skipping {func.__name__}(): invalid robot state {self.state}")
                return None
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


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
        self.distances = []
        self.dt = self.model.opt.timestep
        self.trajectory = {}
        self.body_ids = {}

    def start_simulation(self) -> None:
        """
        Initialize/reset the simulation
        """
        self._reset_simulation()
        self.launch_viewer()

    def step_simulation(self) -> None: 
        """
        Step the MuJoCo simulation forward by one timestep
        """
        mujoco.mj_step(self.model, self.data)

    def _reset_simulation(self) -> None:
        """
        Reset the MuJoCo simulation to its initial state
        """
        mujoco.mj_resetData(self.model, self.data)  

    def launch_viewer(self) -> None:
        """
        Launch a passive viewer for the MuJoCo simulation
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            print(f"Viewer is already running!")

    def sync_viewer(self) -> None:
        """
        Sync the viewer with the current simulation state
        """
        if self.viewer is not None:
            self.viewer.sync()

    def close_simulation(self) -> None:
        """
        Close the MuJoCo simulation cleanly
        """
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None 

        self._reset_simulation()
   
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
    
    def get_joint_positions(self, joint_names: list[str]) -> dict[str, float]:
        """
        Get the positions of specified joint by their names
        :param joint_names: List of joint names to retrieve positions for

        :return: list of joint names and their positions
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
        Get the velocities of specified actuators by their names
        :param joint_names: List of actuator names to retrieve velocities for

        :return: list of actuator names and their velocities
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
        Get the IDs of specified actuators by their names
        :param actuator_names: List of actuator names to retrieve IDs for

        :return: list of actuator IDs
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
        Get a mapping of actuator IDs to their corresponding joint IDs
        :param actuator_names: List of actuator names to retrieve mappings for

        :return: Dictionary mapping actuator IDs to joint IDs
        """
        actuator_to_joint_ids = {}
        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in the model.")
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            actuator_to_joint_ids[actuator_id] = joint_id
        return actuator_to_joint_ids

    def get_robot_state(self) -> RobotState:
        """
        Get the current state of the robot
        :return: Current RobotState
        """
        return self.state

    def state_transition(self, new_state: RobotState) -> None:
        """
        Transition the robot to a new state
        :param new_state: Desired new RobotState
        """
        print(f"Transitioning from {self.state} to {new_state}")
        self.state = new_state

    def modify_equality_constraints(self, disable: bool = True, 
                                    constraints: list = 
                                    ["disc1b", "disc2b", 
                                     "disc3b", "disc4b"], 
                                     all_constraints: list = 
                                     ["disc1b", "disc2b", 
                                      "disc3b", "disc4b"]) -> None:
        """
        Enable or disable equality constraints in the simulation
        :param disable: True to disable (set to 0), False to enable (set to 1)
        :param constraints: List of constraint names to modify
        :param all_constraints: List of all possible constraint names
        """
        for i, name in enumerate(all_constraints):
            if name in constraints:
                self.data.eq_active[i] = 0 if disable else 1

        print(f"Equality constraints updated: {self.data.eq_active}")

    def get_friction_parameters(self, geom1_name: str, geom2_name: str) -> dict[str, float]:
        """
        Get the friction parameters between two bodies
        :param geom1_name: Name of the first geometry
        :param geom2_name: Name of the second geometry

        :return: Dictionary of friction parameters
        """
        geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
        geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)

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
        Set the friction parameters between two bodies
        :param geom1_name: Name of the first geometry
        :param geom2_name: Name of the second geometry
        :param tangential_x: New tangential friction in x direction
        :param tangential_y: New tangential friction in y direction
        :param rolling_x: New rolling friction in x direction
        :param rolling_y: New rolling friction in y direction
        """
        geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name)
        geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name)

        updated_contacts = []

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
                # print(f"Friction parameters updated for contact {i}")
                updated_contacts.append(contact)

    def update_friction_side(self,
                             side: str = "block_a", 
                             mode: str = "extension") -> None:
        """
        Update friction parameters based on the side and mode of operation
        :param side: Side of the robot ("block_a" or "block_b")
        :param mode: Mode of operation ("extension" or "contraction")
        """
        if self.data.ncon == 0:
            print(f"No contacts to update friction for.")
            return
        
        
        # Define geom pairs for both sides
        block_a_geoms = [("cylinder3a_con", "floor"), ("cylinder4c_con", "floor")]
        block_b_geoms = [("cylinder3c_con", "floor"), ("cylinder4a_con", "floor")]

        def get_friction_params(geom_pairs):
            return [self.get_friction_parameters(geom1, geom2) for geom1, geom2 in geom_pairs]
        
        # print(f"Friction parameters before update:")
        # print("Block A:", get_friction_params(block_a_geoms))
        # print("Block B:", get_friction_params(block_b_geoms))

        if mode == "extension":
            block_a_val = 0.0001
            block_b_val = 0.5
        elif mode == "contraction":
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
        # print(f"Friction parameters after update:")
        # print("Block A:", get_friction_params(block_a_geoms))
        # print("Block B:", get_friction_params(block_b_geoms))

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
        Apply velocity control to specified actuators for a given duration
        """
        self.disable_actuator_group(1)
        self.enable_actuator_group(2)

        pid_state = [{"error_prev": 0.0, "error_int": 0.0} for _ in actuator_names]     
        # Actuator Name order: spring1a, spring1c, spring3a, spring3c, spring2a, spring2c, spring4a, spring4c
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

    @require_state(RobotState.IDLE)
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
        Apply position control to obtain an extension motion
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = self.get_actuator_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:
            
            self.step_simulation()
            self.sync_viewer()
            
            while(self.data.ncon == 0):
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

            self.update_friction_side(side="block_a", mode="extension")
            self.update_friction_side(side="block_b", mode="extension")

            start_ctrl = np.zeros(len(actuator_ids))
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
        Apply position control to bring back the robot to its original state
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = self.get_actuator_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:

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

            # while self.viewer.is_running():
            #     self.step_simulation()
            #     self.sync_viewer()
            #     time.sleep(self.dt)
        except Exception as e:
            print(f"Unknown exception: {e}")

    @require_state(RobotState.IDLE)
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
        Perform crawling action by repeated contraction and extension
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = self.get_actuator_ids(actuator_names)
        actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:
            if not lock:
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
                                ["spring2c_motor", "spring2a_motor","spring4c_motor", "spring4a_motor"],
                                duration: float = 0.5,
                                position: float = 2.84,
                                plot: bool = False) -> None:
        """
        Perform twisting motion in one direction
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        
        actuator_ids = self.get_actuator_ids(actuator_names)
        actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc2b", "disc4b"])
            self.modify_equality_constraints(disable=False,
                                             constraints=["disc1b", "disc3b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
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
        Perform twisting motion in one direction
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control


        actuator_ids = self.get_actuator_ids(actuator_names)
        actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names)


        if self.viewer is None:
            self.start_simulation()
        try:
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc1b", "disc3b"])
            self.modify_equality_constraints(disable=False, 
                                             constraints=["disc2b", "disc4b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
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

            while self.viewer.is_running():
                self.step_simulation()
                self.sync_viewer()
                time.sleep(self.dt)

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
        Perform bending motion in one direction
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

        actuator_ids = self.get_actuator_ids(actuator_names)
        actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc1b", "disc4b"])
            self.modify_equality_constraints(disable=False,
                                             constraints=["disc2b", "disc3b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
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
        Perform bending motion in one direction
        """
        self.disable_actuator_group(2)
        self.enable_actuator_group(1) # Enabling position control

       
        actuator_ids = self.get_actuator_ids(actuator_names)
        actuator_to_joint_ids = self.get_actuator_to_joint_ids(actuator_names)

        if self.viewer is None:
            self.start_simulation()
        try:
            self.modify_equality_constraints(disable=True, 
                                         constraints=["disc2b", "disc3b"])
            self.modify_equality_constraints(disable=False,
                                             constraints=["disc1b", "disc4b"])

            self.step_simulation()
            self.sync_viewer()

            start_ctrl = np.zeros(len(actuator_ids))
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

    def euclidean_distance(self, body1: str, body2: str) -> float:
        """
        Compute the Euclidean distance between two bodies in the simulation.

        :param body1: Name of the first body
        :param body2: Name of the second body
        :return: Euclidean distance between the two bodies
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
        PID controller calculation
        :param Kp: Proportional gain
        :param Kd: Derivative gain
        :param Ki: Integral gain

        :return: Control signal, updated previous error, updated integral error
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
                       max_velocity: float = 6.28) -> None:
        """
        Apply PID control to the specified actuators to reach target setpoints
        :param actuator_names: List of actuator names to control
        :param joint_names: List of joint names corresponding to the actuators
        :param target_setpoints: Desired target positions or velocities for the joints
        :param pid_state: List of dictionaries maintaining PID state for each actuator
        :param dt: Time step for control updates
        :param Kp: List of proportional gains (one per actuator)
        :param Kd: List of derivative gains (one per actuator)
        :param Ki: List of integral gains (one per actuator)
        :param position: True for position control, False for velocity control
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
        Record the trajectory of specified bodies over time
        :param tracked_bodies: List of body names to track
        """
        if not hasattr(self, "trajectory"):
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
        View the MuJoCo model in a passive viewer
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

    
         
