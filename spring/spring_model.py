import mujoco
import mujoco.viewer
import numpy as np
import time
import math, os

os.environ['GDK_BACKEND'] = 'x11'

# Create a wavy height field
def generate_wavy_field(width, height, scale=0.5, amp=0.1):
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = amp * np.sin(scale*1) * np.cos(scale*j)
    return world
wavy_field = generate_wavy_field(50, 50)

rect_bot = """
<mujoco>
  <option timestep="0.001"/>

  <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.5 0.5 0.5" width="300" height="300" mark="edge" markrgb="0.2 0.2 0.2" />
        <material name="gridmat" texture="grid" texrepeat="5 5" reflectance="0.1"/>

        <hfield name="terrain" size="5 5 0.5 0.01" nrow="50" ncol="50"/>
    </asset>
   
  <!-- Add visual and lighting elements -->
  <visual>
    <global offwidth="800" offheight="800"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
  </visual>
  
  <worldbody>
    <!-- Add ground plane -->
    <geom type="hfield" hfield="terrain" material="gridmat" friction="1 0.5 0.1"/>
    
    <!-- Add light -->
    <light name="light" pos="0 0 3" dir="0 0 -1" directional="true" castshadow="true"/>
    
    <!-- Camera positioned to view the scene -->
    <camera name="fixed" pos="1.5 1.5 0.5" xyaxes="1 0 0 0 1 0"/>
    
    <!-- First rectangular block with its sites -->
    <body name="block1" pos="0 0 0.1">
      <freejoint name="block1_joint"/>
      <geom type="box" size="0.05 0.1 0.1" rgba="1 0 0 1" mass="1.0"/>
      <site name="site1a" pos="0.05 0.05 0.05"/>
      <site name="site2a" pos="0.05 -0.05 0.05"/>
      <site name="site3a" pos="0.05 0.05 -0.05"/>
      <site name="site4a" pos="0.05 -0.05 -0.05"/>
    </body>
    
    <!-- Second rectangular block with its sites -->
    <body name="block2" pos="0.5 0 0.1">
      <freejoint name="block2_joint"/>
      <geom type="box" size="0.05 0.1 0.1" rgba="0 0 1 1" mass="1.0"/>
      <site name="site1b" pos="-0.05 0.05 0.05"/>
      <site name="site2b" pos="-0.05 -0.05 0.05"/>
      <site name="site3b" pos="-0.05 0.05 -0.05"/>
      <site name="site4b" pos="-0.05 -0.05 -0.05"/>
    </body>
  </worldbody>
  
  <tendon>
    <!-- Four spring-damper connections between the blocks -->
    <spatial name="spring1" width="0.01" rgba="0 1 0 1" stiffness="500" damping="10">
      <site site="site1a"/>
      <site site="site1b"/>
    </spatial>
    <spatial name="spring2" width="0.01" rgba="0 1 0 1" stiffness="500" damping="10">
      <site site="site2a"/>
      <site site="site2b"/>
    </spatial>
    <spatial name="spring3" width="0.01" rgba="0 1 0 1" stiffness="500" damping="10">
      <site site="site3a"/>
      <site site="site3b"/>
    </spatial>
    <spatial name="spring4" width="0.01" rgba="0 1 0 1" stiffness="500" damping="10">
      <site site="site4a"/>
      <site site="site4b"/>
    </spatial>
  </tendon>
</mujoco>
"""

wavy_terrain = """
<mujoco>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.5 0.5 0.5" width="300" height="300" mark="edge" markrgb="0.2 0.2 0.2" />
        <material name="gridmat" texture="grid" texrepeat="5 5" reflectance="0.1"/>

        <hfield name="terrain" size="5 5 0.5 0.01" nrow="50" ncol="50"/>
    </asset>
    <worldbody>
        <geom type="hfield" hfield="terrain" material="gridmat" friction="1 0.5 0.1"/>
        <light pos="0 0 3"/>
        <camera name="free" pos="0 -3 2" xyaxes="1 0 0 0 1 2"/>
    </worldbody>
</mujoco>
"""

# def main():
#     model = mujoco.MjModel.from_xml_string(rect_bot)
#     data = mujoco.MjData(model)
#     # Comment this out for a wavy terrain!
#     # model.hfield_data[:] = wavy_field.flatten()
#     try:
#         # Get joint IDs instead of body IDs for safer velocity access
#         joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'block1_joint')
#         joint2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'block2_joint')
        
#         if joint1_id == -1 or joint2_id == -1:
#             raise ValueError("Could not find both joints in the model")
        
#         # Calculate proper velocity indices
#         # Each freejoint has 6 DOFs (3 position, 3 rotation)
#         vel1_idx = joint1_id * 6 + 3  # Start of rotational velocities
#         vel2_idx = joint2_id * 6 + 3
        
#         with mujoco.viewer.launch_passive(model, data) as viewer:
#             # Camera setup
#             viewer.cam.distance = 2.0
#             viewer.cam.azimuth = 45
#             viewer.cam.elevation = -20
            
#             # Rotation parameters
#             rotation_duration = 2.0  # seconds for half rotation
#             target_velocity = np.pi/rotation_duration  # rad/s for 180 degrees
#             current_active = 1  # Start with block1
#             last_switch = 0.0
#             start_delay = 1.0  # Initial delay before starting
            
#             # Simulation loop
#             while viewer.is_running():
#                 # Skip initial delay
#                 if data.time < start_delay:
#                     mujoco.mj_step(model, data)
#                     viewer.sync()
#                     time.sleep(0.001)
#                     continue
                
#                 # Switch active block periodically
#                 if data.time - last_switch > rotation_duration:
#                     current_active = 2 if current_active == 1 else 1
#                     last_switch = data.time
                
#                 # Apply rotational velocity to active block
#                 if current_active == 1:
#                     # Block1 rotates clockwise
#                     data.qvel[vel1_idx] = target_velocity  # X-axis rotation
#                     data.qvel[vel2_idx] = 0
#                 else:
#                     # Block2 rotates counter-clockwise
#                     data.qvel[vel2_idx] = target_velocity  # X-axis rotation
#                     data.qvel[vel1_idx] = 0
                
#                 mujoco.mj_step(model, data)
#                 viewer.sync()
#                 time.sleep(0.001)
                
#     except Exception as e:
#         print(f"Simulation error: {str(e)}")
#     finally:
#         del model
#         del data

# if __name__ == "__main__":
#     main()

# model = mujoco.MjModel.from_xml_path('/home/redhairedlynx/Documents/academics/hsa/spring/modified_joints.xml')
# data = mujoco.MjData(model)

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     for _ in range(100000000000000000000000):
#         mujoco.mj_step(model, data)
#         viewer.sync()

#         # time.sleep(0.01)

import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # Load your XML from string or file
   
    model = mujoco.MjModel.from_xml_path('/home/redhairedlynx/Documents/academics/hsa/spring/spring_blocks.xml')
    data = mujoco.MjData(model)

    # Get the actuator index
    # actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "spring_motor")
    # if actuator_id == -1:
    #     raise RuntimeError("Actuator 'spring_twist' not found.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()

        while viewer.is_running():
            elapsed = time.time() - t0

            # # Apply sinusoidal torque to the spring
            # amplitude = 1.0  # Try increasing to 3.0 or 5.0 if motion is weak
            # freq = 10.0       # 1 Hz oscillation
            # torque = amplitude * abs(np.sin(2 * np.pi * freq * elapsed))
           
            # data.ctrl[actuator_id] = 115.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1)

if __name__ == "__main__":
    main()
