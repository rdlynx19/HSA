import mujoco
import mujoco.viewer
import numpy as np
import noise

# Create a wavy height field
def generate_wavy_field(width, height, scale=0.5, amp=0.5):
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = amp * np.sin(scale*1) * np.cos(scale*j)
    return world
wavy_field = generate_wavy_field(50, 50)

# Generate noise for terrain
def generate_perlin_noise(width, height, scale=0.1):
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = noise.pnoise2(i*scale, j*scale, octaves=2)
    return world
height_field = generate_perlin_noise(50, 50)

# Stair Terrain
def stairs_terrain(model, step_height=0.2, step_width=0.5):
    height = np.zeros((model.hfield_nrow[0], model.hfield_ncol[0]))
    for i in range(height.shape[0]):
        height[i,:] = (i // 20) * step_height  # 20 pixels per step


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
rocky_terrain = """
<mujoco>
  <asset>
    <hfield name="rocks" size="5 5 0.5 0.01" nrow="50" ncol="50"/>
    <material name="gray" rgba="0.7 0.7 0.7 1"/>
  </asset>
  <worldbody>
    <geom type="hfield" hfield="rocks" material="gray" friction="1 0.5 0.1"/>
    <light pos="0 0 3"/>
    <camera pos="0 -3 2" xyaxes="1 0 0 0 1 2"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(rocky_terrain)
data = mujoco.MjData(model)
# Select the pattern to load into world
model.hfield_data[:] = height_field.flatten()

with mujoco.viewer.launch_passive(model, data) as viewer:

  while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
