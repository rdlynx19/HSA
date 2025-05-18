import mujoco
import mediapy as media
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt


tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4" gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" mass="0.1"/>
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008" mass="0.05" />
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3" />
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(tippe_top)
data = mujoco.MjData(model)

# Interactive mujoco viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
  mujoco.mj_resetDataKeyframe(model, data, 0)
  print("Initial qvel (angular velocity): ", data.qvel)
  mujoco.mj_forward(model, data)

  while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()

