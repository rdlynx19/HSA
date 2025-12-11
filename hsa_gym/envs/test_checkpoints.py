import matplotlib.pyplot as plt
import numpy as np
# Assuming HSAEnv is accessible in your environment
from hsa_gym.envs.hsa_constrained import HSAEnv 

# =========================================================
# HELPER FUNCTION FOR Z-HEIGHT VERIFICATION (Unchanged)
# =========================================================
def get_terrain_height(env, x, y):
    """
    Get the world Z-height of the terrain at a given world (X, Y) coordinate.
    """
    hfield_size = env.model.hfield_size[0]
    x_half, y_half, z_max, base_height = hfield_size
    
    nrow = env.model.hfield_nrow[0]
    ncol = env.model.hfield_ncol[0]
    
    # Map world coordinates to grid indices
    grid_i = int((x + x_half) / (2 * x_half) * (nrow - 1))
    grid_j = int((y + y_half) / (2 * y_half) * (ncol - 1))
    
    # Clamp to valid range
    grid_i = np.clip(grid_i, 0, nrow - 1)
    grid_j = np.clip(grid_j, 0, ncol - 1)

    # Get normalized height (0 to 1)
    terrain_data = env.model.hfield_data.reshape(nrow, ncol)
    stored_height = terrain_data[grid_i, grid_j]

    # Convert to actual height
    actual_height = base_height + stored_height * z_max
    return actual_height


# =========================================================
# MAIN PLOTTING SCRIPT
# =========================================================
# 1. Initialize Environment
env = HSAEnv(terrain_type="spiral", enable_terrain=True)

# 2. Extract Terrain Data and Extents
hfield = env.model.hfield_data.reshape(
    env.model.hfield_nrow[0], env.model.hfield_ncol[0]
)

# World extents for axes
x_half, y_half, z_max, base_height = env.model.hfield_size[0]
extent = [-x_half, x_half, -y_half, y_half]

# 3. Extract and Prepare Checkpoint Data
# Now that the generation is CW, we use standard, non-flipped Y-coordinates:
checkpoints = np.array(env._checkpoint_positions)
checkpoints_y = checkpoints[:, 1] 
checkpoints_x = checkpoints[:, 0]


# 4. Plot Terrain and Checkpoints
plt.figure(figsize=(8, 8))

# --- FINAL VISUALIZATION FIX ---
# Apply flipud to the hfield data BEFORE transpose/plotting.
# This corrects the vertical orientation of the heightmap image itself.
hfield_flipped = np.flipud(hfield) 

plt.imshow(
    hfield_flipped.T, # Transpose the flipped data
    origin="lower",
    extent=extent
)
plt.colorbar(label="Normalized Height (0-1)")

# Calculate Z-heights for coloring the checkpoints (diagnostic)
checkpoint_heights = [get_terrain_height(env, p[0], p[1]) for p in checkpoints]

# Plot the goals using the standard Y-coordinates (Rotation is handled in generate_spiral_track)
plt.scatter(
    checkpoints_x, 
    checkpoints_y, 
    c=checkpoint_heights, 
    cmap='viridis',
    s=80, 
    edgecolor='white',
    label="Checkpoints"
)
plt.colorbar(label="World Z-Height at Checkpoint (m)")


plt.title("Aligned Spiral Checkpoints on Correctly Oriented Terrain (CW)")
plt.xlabel("World X-Coordinate")
plt.ylabel("World Y-Coordinate")
plt.grid(True, alpha=0.3)
plt.show()

# Print diagnostics
print(f"\n--- Z-Height Verification ---")
print(f"Goal path verification (Terrain image and goal path should now be aligned):")
print(f"  Mean Z-height at checkpoints: {np.mean(checkpoint_heights):.4f} m")
print(f"  Standard Deviation of Z-height: {np.std(checkpoint_heights):.4f} m")
print(f"-----------------------------")