"""
Procedural Terrain Generation Utilities for MuJoCo Heightfields.

This module provides various functions to generate 2D NumPy arrays representing 
heightfields (terrains) for use in MuJoCo simulations. The array values correspond 
to height, typically normalized or scaled relative to the maximum height defined 
in the MuJoCo XML model.

Functions include generation methods for:
* Flat surfaces, inclines, and ramps.
* Stochastic surfaces like Perlin noise and craters.
* Structured environments like spiral tracks and corridors.

The main entry point is `generate_terrain`, which dispatches to the specific 
generation function based on the requested type.
"""
import numpy as np
import noise
from numpy.typing import NDArray 

# --- Utility Functions ---

def ensure_flat_spawn_zone(terrain: NDArray[np.float64], 
                           spawn_center_x: float = 0.5, 
                           spawn_center_y: float = 0.5, 
                           spawn_radius: float = 0.15) -> NDArray[np.float64]:
    """
    Ensure a flat area in the terrain for the robot's initial spawn location.

    The function sets the height to 0.0 within a specified circular region and 
    applies a smooth, linear blend transition zone around the perimeter to 
    connect the flat area with the surrounding terrain. 

    :param terrain: The input heightmap array.
    :type terrain: NDArray[np.float64]
    :param spawn_center_x: Center of the flat zone as a fraction (0.0 to 1.0) of the terrain width.
    :type spawn_center_x: float
    :param spawn_center_y: Center of the flat zone as a fraction (0.0 to 1.0) of the terrain height.
    :type spawn_center_y: float
    :param spawn_radius: Radius of the flat zone as a fraction of the terrain size.
    :type spawn_radius: float
    :returns: The modified heightmap array with the flat spawn zone.
    :rtype: NDArray[np.float64]
    """
    width, height = terrain.shape
    
    # Convert to grid coordinates
    center_i = int(spawn_center_x * width)
    center_j = int(spawn_center_y * height)
    radius_cells = int(spawn_radius * min(width, height))
    
    # Flatten circular region
    for i in range(width):
        for j in range(height):
            dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            if dist <= radius_cells:
                terrain[i, j] = 0.0  # Flat ground height
            elif dist <= radius_cells * 1.2:  # Smooth transition zone
                # Linear blend from flat to terrain
                blend_factor = (dist - radius_cells) / (radius_cells * 0.2)
                terrain[i, j] = terrain[i, j] * blend_factor
    
    return terrain


# --- Terrain Generation Functions ---

def generate_wavy_field(width: int, height: int, scale: float = 0.5, amp: float = 0.5) -> NDArray[np.float64]:
    """
    Generate a terrain heightmap based on a 2D sine-cosine wave pattern.

    :param width: Terrain grid width (number of cells).
    :type width: int
    :param height: Terrain grid height (number of cells).
    :type height: int
    :param scale: Controls the frequency/wavelength of the wave pattern.
    :type scale: float
    :param amp: Controls the amplitude (maximum height/depth) of the waves.
    :type amp: float
    :returns: The heightmap array.
    :rtype: NDArray[np.float64]
    """
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = amp * np.sin(scale*i) * np.cos(scale*j)
    return world

def generate_perlin_noise(width: int, height: int, scale: float = 0.1) -> NDArray[np.float64]:
    """
    Generate a terrain heightmap using 2D Perlin noise for smooth, fractal features.

    :param width: Terrain grid width (number of cells).
    :type width: int
    :param height: Terrain grid height (number of cells).
    :type height: int
    :param scale: Controls the density and scale of the noise features.
    :type scale: float
    :returns: The heightmap array.
    :rtype: NDArray[np.float64]
    """
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = noise.pnoise2(i*scale, j*scale, octaves=2)
    return world

def generate_stairs_terrain(width: int, height: int, step_height: float = 0.02, pixels_per_step: int = 5) -> NDArray[np.float64]:
    """
    Generate a terrain consisting of a straight staircase along the X-axis.

    :param width: Terrain grid width (number of cells).
    :type width: int
    :param height: Terrain grid height (number of cells).
    :type height: int
    :param step_height: The change in height for each step/tread.
    :type step_height: float
    :param pixels_per_step: The number of grid cells (pixels) defining the length of each step.
    :type pixels_per_step: int
    :returns: The heightmap array.
    :rtype: NDArray[np.float64]
    """
    terrain = np.zeros((width, height))
    for i in range(width):
        terrain[i, :] = (i // pixels_per_step) * step_height
    return terrain

def generate_flat_terrain(width: int, height: int) -> NDArray[np.float64]:
    """
    Generate a completely flat terrain heightmap at zero height.

    :param width: Terrain grid width (number of cells).
    :type width: int
    :param height: Terrain grid height (number of cells).
    :type height: int
    :returns: The heightmap array (all zeros).
    :rtype: NDArray[np.float64]
    """
    return np.zeros((width, height))

def generate_flat_with_incline(width: int, height: int, 
                               incline_start_x: float = 0.3, incline_end_x: float = 0.7,
                               incline_start_y: float = 0.3, incline_end_y: float = 0.7,
                               angle: float = 15, direction: str = 'x') -> NDArray[np.float64]:
    """
    Generate flat terrain with a localized incline or ramp created within a specified rectangular region.

    :param width: Terrain grid width.
    :type width: int
    :param height: Terrain grid height.
    :type height: int
    :param incline_start_x: Starting X position of the incline region as a fraction (0-1).
    :type incline_start_x: float
    :param incline_end_x: Ending X position of the incline region as a fraction (0-1).
    :type incline_end_x: float
    :param incline_start_y: Starting Y position of the incline region as a fraction (0-1).
    :type incline_start_y: float
    :param incline_end_y: Ending Y position of the incline region as a fraction (0-1).
    :type incline_end_y: float
    :param angle: Slope angle in degrees.
    :type angle: float
    :param direction: Direction of the slope within the region: ``'x'``, ``'y'``, or ``'diagonal'``.
    :type direction: str
    :returns: The heightmap array with the ramp feature.
    :rtype: NDArray[np.float64]
    """
    # Start with flat terrain
    terrain = np.zeros((width, height))
    
    # Convert fractions to grid indices
    start_i = int(incline_start_x * width)
    end_i = int(incline_end_x * width)
    start_j = int(incline_start_y * height)
    end_j = int(incline_end_y * height)
    
    # Calculate slope
    slope = np.tan(np.radians(angle))
    
    # Create incline in the specified region
    incline_width = end_i - start_i
    incline_height = end_j - start_j
    
    if direction == 'x':
        # Slope increases along x direction within the region
        for i in range(start_i, end_i):
            local_progress = (i - start_i) / incline_width
            incline_height_val = local_progress * slope
            terrain[i, start_j:end_j] = incline_height_val
    
    elif direction == 'y':
        # Slope increases along y direction within the region
        for j in range(start_j, end_j):
            local_progress = (j - start_j) / incline_height
            incline_height_val = local_progress * slope
            terrain[start_i:end_i, j] = incline_height_val
    
    elif direction == 'diagonal':
        # Slope increases diagonally within the region
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                local_progress_x = (i - start_i) / incline_width
                local_progress_y = (j - start_j) / incline_height
                local_progress = (local_progress_x + local_progress_y) / 2
                terrain[i, j] = local_progress * slope
    
    return terrain

def generate_incline_terrain(width: int, height: int, angle: float = 10, direction: str = 'x') -> NDArray[np.float64]:
    """
    Generate a full-width terrain with a uniform slope across the entire grid.

    :param width: Terrain grid width.
    :type width: int
    :param height: Terrain grid height.
    :type height: int
    :param angle: Slope angle in degrees.
    :type angle: float
    :param direction: Direction of the slope: ``'x'``, ``'y'``, or ``'diagonal'``.
    :type direction: str
    :returns: The heightmap array with the incline.
    :rtype: NDArray[np.float64]
    """
    terrain = np.zeros((width, height))
    slope = np.tan(np.radians(angle))
    
    if direction == 'x':
        for i in range(width):
            terrain[i, :] = (i / width) * slope
    elif direction == 'y':
        for j in range(height):
            terrain[:, j] = (j / height) * slope
    elif direction == 'diagonal':
        for i in range(width):
            for j in range(height):
                terrain[i, j] = ((i + j) / (width + height)) * slope
    
    return terrain

def generate_flat_with_ramp(width: int, height: int,
                            ramp_center_x: float = 0.5, ramp_center_y: float = 0.5,
                            ramp_length: float = 0.3, ramp_width: float = 0.3,
                            ramp_height: float = 0.2, direction: str = 'x',
                            smooth_edges: bool = True) -> NDArray[np.float64]:
    """
    Generate flat terrain with a smooth, localized ramp feature.

    :param width: Terrain grid width.
    :type width: int
    :param height: Terrain grid height.
    :type height: int
    :param ramp_center_x: Center X position of the ramp as a fraction (0-1).
    :type ramp_center_x: float
    :param ramp_center_y: Center Y position of the ramp as a fraction (0-1).
    :type ramp_center_y: float
    :param ramp_length: Length of the ramp (distance over which height changes) as a fraction of terrain size.
    :type ramp_length: float
    :param ramp_width: Width of the ramp as a fraction of terrain size.
    :type ramp_width: float
    :param ramp_height: Maximum height of the ramp.
    :type ramp_height: float
    :param direction: Direction of the slope: ``'x'``, ``'-x'``, ``'y'``, or ``'-y'``.
    :type direction: str
    :param smooth_edges: If True, applies a smooth falloff along the ramp's width.
    :type smooth_edges: bool
    :returns: The heightmap array with the localized ramp.
    :rtype: NDArray[np.float64]
    """
    # Start with flat terrain
    terrain = np.zeros((width, height))
    
    # Convert to grid coordinates
    center_i = int(ramp_center_x * width)
    center_j = int(ramp_center_y * height)
    length_cells = int(ramp_length * width)
    width_cells = int(ramp_width * height)
    
    # Define ramp boundaries
    if direction in ['x', '-x']:
        half_length = length_cells // 2
        half_width = width_cells // 2
        start_i = center_i - half_length
        end_i = center_i + half_length
        start_j = center_j - half_width
        end_j = center_j + half_width
        
        for i in range(max(0, start_i), min(width, end_i)):
            for j in range(max(0, start_j), min(height, end_j)):
                # Calculate progress along ramp
                if direction == 'x':
                    progress = (i - start_i) / length_cells
                else:  # '-x'
                    progress = 1 - (i - start_i) / length_cells
                
                # Linear ramp height
                ramp_z = progress * ramp_height
                
                # Optional: smooth edges
                if smooth_edges:
                    # Distance from center line
                    dist_from_center = abs(j - center_j) / half_width
                    if dist_from_center > 0.7:  # Fade near edges
                        edge_factor = 1 - (dist_from_center - 0.7) / 0.3
                        ramp_z *= max(0, edge_factor)
                
                terrain[i, j] = ramp_z
    
    elif direction in ['y', '-y']:
        half_length = length_cells // 2
        half_width = width_cells // 2
        start_i = center_i - half_width
        end_i = center_i + half_width
        start_j = center_j - half_length
        end_j = center_j + half_length
        
        for i in range(max(0, start_i), min(width, end_i)):
            for j in range(max(0, start_j), min(height, end_j)):
                # Calculate progress along ramp
                if direction == 'y':
                    progress = (j - start_j) / length_cells
                else:  # '-y'
                    progress = 1 - (j - start_j) / length_cells
                
                # Linear ramp height
                ramp_z = progress * ramp_height
                
                # Optional: smooth edges
                if smooth_edges:
                    dist_from_center = abs(i - center_i) / half_width
                    if dist_from_center > 0.7:
                        edge_factor = 1 - (dist_from_center - 0.7) / 0.3
                        ramp_z *= max(0, edge_factor)
                
                terrain[i, j] = ramp_z
    
    return terrain

def generate_crater_terrain(width: int, height: int, num_craters: int = 20, crater_radius_range: tuple[int, int] = (2, 5), 
                            crater_depth_range: tuple[float, float] = (0.05, 0.15)) -> NDArray[np.float64]:
    """
    Generate terrain with random craters (small, Gaussian-shaped depressions).

    :param width: Terrain grid width.
    :type width: int
    :param height: Terrain grid height.
    :type height: int
    :param num_craters: Number of craters to generate.
    :type num_craters: int
    :param crater_radius_range: Tuple of (min_radius, max_radius) in grid units.
    :type crater_radius_range: tuple[int, int]
    :param crater_depth_range: Tuple of (min_depth, max_depth) in height units.
    :type crater_depth_range: tuple[float, float]
    :returns: The heightmap array with craters.
    :rtype: NDArray[np.float64]
    """
    terrain = np.zeros((width, height))
    
    # Generate random craters
    for _ in range(num_craters):
        # Random crater center
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        # Random crater properties
        radius = np.random.uniform(*crater_radius_range)
        depth = np.random.uniform(*crater_depth_range)
        
        # Create crater using Gaussian-like depression
        for i in range(width):
            for j in range(height):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist < radius:
                    # Smooth crater shape (inverted Gaussian)
                    crater_profile = depth * np.exp(-(dist**2) / (2 * (radius/3)**2))
                    terrain[i, j] -= crater_profile
    
    return terrain


def generate_poles_terrain(width: int, height: int, num_poles: int = 8, pole_radius_range: tuple[int, int] = (3, 6), 
                           pole_height_range: tuple[float, float] = (0.3, 0.6), min_spacing: int = 8) -> NDArray[np.float64]:
    """
    Generate terrain with large cylindrical poles/pillars placed randomly with minimum spacing.

    :param width: Terrain grid width.
    :type width: int
    :param height: Terrain grid height.
    :type height: int
    :param num_poles: Number of poles to place.
    :type num_poles: int
    :param pole_radius_range: Tuple of (min_radius, max_radius) in grid units.
    :type pole_radius_range: tuple[int, int]
    :param pole_height_range: Tuple of (min_height, max_height) in height units.
    :type pole_height_range: tuple[float, float]
    :param min_spacing: Minimum required distance (in grid cells) between pole centers.
    :type min_spacing: int
    :returns: The heightmap array with poles.
    :rtype: NDArray[np.float64]
    """
    terrain = np.zeros((width, height))
    pole_positions = []
    
    attempts = 0
    max_attempts = num_poles * 50
    
    while len(pole_positions) < num_poles and attempts < max_attempts:
        attempts += 1
        
        # Random pole center
        cx = np.random.randint(pole_radius_range[1], width - pole_radius_range[1])
        cy = np.random.randint(pole_radius_range[1], height - pole_radius_range[1])
        
        # Check spacing from existing poles
        valid = True
        for existing_cx, existing_cy, _ in pole_positions:
            dist = np.sqrt((cx - existing_cx)**2 + (cy - existing_cy)**2)
            if dist < min_spacing:
                valid = False
                break
        
        if valid:
            radius = np.random.uniform(*pole_radius_range)
            pole_height = np.random.uniform(*pole_height_range)
            pole_positions.append((cx, cy, radius))
            
            # Create pole
            for i in range(width):
                for j in range(height):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist <= radius:
                        # Smooth edges with a small falloff
                        if dist > radius - 1:
                            # Edge smoothing
                            blend = radius - dist
                            terrain[i, j] = max(terrain[i, j], pole_height * blend)
                        else:
                            terrain[i, j] = max(terrain[i, j], pole_height)
    
    return terrain

def generate_corridor_terrain(width: int, height: int, 
                              corridor_width: float = 0.85,
                              corridor_axis: str = 'y',
                              corridor_center: float = 0.5,
                              wall_height: float = 0.5) -> NDArray[np.float64]:
    """
    Generate flat terrain with thick walls creating a narrow corridor along a specified axis.

    The method estimates the number of cells needed to match the requested ``corridor_width``
    based on a fixed `WORLD_SIZE` of 10.0 meters.

    :param width: Terrain grid width.
    :type width: int
    :param height: Terrain grid height.
    :type height: int
    :param corridor_width: Desired width of the corridor in meters (e.g., 0.85m).
    :type corridor_width: float
    :param corridor_axis: Axis along which the corridor extends (``'x'`` or ``'y'``). The walls run perpendicular to this axis.
    :type corridor_axis: str
    :param corridor_center: Center of the corridor as a fraction (0-1) of the terrain size across the axis perpendicular to the corridor.
    :type corridor_center: float
    :param wall_height: Height of the walls.
    :type wall_height: float
    :returns: The heightmap array with the corridor structure.
    :rtype: NDArray[np.float64]
    """
    terrain = np.zeros((width, height))
    
    WORLD_SIZE = 10.0
    
    if corridor_axis == 'y':
        cell_size = WORLD_SIZE / width
        
        # Calculate cells needed
        corridor_cells_float = corridor_width / cell_size
        corridor_cells = max(1, int(np.round(corridor_cells_float)))
        
        center_i = int(corridor_center * width)
        
        # Define corridor boundaries
        corridor_half_cells = corridor_cells // 2
        start_i = max(0, center_i - corridor_half_cells)
        end_i = min(width, center_i + corridor_half_cells + (corridor_cells % 2))
        
        # Fill with walls
        terrain[:, :] = wall_height
        
        # Carve out corridor
        terrain[start_i:end_i, :] = 0.0
        
        actual_cells = end_i - start_i
        actual_width = actual_cells * cell_size
        
        print(f"[Corridor-Y] Request: {corridor_width:.2f}m → Actual: {actual_width:.2f}m ({actual_cells} cells)")
        
    elif corridor_axis == 'x':
        cell_size = WORLD_SIZE / height
        
        corridor_cells_float = corridor_width / cell_size
        corridor_cells = max(1, int(np.round(corridor_cells_float)))
        
        center_j = int(corridor_center * height)
        
        # Define corridor boundaries
        corridor_half_cells = corridor_cells // 2
        start_j = max(0, center_j - corridor_half_cells)
        end_j = min(height, center_j + corridor_half_cells + (corridor_cells % 2))
        
        terrain[:, :] = wall_height
        terrain[:, start_j:end_j] = 0.0
        
        actual_cells = end_j - start_j
        actual_width = actual_cells * cell_size
        
        print(f"[Corridor-X] Request: {corridor_width:.2f}m → Actual: {actual_width:.2f}m ({actual_cells} cells)")
    
    else:
        raise ValueError(f"corridor_axis must be 'x' or 'y', got: {corridor_axis}")
    
    return terrain

def generate_spiral_track(width: int, height: int,
                         start_radius: float = 0.5,
                         end_radius: float = 4.0,
                         num_turns: float = 2.0,
                         track_width: float = 0.8,
                         wall_height: float = 0.5) -> NDArray[np.float64]:
    """
    Generate a spiral track starting from the origin (center of the grid) and spiraling outward to the edge.

    The path is carved out of high walls, leaving a flat, navigable track based on an 
    Archimedean spiral ($r = a + b\theta$).

    :param width: Grid width.
    :type width: int
    :param height: Grid height.
    :type height: int
    :param start_radius: Starting radius of the spiral in meters (inner radius).
    :type start_radius: float
    :param end_radius: Ending radius of the spiral in meters (outer edge).
    :type end_radius: float
    :param num_turns: Number of complete $360^\circ$ rotations the spiral makes.
    :type num_turns: float
    :param track_width: Width of the flat track region in meters.
    :type track_width: float
    :param wall_height: Height of the walls surrounding the track.
    :type wall_height: float
    :returns: The heightmap array with the spiral track structure.
    :rtype: NDArray[np.float64]
    """
    terrain = np.zeros((width, height))
    
    WORLD_SIZE = 10.0
    cell_size = WORLD_SIZE / width
    
    # Fill with walls
    terrain[:, :] = wall_height
    
    # Track width parameters
    half_width = track_width / 2
    
    # Grid center (corresponds to world origin 0, 0)
    center_i = width // 2
    center_j = height // 2
    
    # Total angle to cover
    total_angle = 2 * np.pi * num_turns
    
    # For each grid cell, determine if it's on the spiral
    for i in range(width):
        for j in range(height):
            # Convert to world coordinates (centered at origin)
            x = (i - center_i) * cell_size
            y = (j - center_j) * cell_size
            
            # Calculate polar coordinates from origin
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            # Normalize angle to [0, 2π]
            if theta < 0:
                theta += 2 * np.pi
            
            # For each possible turn the spiral makes, check if point is on spiral
            for turn_num in range(int(np.ceil(num_turns)) + 1):
                # Angle with turn offset
                angle = theta + (turn_num * 2 * np.pi)
                
                # Only consider angles within total spiral range
                if angle > total_angle:
                    continue
                
                # Calculate expected radius at this angle on the spiral
                # r = start_radius + (end_radius - start_radius) * (angle / total_angle)
                progress = angle / total_angle
                expected_radius = start_radius + (end_radius - start_radius) * progress
                
                # Check if current point's radius matches expected radius (within track width)
                if abs(r - expected_radius) <= half_width:
                    terrain[i, j] = 0.0
                    break  # Found valid spiral section
    
    # Calculate actual track width in cells
    corridor_cells = max(1, int(np.round(track_width / cell_size)))
    actual_width = corridor_cells * cell_size
    
    # Calculate approximate spiral length
    avg_radius = (start_radius + end_radius) / 2
    spiral_length = avg_radius * total_angle
    
    print(f"[Spiral Track] Origin→Outward, "
          f"R: {start_radius:.1f}→{end_radius:.1f}m, "
          f"Turns: {num_turns:.1f}, "
          f"Width: {actual_width:.2f}m, "
          f"Length: ~{spiral_length:.1f}m")
    
    return terrain

def generate_terrain(terrain_type: str, 
                     width: int = 50, 
                     height: int = 50, 
                     ensure_flat_spawn: bool = True, 
                     **kwargs) -> NDArray[np.float64]:
    """
    Main entry function to dispatch terrain generation based on the type requested.

    This function calls the appropriate generation helper and applies the 
    `ensure_flat_spawn_zone` utility if required. 

    :param terrain_type: Specifies the type of terrain to generate: ``'wavy'``, ``'perlin'``, ``'stairs'``, 
        ``'flat'``, ``'incline'``, ``'craters'``, ``'poles'``, ``'flat_with_incline'``, 
        ``'flat_with_ramp'``, ``'corridor'``, or ``'spiral'``.
    :type terrain_type: str
    :param width: Terrain grid width (number of cells).
    :type width: int
    :param height: Terrain grid height (number of cells).
    :type height: int
    :param ensure_flat_spawn: If True, calls :py:func:`~ensure_flat_spawn_zone` to flatten the center area.
    :type ensure_flat_spawn: bool
    :param kwargs: Additional keyword arguments passed to the specific terrain generation function (e.g., `angle`, `num_craters`).
    :type kwargs: dict
    :returns: The generated heightmap array.
    :rtype: NDArray[np.float64]
    :raises ValueError: If an unknown `terrain_type` is provided.
    """
    if terrain_type == 'wavy':
        scale = kwargs.get('scale', 0.3)
        amp = kwargs.get('amp', 0.15)
        terrain = generate_wavy_field(width, height, scale, amp)

    elif terrain_type == 'perlin':
        scale = kwargs.get('scale', 0.1)
        terrain = generate_perlin_noise(width, height, scale)

    elif terrain_type == 'stairs':
        step_height = kwargs.get('step_height', 0.08)
        pixels_per_step = kwargs.get('pixels_per_step', 5)
        terrain = generate_stairs_terrain(width, height, step_height, pixels_per_step)

    elif terrain_type == 'flat':
        terrain = generate_flat_terrain(width, height)

    elif terrain_type == 'incline':
        angle = kwargs.get('angle', 20)
        direction = kwargs.get('direction', 'y')
        terrain = generate_incline_terrain(width, height, angle, direction)
    
    elif terrain_type == 'flat_with_incline':
        incline_start_x = kwargs.get('incline_start_x', 0.3)
        incline_end_x = kwargs.get('incline_end_x', 0.7)
        incline_start_y = kwargs.get('incline_start_y', 0.3)
        incline_end_y = kwargs.get('incline_end_y', 0.7)
        angle = kwargs.get('angle', 15)
        direction = kwargs.get('direction', 'x')
        terrain = generate_flat_with_incline(width, height,
                                         incline_start_x, incline_end_x,
                                         incline_start_y, incline_end_y,
                                         angle, direction)
    
    elif terrain_type == 'flat_with_ramp':
        ramp_center_x = kwargs.get('ramp_center_x', 0.5)
        ramp_center_y = kwargs.get('ramp_center_y', 0.5)
        ramp_length = kwargs.get('ramp_length', 0.3)
        ramp_width = kwargs.get('ramp_width', 0.3)
        ramp_height = kwargs.get('ramp_height', 0.2)
        direction = kwargs.get('direction', 'x')
        smooth_edges = kwargs.get('smooth_edges', True)
        terrain = generate_flat_with_ramp(width, height,
                                      ramp_center_x, ramp_center_y,
                                      ramp_length, ramp_width,
                                      ramp_height, direction, smooth_edges)
    
    elif terrain_type == 'craters':
        num_craters = kwargs.get('num_craters', 25)
        crater_radius_range = kwargs.get('crater_radius_range', (3, 5))
        crater_depth_range = kwargs.get('crater_depth_range', (0.10, 0.20))
        terrain = generate_crater_terrain(width, height, num_craters, 
                                      crater_radius_range, crater_depth_range)
    
    elif terrain_type == 'poles':
        num_poles = kwargs.get('num_poles', 75)
        pole_radius_range = kwargs.get('pole_radius_range', (1, 1))
        pole_height_range = kwargs.get('pole_height_range', (1.5, 2.5))
        min_spacing = kwargs.get('min_spacing', 3)
        terrain = generate_poles_terrain(width, height, num_poles, pole_radius_range,
                                     pole_height_range, min_spacing)
    elif terrain_type == 'corridor':
        corridor_width = kwargs.get('corridor_width', 0.60)
        corridor_axis = kwargs.get('corridor_axis', 'y')
        corridor_center = kwargs.get('corridor_center', 0.5)
        wall_height = kwargs.get('wall_height', 0.5)
        terrain = generate_corridor_terrain(width, height, corridor_width,
                                            corridor_axis, corridor_center, wall_height)
    
    elif terrain_type == 'spiral':
        start_radius = kwargs.get('start_radius', 0.5)
        end_radius = kwargs.get('end_radius', 4.0)
        num_turns = kwargs.get('num_turns', 2.0)
        track_width = kwargs.get('track_width', 0.8)
        wall_height = kwargs.get('wall_height', 0.5)
        
        terrain = generate_spiral_track(
            width, height,
            start_radius=start_radius,
            end_radius=end_radius,
            num_turns=num_turns,
            track_width=track_width,
            wall_height=wall_height
        )

    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")
     
    if ensure_flat_spawn:
        spawn_center_x = kwargs.get('spawn_center_x', 0.5)
        spawn_center_y = kwargs.get('spawn_center_y', 0.5)
        spawn_radius = kwargs.get('spawn_radius', 0.085)
        terrain = ensure_flat_spawn_zone(terrain, 
                                         spawn_center_x, 
                                         spawn_center_y, 
                                         spawn_radius)
        
    return terrain