import numpy as np
import noise

def ensure_flat_spawn_zone(terrain, 
                           spawn_center_x  = 0.5, 
                           spawn_center_y = 0.5, 
                           spawn_radius = 0.15):
    """
    Ensure a flat area in the terrain for robot spawning
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


# Terrain generation functions
def generate_wavy_field(width, height, scale=0.5, amp=0.5):
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = amp * np.sin(scale*i) * np.cos(scale*j)
    return world

def generate_perlin_noise(width, height, scale=0.1):
    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = noise.pnoise2(i*scale, j*scale, octaves=2)
    return world

def generate_stairs_terrain(width, height, step_height=0.02, pixels_per_step=5):
    """Generate staircase terrain"""
    terrain = np.zeros((width, height))
    for i in range(width):
        terrain[i, :] = (i // pixels_per_step) * step_height
    return terrain

def generate_flat_terrain(width, height):
    """Generate flat terrain"""
    return np.zeros((width, height))

def generate_flat_with_incline(width, height, 
                               incline_start_x=0.3, incline_end_x=0.7,
                               incline_start_y=0.3, incline_end_y=0.7,
                               angle=15, direction='x'):
    """
    Generate flat terrain with a localized incline/ramp
    
    Args:
        width: terrain grid width
        height: terrain grid height
        incline_start_x: start position of incline as fraction (0-1) along width
        incline_end_x: end position of incline as fraction (0-1) along width
        incline_start_y: start position of incline as fraction (0-1) along height
        incline_end_y: end position of incline as fraction (0-1) along height
        angle: slope angle in degrees
        direction: 'x', 'y', or 'diagonal' - direction of slope within the incline region
    
    Returns:
        terrain: height field array
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

def generate_incline_terrain(width, height, angle=10, direction='x'):
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

def generate_flat_with_ramp(width, height,
                            ramp_center_x=0.5, ramp_center_y=0.5,
                            ramp_length=0.3, ramp_width=0.3,
                            ramp_height=0.2, direction='x',
                            smooth_edges=True):
    """
    Generate flat terrain with a smooth ramp at specified position
    
    Args:
        width: terrain grid width
        height: terrain grid height
        ramp_center_x: center x position as fraction (0-1)
        ramp_center_y: center y position as fraction (0-1)
        ramp_length: length of ramp as fraction of terrain size
        ramp_width: width of ramp as fraction of terrain size
        ramp_height: maximum height of ramp (in height units)
        direction: 'x' (ramp goes up in +x), '-x', 'y', '-y'
        smooth_edges: if True, smooth the transition to flat terrain
    
    Returns:
        terrain: height field array
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

def generate_crater_terrain(width, height, num_craters=20, crater_radius_range=(2, 5), 
                            crater_depth_range=(0.05, 0.15)):
    """
    Generate terrain with random craters (small depressions)
    
    Args:
        width: terrain grid width
        height: terrain grid height
        num_craters: number of craters to generate
        crater_radius_range: tuple of (min_radius, max_radius) in grid units
        crater_depth_range: tuple of (min_depth, max_depth) in height units
    
    Returns:
        terrain: height field array with craters
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


def generate_poles_terrain(width, height, num_poles=8, pole_radius_range=(3, 6), 
                           pole_height_range=(0.3, 0.6), min_spacing=8):
    """
    Generate terrain with large cylindrical poles/pillars
    
    Args:
        width: terrain grid width
        height: terrain grid height
        num_poles: number of poles to place
        pole_radius_range: tuple of (min_radius, max_radius) in grid units
        pole_height_range: tuple of (min_height, max_height)
        min_spacing: minimum distance between pole centers
    
    Returns:
        terrain: height field array with poles
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

def generate_corridor_terrain(width, height, 
                              corridor_width=0.85,
                              corridor_axis='y',
                              corridor_center=0.5,
                              wall_height=0.5):
    """
    Generate narrow corridor with walls along specified axis
    """
    terrain = np.zeros((width, height))
    
    WORLD_SIZE = 10.0
    
    if corridor_axis == 'y':
        cell_size = WORLD_SIZE / width  # 10m / 50 cells = 0.2m per cell
        
        # Calculate cells needed (no forced rounding to even!)
        corridor_cells_float = corridor_width / cell_size
        corridor_cells = max(1, int(np.round(corridor_cells_float)))  # At least 1 cell
        
        actual_width = corridor_cells * cell_size
        
        center_i = int(corridor_center * width)
        
        # Handle both odd and even cell counts
        if corridor_cells % 2 == 0:
            # Even number of cells - symmetric split
            corridor_half_cells = corridor_cells // 2
            start_i = max(0, center_i - corridor_half_cells)
            end_i = min(width, center_i + corridor_half_cells)
        else:
            # Odd number of cells - center cell + symmetric sides
            corridor_half_cells = corridor_cells // 2
            start_i = max(0, center_i - corridor_half_cells)
            end_i = min(width, center_i + corridor_half_cells + 1)  # +1 for center cell
        
        # Fill with walls
        terrain[:, :] = wall_height
        
        # Carve out corridor
        terrain[start_i:end_i, :] = 0.0
        
        actual_cells = end_i - start_i
        actual_width = actual_cells * cell_size
        
        print(f"[Corridor-Y] Request: {corridor_width:.2f}m → Actual: {actual_width:.2f}m ({actual_cells} cells)")
        
    elif corridor_axis == 'x':
        cell_size = WORLD_SIZE / height  # 10m / 50 cells = 0.2m per cell
        
        corridor_cells_float = corridor_width / cell_size
        corridor_cells = max(1, int(np.round(corridor_cells_float)))
        
        actual_width = corridor_cells * cell_size
        
        center_j = int(corridor_center * height)
        
        if corridor_cells % 2 == 0:
            corridor_half_cells = corridor_cells // 2
            start_j = max(0, center_j - corridor_half_cells)
            end_j = min(height, center_j + corridor_half_cells)
        else:
            corridor_half_cells = corridor_cells // 2
            start_j = max(0, center_j - corridor_half_cells)
            end_j = min(height, center_j + corridor_half_cells + 1)
        
        terrain[:, :] = wall_height
        terrain[:, start_j:end_j] = 0.0
        
        actual_cells = end_j - start_j
        actual_width = actual_cells * cell_size
        
        print(f"[Corridor-X] Request: {corridor_width:.2f}m → Actual: {actual_width:.2f}m ({actual_cells} cells)")
    
    else:
        raise ValueError(f"corridor_axis must be 'x' or 'y', got: {corridor_axis}")
    
    return terrain

def generate_spiral_track(width, height,
                         start_radius=0.5,
                         end_radius=4.0,
                         num_turns=2.0,
                         track_width=0.8,
                         wall_height=0.5):
    """
    Generate a spiral track starting from origin (0, 0)
    
    Args:
        width: grid width
        height: grid height
        start_radius: starting radius in meters (should be small, like 0.5m)
        end_radius: ending radius in meters (outer edge, like 4.0m)
        num_turns: number of complete 360° rotations
        track_width: width of the track in meters
        wall_height: height of walls
    
    Returns:
        terrain: height field with spiral track
        
    The spiral always starts near (0, 0) where the robot spawns
    and spirals outward to the edge of the terrain.
    
    Examples:
        # Gentle outward spiral (2 turns)
        start_radius=0.5, end_radius=4.0, num_turns=2.0
        
        # Tight spiral (3 turns)
        start_radius=0.3, end_radius=4.5, num_turns=3.0
        
        # Wide gentle spiral (1.5 turns)
        start_radius=0.8, end_radius=3.5, num_turns=1.5
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
            # (handles the multi-turn wrapping)
            for turn_num in range(int(np.ceil(num_turns)) + 1):
                # Angle with turn offset
                angle = theta + (turn_num * 2 * np.pi)
                
                # Only consider angles within total spiral range
                if angle > total_angle:
                    continue
                
                # Calculate expected radius at this angle on the spiral
                # Archimedean spiral: r = a + b * theta
                # We want: r(0) = start_radius, r(total_angle) = end_radius
                # So: r = start_radius + (end_radius - start_radius) * (angle / total_angle)
                progress = angle / total_angle  # 0 to 1
                expected_radius = start_radius + (end_radius - start_radius) * progress
                
                # Check if current point's radius matches expected radius (within track width)
                if abs(r - expected_radius) <= half_width:
                    terrain[i, j] = 0.0
                    break  # Found valid spiral section
    
    # Calculate actual track width in cells
    corridor_cells = max(1, int(np.round(track_width / cell_size)))
    actual_width = corridor_cells * cell_size
    
    # Calculate approximate spiral length
    # Arc length of spiral ≈ (avg_radius) * total_angle
    avg_radius = (start_radius + end_radius) / 2
    spiral_length = avg_radius * total_angle
    
    print(f"[Spiral Track] Origin→Outward, "
          f"R: {start_radius:.1f}→{end_radius:.1f}m, "
          f"Turns: {num_turns:.1f}, "
          f"Width: {actual_width:.2f}m, "
          f"Length: ~{spiral_length:.1f}m")
    
    return terrain

def generate_terrain(terrain_type, 
                     width=50, 
                     height=50, 
                     ensure_flat_spawn=True, 
                     **kwargs):
    """
    Generate terrain based on type
    
    Args:
        terrain_type: 'wavy', 'perlin', 'stairs', 'flat', 'incline', 'craters', 'poles',
                     'flat_with_incline', 'flat_with_ramp'
        width: terrain grid width
        height: terrain grid height
        **kwargs: additional parameters for specific terrain types
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
        start_radius = kwargs.get('start_radius', 0.5)    # Start small (near origin)
        end_radius = kwargs.get('end_radius', 4.0)        # End large (at edge)
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