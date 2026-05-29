import numpy as np
from scipy.ndimage import label, generate_binary_structure, binary_erosion, binary_dilation
from collections import deque

def neighbors_26connectivity(i, j, k, grid):
    """
    Return valid 26-connected neighboring cells for a 3D grid position.

    Checks all surrounding voxels (including diagonals in all directions)
    around the cell `(i, j, k)` and returns those that are free
    (`grid == 0`).

    Args:
        i (int): X/row index of the current cell.
        j (int): Y/column index of the current cell.
        k (int): Z/depth index of the current cell.
        grid (np.ndarray): 3D occupancy grid where free cells are marked as 0.

    Returns:
        list[tuple[int, int, int]]: List of valid neighboring coordinates
        as `(i, j, k)` tuples.
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue 
                if grid[i + dx, j + dy, k + dz] == 0:
                    neighbors.append((i + dx, j + dy, k + dz))
    return neighbors

def calculate_dense(grid):
    """
    Calculate the density of occupied cells in a grid.
    
    Returns the ratio of cells with values greater than 1
    relative to the total number of cells.
    """
    return np.count_nonzero(grid > 1) / (len(grid)**3)

def place_obsatcle(map, obs_size, l, dense, rng):
    """
    Place and grow a random obstacle in an occupancy grid.

    A random free cell is selected as the obstacle seed and assigned
    the label `l`. The obstacle is then expanded by iteratively adding
    random 26-connected neighboring free cells until either the desired
    obstacle size is reached or the target grid density is met.

    Parameters:
        map (np.ndarray): Occupancy grid.
        obs_size (int): Maximum number of cells in the obstacle.
        l (int): Label/value assigned to the obstacle cells.
        dense (float): Target obstacle density threshold.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple: Updated grid and resulting density.
    """
    new_obs = []

    empty_positions = np.argwhere(map == 0)

    position = rng.integers(len(empty_positions))
    x, y, z = empty_positions[position]

    map[x, y, z] = l

    new_obs.append((x, y, z))

    dense_temp = calculate_dense(map[1:-1, 1:-1, 1:-1])
    if dense_temp >= dense:
        return map, dense_temp

    # Expand obstacle:
    for _ in range(obs_size - 1):

        i, j, k = new_obs[rng.integers(len(new_obs))]

        neighbors = neighbors_26connectivity(i,j, k, map)

        if len(neighbors) == 0:
            continue
        
        # Choose random neighbor of the candidates:
        idx = rng.integers(len(neighbors))
        ni, nj, nk = neighbors[idx]

        map[ni, nj, nk] = l

        new_obs.append((ni, nj, nk))

        dense_temp = calculate_dense(map[1:-1, 1:-1, 1:-1])
        if dense_temp >= dense:
            break
    
    return map, dense_temp

def place_obsatcle_clusters(map, obs_size, l, dense, rng):
    """
    Place a clustered obstacle in an occupancy grid while maintaining
    separation from existing obstacles.

    A random free cell is selected outside a dilated mask of existing
    obstacles to avoid placing clusters too close together. The obstacle
    is then expanded through random 26-connected neighboring cells until
    the maximum size or target density is reached.

    Parameters:
        map (np.ndarray): Occupancy grid.
        obs_size (int): Maximum number of cells in the obstacle cluster.
        l (int): Label/value assigned to the obstacle cells.
        dense (float): Target obstacle density threshold.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple: Updated grid and resulting density. Returns `-1` as density
        if no valid placement position exists.
    """
    new_obs = []
    
    # Dilated mask for existing obsatcles/clusters
    obstacle_mask = (map > 1)
    obstacle_dilated = binary_dilation(obstacle_mask, structure=np.ones((3,3,3)))
    dilated_mask = obstacle_dilated | (map != 0)

    empty_positions = np.argwhere(~dilated_mask)
    if len(empty_positions) == 0:
        return map, -1 

    position = rng.integers(len(empty_positions))
    x, y, z = empty_positions[position]
 
    map[x, y, z] = l

    new_obs.append((x, y, z))

    dilated_mask[x, y, z] = True
    
    dense_temp = calculate_dense(map[1:-1,1:-1,1:-1])
    if dense_temp >= dense:
        return map, dense_temp
    
    # Expand obstacle:
    for _ in range(obs_size - 1):

        i, j, k = new_obs[rng.integers(len(new_obs))]
        
        neighbors = neighbors_26connectivity(i, j, k, dilated_mask)
        
        if len(neighbors) == 0:
            continue
        
        # Choose random neighbor of the candidates:
        idx = rng.integers(len(neighbors))
        ni, nj, nk = neighbors[idx]

        map[ni, nj, nk] = l

        new_obs.append((ni, nj, nk))

        dilated_mask[ni, nj, nk] = True

        dense_temp = calculate_dense(map[1:-1, 1:-1, 1:-1])
        if dense_temp >= dense:
            break
    
    return map, dense_temp

def surrounded_regions(grid):
    """
    Identify enclosed free-space regions in an occupancy grid.

    Finds connected components of free cells (`grid == 0`) using
    6-connectivity and returns the coordinates of regions that do not
    touch the grid boundary, i.e. fully surrounded regions.

    Parameters:
        grid (np.ndarray): Occupancy grid.

    Returns:
        np.ndarray: Array of coordinates for enclosed free-space cells.
    """
    zero_mask = (grid == 0)

    structure = generate_binary_structure(3, 1)

    # Label connected components
    labeled_array, num_features = label(zero_mask, structure=structure)

    # Find labels touching the boundary 
    boundary_labels = set()
    boundary_slices = [
        labeled_array[1, 1:-1, 1:-1],  
        labeled_array[-2, 1:-1, 1:-1],
        labeled_array[1:-1, 1, 1:-1],  
        labeled_array[1:-1, -2, 1:-1],
        labeled_array[1:-1, 1:-1, 1], 
        labeled_array[1:-1, 1:-1, -2]] 
    for face in boundary_slices:
        boundary_labels.update(np.unique(face))
    
    # Remove boundary labels which is zero (obstacles)
    boundary_labels.discard(0)  

    all_labels = set(range(1, num_features + 1))

    # Find surrounded regions = not touching boundary
    surrounded_labels = all_labels - boundary_labels

    points_to_fill = []
    for region_id in surrounded_labels:
        coords = np.argwhere(labeled_array == region_id)
        points_to_fill.extend(coords)

    return np.array(points_to_fill, dtype=int)

def surrounded_regions_empty_corners(grid):
    """
    Identify enclosed free-space regions while preserving empty corners.

    Finds connected components of free cells (`grid == 0`) using
    6-connectivity and returns coordinates of regions that do not touch
    the grid boundary or specified corner areas.

    Parameters:
        grid (np.ndarray): Occupancy grid.

    Returns:
        np.ndarray: Array of coordinates for enclosed free-space cells.
    """
    zero_mask = (grid == 0)

    structure = generate_binary_structure(3, 1)

    # Label connected components
    labeled_array, num_features = label(zero_mask, structure=structure)

    # Find labels touching the boundary 
    boundary_labels = set()
    boundary_slices = [
        labeled_array[1, 1:-1, 1:-1],     
        labeled_array[-2, 1:-1, 1:-1], 
        labeled_array[1:-1, 1, 1:-1],    
        labeled_array[1:-1, -2, 1:-1],    
        labeled_array[1:-1, 1:-1, 1],    
        labeled_array[1:-1, 1:-1, -2],    
        labeled_array[2:4,2:4,2:4],      
        labeled_array[-4:-2,-4:-2,-4:-2]] 
    for face in boundary_slices:
        boundary_labels.update(np.unique(face))
    
    # Remove boundary labels which is zero (obstacles)
    boundary_labels.discard(0)  

    all_labels = set(range(1, num_features + 1))

    # Find surrounded regions = not touching boundary
    surrounded_labels = all_labels - boundary_labels

    points_to_fill = []
    for region_id in surrounded_labels:
        coords = np.argwhere(labeled_array == region_id)
        points_to_fill.extend(coords)

    return np.array(points_to_fill, dtype=int)

def upscale(grid, factor):
    """
    Upscale a grid by repeating each cell along all dimensions.

    Parameters:
        grid (np.ndarray): Input grid.
        factor (int): Scaling factor.

    Returns:
        np.ndarray: Upscaled grid.
    """
    return np.repeat(np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1), factor, axis=2)

def lazy_BFS(grid, start, goal):
    """
    Perform Breadth-First Search (BFS) on a grid to find a path from start to goal.

    Explores the grid in 6-connectivity, avoiding obstacles.

    Parameters:
        grid (np.ndarray): Occupancy grid.
        start (tuple[int, int]): Starting cell (row, col).
        goal (tuple[int, int]): Target cell (row, col).

    Returns:
        tuple:
            bool: True if a path is found, otherwise False.
    """
    visited = set()
    visited.add(start)

    parent = {start: None}

    queue = deque([start])
  
    connections = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    shape = grid.shape

    while queue:

        node = queue.popleft()

        if node == goal:
            return True
        
        # Check neighbors:
        for c in connections:

            nx = node[0] + c[0]
            ny = node[1] + c[1]
            nz = node[2] + c[2]

            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                # Check if the neighbor is free and not visited:
                if grid[nx, ny, nz] == 0 and (nx, ny, nz) not in visited:
                    visited.add((nx, ny, nz))
                    parent[(nx, ny, nz)] = node 
                    queue.append((nx, ny, nz))
    
    return False

def erode_obs(grid):
    """
    Perform morphological erosion on obstacle regions in a grid.

    Applies binary erosion to all obstacle cells (`grid > 0`) using a
    6-connectivity structuring element, to shrink obstacles.

    Parameters:
        grid (np.ndarray): Input occupancy grid.

    Returns:
        np.ndarray: Grid with eroded obstacles.
    """
    obs_mask = (grid > 0)

    structure = generate_binary_structure(3,1)

    eroded_obs = binary_erosion(obs_mask, structure=structure)

    grid_eroded = np.zeros_like(grid)
    grid_eroded[eroded_obs] = grid[eroded_obs]

    return grid_eroded