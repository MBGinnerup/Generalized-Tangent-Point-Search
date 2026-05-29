import numpy as np

def random_start(grid):
    """
    Return a random free cell from the grid.

    Parameters:
        grid (ndarray): Occupancy grid where 0 indicates free space.

    Returns:
        tuple: Random coordinate in free space.
    """
    return tuple(np.argwhere(grid == 0)[np.random.randint(np.sum(grid == 0))])

def random_goal(grid, start, min_dist = 50):
    """
    Select a random goal cell at a minimum distance from the start.

    Parameters:
        grid (ndarray): Occupancy grid where 0 indicates free space.
        start (tuple): Start coordinate.
        min_dist (float): Minimum Euclidean distance from start.

    Returns:
        tuple: Random goal coordinate satisfying min_dist.
    """
    free_cells = np.argwhere(grid == 0)

    start = np.array(start)

    dists = np.linalg.norm(free_cells - start, axis=1)

    valid = free_cells[dists >= min_dist]

    if len(valid) == 0:
        raise ValueError("No free cell satisfies min_dist")

    idx = np.random.randint(len(valid))
    return tuple(valid[idx])

def random_start_environment(environment, terrain, z_height = 1):
    """
    Sample a random valid start position in a 3D environment.

    The function selects a free (x, y) location from the 2D projection of
    the environment and assigns a z-height above the highest structure.

    Parameters:
        environment (ndarray): 3D occupancy grid of the environment.
        terrain (ndarray): 3D terrain mask.
        z_height (int): Vertical offset above the highest building point.

    Returns:
        tuple: (x, y, z) start coordinate in free space.
    """
    environment_copy = environment.copy()
    environment_copy[terrain] = 0
    xy = np.any(environment_copy, axis=2).astype(np.int32)

    free_xy = np.argwhere(xy == 0)

    idx = np.random.randint(len(free_xy))
    x, y = free_xy[idx]

    z = np.max(np.argwhere(environment[x, y, :])) + z_height

    return (x, y, z)

def random_goal_environment(environment, terrain, start, min_dist = 50, z_height = 1):
    """
    Sample a random valid goal position in a 3D environment.

    The function selects a free (x, y) location that is at least a given
    distance from the start position and assigns a z-height above the
    highest structure at that location.

    Parameters:
        environment (ndarray): 3D occupancy grid of environment.
        terrain (ndarray): 3D terrain mask.
        start (tuple): Start coordinate (x, y, z).
        min_dist (float): Minimum Euclidean distance from start (2D).
        z_height (int): Vertical offset above highest building point.

    Returns:
        tuple: (x, y, z) goal coordinate in free space.
    """
    environment_copy = environment.copy()
    environment_copy[terrain] = 0
    xy = np.any(environment_copy, axis=2).astype(np.int32)

    free_xy = np.argwhere(xy == 0)

    start = np.array(start[:2])  

    dists = np.linalg.norm(free_xy - start, axis=1)

    valid = free_xy[dists >= min_dist]

    if len(valid) == 0:
        raise ValueError("No free cell satisfies min_dist")

    idx = np.random.randint(len(valid))
    x, y = valid[idx]

    z = np.max(np.argwhere(environment[x, y, :])) + z_height

    return (x, y, z)