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