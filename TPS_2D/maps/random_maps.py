import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import numpy as np
from TPS_2D.maps.core import random_map_utils as utils

def random_map(size, dense, distribution, relative_size, small, large, connectivity_guarantee=False, clustered=False, max_iter=500):
    """
    Generate a random binary obstacle map with controllable obstacle density,
    obstacle size distribution, and optional connectivity guarantees.

    The environment is first generated on a coarse grid where each cell corresponds
    to the smallest obstacle resolution determined from `relative_size`. Obstacles
    of varying sizes are then iteratively placed until the target density is reached.
    Finally, the map is upscaled to the requested resolution.

    The generator can optionally ensure that a collision-free path exists between
    the upper-left and lower-right corners by repeatedly generating and validating
    the map using a BFS-based connectivity check.

    Features:
        - Adjustable obstacle density.
        - Probabilistic selection between small and large obstacle sizes.
        - Optional clustered obstacle placement.
        - Automatic removal of isolated free-space regions.
        - Optional connectivity guarantee between start and goal corners.
        - Obstacle erosion between failed generation attempts to improve convergence.

    Parameters:
        size (int):
            Final map size as a square occupancy grid of shape (size x size).
        dense (float):
            Desired obstacle density in the range [0, 1].
        distribution (float):
            Probability of selecting a small obstacle.
            Large obstacle probability is computed as (1 - distribution).
        relative_size (int):
            Relative obstacle scaling factor used to determine the coarse grid
            resolution and smallest obstacle size.
        small (tuple[int, int]):
            Inclusive range (min, max) for randomly generated small obstacle sizes.
        large (tuple[int, int]):
            Inclusive range (min, max) for randomly generated large obstacle sizes.
        connectivity_guarantee (bool, optional):
            If True, repeatedly generates maps until a valid path exists between
            opposite corners. Default is False.
        clustered (bool, optional):
            If True, obstacles are placed in clusters instead of uniformly.
            Default is False.
        max_iter (int, optional):
            Maximum number of generation attempts when connectivity checking
            is enabled. Default is 500.
    Returns:
        np.ndarray or None:
            A binary occupancy grid of type np.int32 where:
                0 = free space
                1 = obstacle
            Returns None if no valid connected map could be generated within
            `max_iter` attempts.
    """
    if not (0 <= dense <= 1):
        raise ValueError("Dense must be between 0 and 1.")

    if not (0 <= distribution <= 1):
        raise ValueError("Choose a distribution between 0 and 1")

    if not (1 <= relative_size <= size) or not isinstance(relative_size, int):
        raise ValueError("Choose an integer between 1 and", size)

    rng = np.random.default_rng()

    dense_temp = 0

    l = 2

    # Distrubution of the obsatcle sizes:
    p_s = distribution
    p_l = 1 - p_s

    # Define smallest obstacle size which can fit inside the grid from the choosen reletive size:
    for k in range(relative_size, 0, -1):
        if size % k == 0:
            obs_smallest_size = k
            break 
        
    # Create empty map with cells responding to smallest obstacle size:
    M = size // obs_smallest_size
    map_obs_cells = np.zeros((M, M))

    # Generate map with obsatcles with a guaranteed path between two corners
    for iter in range(max_iter):

        map_obs_cells[map_obs_cells != 0] += 1

        if connectivity_guarantee:
            # Add ones to corners to make sure no obstacle will be added:
            map_obs_cells[:2,:2] = 1
            map_obs_cells[-2:,-2:] = 1

        map_obs_cells = np.pad(map_obs_cells, pad_width=1, mode='constant', constant_values=1)

        while dense > dense_temp:

            # Find size of obsatcle to be placed:
            small_or_large = rng.choice(['small', 'large'], p=[p_s, p_l])
            if small_or_large == 'small':
                obs_size = rng.integers(small[0], small[1] + 1)
            else:
                obs_size = rng.integers(large[0], large[1] + 1)

            # Place obstacle in map and update dense:
            if clustered:
                map_obs_cells, dense_temp = utils.place_obsatcle_clusters(map_obs_cells, obs_size, l, dense, rng)
                if dense_temp == -1:
                    return None
            else:
                map_obs_cells, dense_temp = utils.place_obsatcle(map_obs_cells, obs_size, l, dense, rng)

            # Check if there is a region zeros surrounded by obstacles and update dense to check condition:
            if connectivity_guarantee:
                fill_idx = utils.surrounded_regions_empty_corners(map_obs_cells)
                if len(fill_idx) > 0:
                    map_obs_cells[fill_idx[:,0], fill_idx[:,1]] = l
                    dense_temp = utils.calculate_dense(map_obs_cells[1:-1, 1:-1])
                    if dense_temp >= dense:
                        break
            else:
                fill_idx = utils.surrounded_regions(map_obs_cells)
                if len(fill_idx) > 0:
                    map_obs_cells[fill_idx[:,0], fill_idx[:,1]] = l
                    dense_temp = utils.calculate_dense(map_obs_cells[1:-1, 1:-1])
                    if dense_temp >= dense:
                        break
            
            # Add 1 to label to create new obsatcle:
            l += 1

        map_obs_cells = map_obs_cells[1:-1, 1:-1]

        map_obs_cells[map_obs_cells != 0] -= 1

        if connectivity_guarantee:
            # Search for a path:
            start = (0,0)
            goal = ((size//obs_smallest_size)-1,(size//obs_smallest_size)-1)
            exist = utils.lazy_BFS(map_obs_cells, start, goal)
            if exist:
                break

            if iter == (max_iter - 1):
                return None

            # Peform erosion to shrink obstacles before placing new obstacles
            map_obs_cells = utils.erode_obs(map_obs_cells)

            dense_temp = utils.calculate_dense(map_obs_cells)
        else:
            break

    # Upscale relative map to the scale of the wanted size:
    map = utils.upscale(map_obs_cells, obs_smallest_size)

    map = np.clip(map,0,1)

    return map.astype(np.int32)