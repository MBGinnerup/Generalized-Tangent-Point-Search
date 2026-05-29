import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import numpy as np
from TPS_2D.maps.core.real_world_map_utils import get_environment

def real_world_map(voxel_size, map_name):
    """
    Create a 2D occupancy map from a 3D environment.

    Loads environment and terrain data, removes terrain regions from
    the environment layer, and projects the result onto the XY-plane.

    Parameters:
        voxel_size (float): Resolution used for voxelization.
        map_name (str): Name of the environment folder.

    Returns:
        ndarray: 2D binary occupancy grid (1 = occupied, 0 = free).
    """
    # Load 3D building and boundary voxel grids
    environment, terrain = get_environment(voxel_size, map_name)

    # Remove boundary regions from building map
    environment[terrain] = 0

    # Project to 2D occupancy grid (XY-plane)
    return np.any(environment, axis=2).astype(np.int32)