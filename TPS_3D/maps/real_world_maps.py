import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import numpy as np
from TPS_3D.maps.core.real_world_map_utils import get_array

def real_world_map(voxel_size, map_name, pad = 3, fill = True):
    """
    Load environment and terrain data for a 3D environment.

    Converts both OBJ files into voxel grids and ensures that
    the terrain grid has the same height as the environment grid.

    Parameters:
        voxel_size (float): Resolution used for voxelization.
        map_name (str): Name of the environment folder.
        pad (int): Number of empty layers added above the environment.
        fill (bool): If True, fills empty space below occupied voxels.

    Returns:
        tuple: environment grid and terrain grid.
    """
    # Load voxelized terrain and environment maps
    terrain = get_array(voxel_size, map_name, 'terrain', pad, fill)
    environment = get_array(voxel_size, map_name, 'environment', pad, fill)

    environment_z = environment.shape[2]
    terrain_z = terrain.shape[2]

    if terrain_z < environment_z:
        z_pad = environment_z - terrain_z

        terrain = np.pad(terrain, pad_width=((0, 0), (0, 0), (0, z_pad)), mode='constant', constant_values=0)

    elif terrain_z > environment_z:
        terrain = terrain[:, :, :environment_z]

    return environment, terrain

