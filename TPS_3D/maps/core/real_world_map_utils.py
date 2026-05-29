import numpy as np
import trimesh
from pathlib import Path

def crop_sides(grid):
    """
    Crop empty space around the x- and y-boundaries of a 3D grid.

    Finds the minimum and maximum occupied coordinates in the grid
    and removes empty regions along the x and y dimensions.

    Parameters:
        grid (ndarray): 3D occupancy grid.

    Returns:
        ndarray: Cropped 3D grid.
    """
    coords = np.argwhere(grid)

    x_min, y_min, _ = coords.min(axis=0)
    x_max, y_max, _ = coords.max(axis=0)

    return grid[x_min:x_max+1,y_min:y_max+1,:]

def crop_and_fill(volume, pad, fill):
    """
    Crop a 3D volume along the z-axis and optionally fill empty space below obstacles.

    The function removes empty layers below the first occupied voxel and keeps
    additional empty layers above the highest occupied voxel based on 'pad'.
    If needed, the volume is extended in the z-direction. Optionally fills all
    voxels below the first occupied cell in each (x, y) column.

    Parameters:
        volume (ndarray): 3D occupancy grid.
        pad (int): Number of empty layers kept above the highest occupied voxel.
        fill (bool): If True, fills empty space below occupied voxels.

    Returns:
        ndarray: Cropped (and optionally filled) 3D volume.
    """
    z_mask = volume.any(axis=(0,1))

    if not z_mask.any():
        return volume

    z_indices = np.where(z_mask)[0]
    min_z = z_indices[0]
    max_z = z_indices[-1]

    # Desired upper z-bound including extra air padding
    desired_max_z = max_z + pad

    # Extend volume if padding exceeds current bounds
    if desired_max_z >= volume.shape[2]:
        extra = desired_max_z - (volume.shape[2] - 1)

        volume = np.pad(volume, pad_width=((0, 0), (0, 0), (0, extra)), mode='constant', constant_values=False)

    # Crop volume to relevant z-range
    vol = volume[:, :, min_z:desired_max_z + 1].copy()

    # Fill everything below first occupied voxel
    if fill:
        first_true = np.argmax(vol, axis=2)

        has_true = vol.any(axis=2)

        z = np.arange(vol.shape[2])

        fill_mask = (z < first_true[:, :, None]) & has_true[:, :, None]

        vol[fill_mask] = True

    return vol

def get_array(voxel_size, map_name, file_name, pad, fill):
    """
    Load an OBJ environment file and convert it into a voxel grid.

    The function loads a mesh, voxelizes it using the specified voxel size,
    removes unnecessary boundaries, optionally fills empty space below
    obstacles, and ensures a ground layer exists.

    Parameters:
        voxel_size (float): Resolution used for voxelization.
        map_name (str): Name of the environment folder.
        file_name (str): Name of the OBJ file.
        pad (int): Number of empty layers kept above the environment.
        fill (bool): If True, fills empty space below occupied voxels.

    Returns:
        ndarray: Processed 3D occupancy grid.
    """
    file_path = (Path(__file__).resolve().parent.parent / "environments" / map_name / f"{file_name}.obj")

    loaded = trimesh.load(file_path)

    # Merge multiple meshes if OBJ contains a scene
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))

    # Use directly if OBJ already contains a single mesh
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    else:
        raise TypeError("Unknown mesh type from trimesh.load()")

    # Convert mesh to voxel representation
    voxelized = mesh.voxelized(pitch=voxel_size)
    vox_array = voxelized.matrix

    # Remove base plate 
    vox_array = vox_array[:,:,int(2*(1+voxel_size)):]

    # Remove empty side space
    vox_array = crop_sides(vox_array)

    # Crop z-direction and optionally fill below structures
    vox_array = crop_and_fill(vox_array, pad, fill)

    # Make sure there is a solid ground layer
    vox_array[:,:,0] = True

    return vox_array