import numpy as np
import numba as nb

@nb.njit(inline='always', fastmath=True)
def dist(p1, p2):
    """Return Euclidean distance between two 3D points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return (dx*dx + dy*dy + dz*dz) ** 0.5

@nb.njit
def collision_obs(labels, a, b):
    """
    Check line-of-sight collision between two voxels in a 3D occupancy grid.

    Uses the Amanatides & Woo (3D DDA) algorithm for efficient voxel traversal along a ray.

    Parameters
    ----------
    labels : np.ndarray
        3D voxel grid where 0 represents free space and non-zero values
        represent obstacles or labeled regions.
    a : tuple
        Start voxel index (x, y, z).
    b : tuple
        End voxel index (x, y, z).

    Returns
    -------
    int
        Obstacle label if a collision is detected.
    -1
        If the ray exits the grid boundaries.
    None
        If no collision occurs and the target voxel is reached.
    """
    nx, ny, nz = labels.shape

    # Start/end at voxel centers
    x0, y0, z0 = a[0] + 0.5, a[1] + 0.5, a[2] + 0.5
    ix, iy, iz = int(a[0]), int(a[1]), int(a[2])
    
    x1, y1, z1 = b[0] + 0.5, b[1] + 0.5, b[2] + 0.5
    ex, ey, ez = int(b[0]), int(b[1]), int(b[2])

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    inv_dx = 1.0 / dx if dx != 0 else 1e30
    inv_dy = 1.0 / dy if dy != 0 else 1e30
    inv_dz = 1.0 / dz if dz != 0 else 1e30

    tDeltaX = abs(inv_dx)
    tDeltaY = abs(inv_dy)
    tDeltaZ = abs(inv_dz)

    fx, fy, fz = ix, iy, iz

    # Initial ray-box intersection distances
    if dx > 0:
        tMaxX = (fx + 1 - x0) * inv_dx
    else:
        tMaxX = (x0 - fx) * -inv_dx if dx != 0 else 1e30

    if dy > 0:
        tMaxY = (fy + 1 - y0) * inv_dy
    else:
        tMaxY = (y0 - fy) * -inv_dy if dy != 0 else 1e30

    if dz > 0:
        tMaxZ = (fz + 1 - z0) * inv_dz
    else:
        tMaxZ = (z0 - fz) * -inv_dz if dz != 0 else 1e30

    # Amanatides & Woo (3D DDA) grid traversal:
    while True:
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            return -1

        lab = labels[ix, iy, iz]
        if lab != 0:
            return lab
        if ix == ex and iy == ey and iz == ez:
            return None

        # Step along axis with smallest t-value
        min_t = tMaxX
        axis = 0
        if tMaxY < min_t:
            min_t = tMaxY
            axis = 1
        if tMaxZ < min_t:
            axis = 2
            
        if axis == 0:
            ix += step_x
            tMaxX += tDeltaX
        elif axis == 1:
            iy += step_y
            tMaxY += tDeltaY
        else:
            iz += step_z
            tMaxZ += tDeltaZ

@nb.njit
def collision_label(labels, a, b, l):
    """
    Check line-of-sight collision between two voxels using voxel traversal.

    Uses the Amanatides & Woo (3D DDA) algorithm for efficient voxel traversal along a ray.

    Parameters
    ----------
    labels : np.ndarray
        3D voxel grid where 0 represents free space and non-zero values
        represent obstacles or labeled regions.
    a : tuple
        Start voxel index (x, y, z).
    b : tuple
        End voxel index (x, y, z).
    l : int
        Target label to detect during traversal.

    Returns
    -------
    int or None
        l:
            If a voxel with label `l` is encountered.
        None:
            If no matching label is found along the path and the target
            voxel is reached.
        -1:
            If the ray exits the grid boundaries.
    """
    nx, ny, nz = labels.shape

    # Start/end at voxel centers
    x0, y0, z0 = a[0] + 0.5, a[1] + 0.5, a[2] + 0.5
    ix, iy, iz = int(a[0]), int(a[1]), int(a[2])

    x1, y1, z1 = b[0] + 0.5, b[1] + 0.5, b[2] + 0.5
    ex, ey, ez = int(b[0]), int(b[1]), int(b[2])

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    inv_dx = 1.0 / dx if dx != 0 else 1e30
    inv_dy = 1.0 / dy if dy != 0 else 1e30
    inv_dz = 1.0 / dz if dz != 0 else 1e30

    tDeltaX = abs(inv_dx)
    tDeltaY = abs(inv_dy)
    tDeltaZ = abs(inv_dz)

    fx, fy, fz = ix, iy, iz

    # Initial ray-box intersection distances
    if dx > 0:
        tMaxX = (fx + 1 - x0) * inv_dx
    else:
        tMaxX = (x0 - fx) * -inv_dx if dx != 0 else 1e30

    if dy > 0:
        tMaxY = (fy + 1 - y0) * inv_dy
    else:
        tMaxY = (y0 - fy) * -inv_dy if dy != 0 else 1e30

    if dz > 0:
        tMaxZ = (fz + 1 - z0) * inv_dz
    else:
        tMaxZ = (z0 - fz) * -inv_dz if dz != 0 else 1e30

    # Amanatides & Woo (3D DDA) grid traversal:
    while True:

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
            return -1

        lab = labels[ix, iy, iz]
        if lab == l:
            return l

        if ix == ex and iy == ey and iz == ez:
            return None

        # Step along axis with smallest t-value
        min_t = tMaxX
        axis = 0
        if tMaxY < min_t:
            min_t = tMaxY
            axis = 1
        if tMaxZ < min_t:
            axis = 2

        if axis == 0:
            ix += step_x
            tMaxX += tDeltaX
        elif axis == 1:
            iy += step_y
            tMaxY += tDeltaY
        else:
            iz += step_z
            tMaxZ += tDeltaZ

@nb.njit
def argwhere(labels, target_label):
    """
    Return indices of all voxels matching a target label in a 3D grid.

    Parameters
    ----------
    labels : np.ndarray
        3D voxel grid containing integer labels.
    target_label : int
        Label value to search for.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing coordinates of matching voxels.
    """
    nx, ny, nz = labels.shape

    out = np.empty((nx * ny * nz, 3), dtype=np.int32)
    k = 0

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if labels[x, y, z] == target_label:
                    out[k, 0] = x
                    out[k, 1] = y
                    out[k, 2] = z
                    k += 1

    return out[:k]