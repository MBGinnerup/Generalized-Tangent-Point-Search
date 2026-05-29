import numba as nb
import numpy as np

@nb.njit(inline='always', fastmath=True)
def dist(p1, p2):
    """Return Euclidean distance between two 2D points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx*dx + dy*dy) ** 0.5

@nb.njit
def collision_obs(labels, node1, node2):
    """
    Check line-of-sight between two grid nodes and detect obstacles along the path.

    The function uses a discrete Bresenham-style ray traversal between two points
    in a labeled occupancy grid. It returns as soon as an obstacle is encountered,
    including handling of diagonal corner cases to avoid "corner-cutting".

    Parameters
    ----------
    labels : np.ndarray
        Labeled occupancy grid where 0 indicates free space and any non-zero value
        represents an obstacle ID.
    node1 : array-like
        Start node (x, y).
    node2 : array-like
        End node (x, y).

    Returns
    -------
    int or None
        - obstacle label (int): if a collision with an obstacle is detected
        - -1: if the ray exits the grid bounds
        - None: if the path is collision-free
    """
    x0, y0 = int(round(node1[0])), int(round(node1[1]))
    x1, y1 = int(round(node2[0])), int(round(node2[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy
    prev_x, prev_y = x0, y0

    x_max, y_max = labels.shape

    while True:

        if x0 < 0 or x0 >= x_max or y0 < 0 or y0 >= y_max:
            return -1  

        if labels[x0][y0] != 0:
            return labels[x0][y0]  

        dx_step = x0 - prev_x
        dy_step = y0 - prev_y

        # Diagonal corner-check to prevent corner-cutting
        if dx_step != 0 and dy_step != 0:

            l1 = labels[prev_x][y0]
            l2 = labels[x0][prev_y]

            # If moving diagonally between two obstacle cells of same label
            if l1 != 0 and l2 != 0 and l1 == l2:
                return l1

        if x0 == x1 and y0 == y1:
            break

        prev_x, prev_y = x0, y0

        # Bresenham step update
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return None

@nb.njit
def collision_label(labels, node1, node2, l):
    """
    Checks whether a line-of-sight between two nodes intersects a specific labeled obstacle.

    The function traces a Bresenham-like ray between node1 and node2 and returns
    the label 'l' if the path intersects any voxel belonging to that label.
    A diagonal corner check is included to prevent corner-cutting between adjacent
    cells of the same obstacle label.

    Parameters
    ----------
    labels : np.ndarray
        Labeled occupancy grid where 0 indicates free space and any non-zero value
        represents an obstacle ID.
    node1 : tuple
        Start position (x, y).
    node2 : tuple
        End position (x, y).
    l : int
        Obstacle label to test collision against.

    Returns
    -------
    int or None
        Returns 'l' if a collision with the specified obstacle is detected.
        Returns -1 if the ray exits the grid boundaries.
        Returns None if no collision occurs.
    """
    x0, y0 = int(round(node1[0])), int(round(node1[1]))
    x1, y1 = int(round(node2[0])), int(round(node2[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy
    prev_x, prev_y = x0, y0

    x_max, y_max = labels.shape

    while True:

        if x0 < 0 or x0 >= x_max or y0 < 0 or y0 >= y_max:
            return -1 

        if labels[x0, y0] == l:
            return l

        dx_step = x0 - prev_x
        dy_step = y0 - prev_y

        # Diagonal corner-check to prevent corner-cutting
        if dx_step != 0 and dy_step != 0:

            l1 = labels[prev_x, y0]
            l2 = labels[x0, prev_y]

            # If moving diagonally between two obstacle cells of same label
            if l1 == l and l2 == l:
                return l

        if x0 == x1 and y0 == y1:
            break

        prev_x, prev_y = x0, y0

        # Bresenham step update
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return None

@nb.njit
def argwhere(labels, target_label):
    """
    Return indices of all voxels matching a target label in a 2D grid.

    Parameters
    ----------
    labels : np.ndarray
        2D voxel grid containing integer labels.
    target_label : int
        Label value to search for.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing coordinates of matching voxels.
    """
    nx, ny = labels.shape

    # worst-case allocation
    out = np.empty((nx * ny, 2), dtype=np.int32)
    k = 0

    for x in range(nx):
        for y in range(ny):

            if labels[x, y] == target_label:
                out[k, 0] = x
                out[k, 1] = y
                k += 1

    return out[:k]

