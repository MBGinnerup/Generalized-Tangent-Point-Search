import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
import numpy as np
import numba as nb
from TPS_2D.planners.core import planners_utils as utils

@nb.njit
def point_in_hull(q, hull_pts, tol):
    """
    Check whether a point lies inside a convex hull.

    The function performs an orientation test against all directed
    edges of the convex hull. A point is considered inside only if it
    lies on the same (positive) side of every edge within a numerical
    tolerance.

    Parameters
    ----------
    q : array-like
        Query point (x, y).
    hull_pts : np.ndarray
        Ordered convex hull vertices of shape (N, 2).
    tol : float
        Numerical tolerance for cross-product sign tests.

    Returns
    -------
    bool
        True if the point lies inside the convex hull, otherwise False.
    """
    n = hull_pts.shape[0]

    # Check point against each directed hull edge
    for i in range(n):
        p0 = hull_pts[i]
        p1 = hull_pts[(i+1) % n]

        # Edge vector
        edge_x = p1[0] - p0[0]
        edge_y = p1[1] - p0[1]

        # Vector from edge start to query point
        to_point_x = q[0] - p0[0]
        to_point_y = q[1] - p0[1]

        # Signed 2D cross product (orientation test)
        cross = edge_x*to_point_y - edge_y*to_point_x

        # If point lies on wrong side of any edge → outside hull
        if cross <= tol:
            return False
    return True

@nb.njit
def add_tangent(found, p1, p2, rows, cols, added, count, tangents_array, labels):
    """
    Add valid tangent candidates.

    The function checks the endpoints of a segment (p1, p2) and adds them
    as tangent candidates if they lie within bounds, belong to a valid
    obstacle region, and have not already been added.

    Parameters
    ----------
    found : bool
        Indicates whether at least one valid tangent has been found.
    p1, p2 : array-like
        Endpoints of a hull edge segment.
    rows, cols : int
        Grid dimensions for boundary checking.
    added : np.ndarray
        Boolean grid marking already added tangent points.
    count : int
        Current number of stored tangent points.
    tangents_array : np.ndarray
        Output array storing tangent points.
    labels : np.ndarray
        Labeled grid used to validate obstacle membership.

    Returns
    -------
    found : bool
        Updated flag indicating if any valid tangent was added.
    count : int
        Updated number of stored tangent points.
    """
    r1 = int(p1[0])
    c1 = int(p1[1])
    r2 = int(p2[0])
    c2 = int(p2[1])

    # p1
    if 0 <= r1 < rows and 0 <= c1 < cols and labels[r1, c1] == 0:
        found = True
        if added[r1,c1] == 0:
            tangents_array[count,0] = r1
            tangents_array[count,1] = c1
            added[r1,c1] = 1
            count += 1

    # p2
    if 0 <= r2 < rows and 0 <= c2 < cols and labels[r2, c2] == 0:
        found = True
        if added[r2,c2] == 0:
            tangents_array[count,0] = r2
            tangents_array[count,1] = c2
            added[r2,c2] = 1
            count += 1
    
    return found, count

@nb.njit
def tangent_points(hull_pts, q, labels, label_l, tol, max_divisions):
    """
    Compute tangent points from a query point to a convex hull.

    The function finds candidate tangent points between a query point 'q'
    and a convex hull defined by 'hull_pts'. Two cases are handled:

    - Case 1: q is outside the hull → true tangents are identified by
      checking sign consistency of cross products.
    - Case 2: q is inside the hull → tangents are approximated using
      adaptive edge subdivision until valid visibility (line-of-sight)
      conditions are satisfied.

    Parameters
    ----------
    hull_pts : np.ndarray
        Ordered convex hull vertices of shape (N, 2).
    q : np.ndarray
        Query point (x, y).
    labels : np.ndarray
        Labeled grid used for obstacle identification and collision checks.
    label_l : int
        Label of the current obstacle.
    tol : float
        Numerical tolerance for geometric tests (e.g. cross product sign).
    max_divisions : int
        Maximum subdivision depth for edge sampling in the inside-hull case.

    Returns
    -------
    tangents : np.ndarray
        Array of detected tangent points.
    found : bool
        True if at least one valid tangent was found.
    """
    rows, cols = labels.shape
    N = hull_pts.shape[0]

    tangents_array = np.empty((N*2, 2), dtype=np.int32)
    count = 0

    added = np.zeros((rows, cols), dtype=np.uint8)

    q_r = int(q[0])
    q_c = int(q[1])

    added[q_r, q_c] = 1

    # CASE: q outside hull 
    if not point_in_hull(q, hull_pts, tol):
        for i in range(N):
            p = hull_pts[i]

            r = int(p[0])
            c = int(p[1])

            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue

            if labels[r, c] != 0:
                continue

            if added[r, c] == 1:
                continue

            v_x = p[0] - q[0]
            v_y = p[1] - q[1]

            all_pos = True
            all_neg = True

            # Check orientation of all other hull vertices
            for j in range(N):
                if j == i:
                    continue

                w_x = hull_pts[j,0] - q[0]
                w_y = hull_pts[j,1] - q[1]

                # Cross product determines relative orientation
                cross = v_x * w_y - v_y * w_x

                # If points lie on different sides, it's not a tangent
                if cross < -tol:
                    all_pos = False
                if cross > tol:
                    all_neg = False

            # Valid tangent if all points lie on one side of line q -> p
            if all_pos or all_neg:
                tangents_array[count,0] = r
                tangents_array[count,1] = c
                count += 1

        return tangents_array[:count]
    
    # CASE: q inside hull 
    for divisions in range(2, max_divisions + 2):
        found = False

        for i in range(N):
            p1 = hull_pts[i]
            p2 = hull_pts[(i+1) % N]

            # max divisions
            if divisions == (max_divisions + 1):
                
                # Only endpoints
                for p in (p1, p2):

                    px = int(p[0])
                    py = int(p[1])

                    if px < 0 or px >= rows or py < 0 or py >= cols:

                        if px == q_r and py == q_c:
                            found, count = add_tangent(found, p1, p2, rows, cols, added, count, tangents_array, labels)
                            if found:
                                break
                            continue

                        if utils.collision_label(labels, q, (px, py), label_l) is None:
                            found, count = add_tangent(found, p1, p2, rows, cols, added, count, tangents_array, labels)
                            if found:
                                break

                continue 

            # Subdivision
            for t_i in range(1, divisions):
                t = t_i / divisions

                px = int(p1[0] * (1 - t) + p2[0] * t)
                py = int(p1[1] * (1 - t) + p2[1] * t)

                if px < 0 or px >= rows or py < 0 or py >= cols:
                    continue

                if px == q_r and py == q_c:
                    found, count = add_tangent(found, p1, p2, rows, cols, added, count, tangents_array, labels)
                    if found:
                        break
                    continue

                if utils.collision_label(labels, q, (px, py), label_l) is None:
                    found, count = add_tangent(found, p1, p2, rows, cols, added, count, tangents_array, labels)
                    if found:
                        break

        if found:
            break

    return tangents_array[:count]

def tangents(hull_pts, q, labels, label_l, tol=1e-9, max_divisions=5):
    """
    Return tangent points from a query point to a convex hull obstacle.

    Parameters
    ----------
    hull_pts : tuple
        Convex hull points.
    q : tuple
        Query point.
    labels : np.ndarray
        2D occupancy grid.
    label_l : int
        Target obstacle label used for collision checking.
    tol : float, optional
        Tolerance for hull membership testing.
    max_divisions : int, optional
        Maximum subdivision depth for interior tangent search.

    Returns
    -------
    list
        List of tangent points as (x, y) tuples.
    """
    tangents_array = tangent_points(hull_pts, np.array(q), labels, label_l, tol, max_divisions)

    tangents_tuples = [(r, c) for r, c in tangents_array]

    return tangents_tuples

@nb.njit
def dilate(points):
    """
    Perform one-step 2D morphological dilation on a set of points.

    Each point expands to its 8 neighboring voxels and itself.
    Duplicate points are removed using a local occupancy grid.

    Parameters
    ----------
    points : np.ndarray
        Array of coordinates with shape (N, 2).

    Returns
    -------
    np.ndarray
        Dilated coordinates.
    """
    n = points.shape[0]

    # Worst-case output size: each point contributes up to 9 voxels
    out = np.empty((n * 9, 2), dtype=np.int32)
    k = 0

    # Bounding box
    rmin = points[0, 0]
    cmin = points[0, 1]
    rmax = points[0, 0]
    cmax = points[0, 1]

    for i in range(n):
        r = points[i, 0]
        c = points[i, 1]
        if r < rmin: rmin = r
        if r > rmax: rmax = r
        if c < cmin: cmin = c
        if c > cmax: cmax = c

    # Add padding to avoid boundary checks during dilation
    H = rmax - rmin + 3
    W = cmax - cmin + 3

    seen = np.zeros((H, W), dtype=np.uint8)

    for i in range(n):

        r = points[i, 0]
        c = points[i, 1]

        rr = r - rmin + 1
        cc = c - cmin + 1

        # center
        if seen[rr, cc] == 0:
            seen[rr, cc] = 1
            out[k, 0] = r
            out[k, 1] = c
            k += 1

        # 8-neighbour dilation
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):

                if dr == 0 and dc == 0:
                    continue

                nr = rr + dr
                nc = cc + dc

                if seen[nr, nc] == 0:
                    seen[nr, nc] = 1
                    out[k, 0] = r + dr
                    out[k, 1] = c + dc
                    k += 1

    return out[:k]

@nb.njit
def boundary_points(points):
    """
    Extract boundary points from a set of occupied 2D points.

    The function creates a local occupancy grid around the point set and
    classifies a point as a boundary point if at least one of its
    8 neighboring points is empty or outside the bounding box.

    Parameters
    ----------
    points : np.ndarray
        Array of occupied voxel coordinates with shape (N, 2).

    Returns
    -------
    np.ndarray
        Array containing only boundary coordinates.
    """
    n = points.shape[0]

    rmin = points[0,0]
    rmax = points[0,0]
    cmin = points[0,1]
    cmax = points[0,1]

    # Bounding box
    for i in range(n):
        r = points[i,0]
        c = points[i,1]
        if r < rmin: rmin = r
        if r > rmax: rmax = r
        if c < cmin: cmin = c
        if c > cmax: cmax = c

    H = rmax - rmin + 1
    W = cmax - cmin + 1

    grid = np.zeros((H, W), dtype=np.uint8)

    for i in range(n):
        grid[points[i,0] - rmin, points[i,1] - cmin] = 1

    # Boundary extraction
    out = np.empty((n, 2), dtype=np.int32)
    k = 0

    for i in range(n):

        r = points[i,0] - rmin
        c = points[i,1] - cmin

        boundary = False

        for dr in (-1,0,1):
            for dc in (-1,0,1):

                if dr == 0 and dc == 0:
                    continue

                nr = r + dr
                nc = c + dc

                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    boundary = True
                    break

                if grid[nr, nc] == 0:
                    boundary = True
                    break

            if boundary:
                break

        if boundary:
            out[k,0] = points[i,0]
            out[k,1] = points[i,1]
            k += 1

    return out[:k]

@nb.njit
def collinear_points(points):
    """
    Find the two most distant points in a set.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points.

    Returns
    -------
    np.ndarray
        Two points (shape (2, 2)) that are farthest apart.
    """
    N = points.shape[0]
    max_dist2 = -1.0
    i_max, j_max = 0, 1

    for i in range(N):
        for j in range(i+1, N):
            dx = points[i,0] - points[j,0]
            dy = points[i,1] - points[j,1]
            dist2 = dx*dx + dy*dy  
            if dist2 > max_dist2:
                max_dist2 = dist2
                i_max = i
                j_max = j

    return points[[i_max, j_max]]

@nb.njit
def unique_rows_numba(points):
    """
    Remove duplicate 2D points.

    The function sorts points x, then y and marks
    duplicates in-place using a sentinel value. Only the first occurrence
    of each unique point is kept in the final output array.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing 2D points.

    Returns
    -------
    result : np.ndarray
        Array containing only unique 2D points.
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0,2), dtype=points.dtype)
    
    # First x, then y
    idx = np.argsort(points[:,0])
    pts = points[idx].copy()
    
    # Sentinel value used to mark duplicates
    MARK = -999999
    for i in range(1, n):
        for j in range(i-1, -1, -1):
            if pts[i,0] == pts[j,0] and pts[i,1] == pts[j,1]:
                pts[i,0] = MARK
                break
    
     # Count non-marked (unique) points
    count = 0
    for i in range(n):
        if pts[i,0] != MARK:
            count += 1

    result = np.empty((count,2), dtype=points.dtype)

    # Fill result with unique points
    idx2 = 0
    for i in range(n):
        if pts[i,0] != MARK:
            result[idx2,0] = pts[i,0]
            result[idx2,1] = pts[i,1]
            idx2 += 1
    return result

@nb.njit(fastmath=True)
def cross(a,b,c):
    """Return the signed 2D cross product used for orientation tests."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

@nb.njit
def remove_duplicates_set(lst):
    """
    Removes duplicate 2D points while preserving order.

    The function iterates through a list of points and keeps only the
    first occurrence of each unique (x, y) coordinate pair. A set is
    used to track previously seen points for efficient duplicate detection.

    Parameters
    ----------
    lst : list of array-like
        List of 2D points, where each element is [x, y].

    Returns
    -------
    res : numba.typed.List
        List of unique 2D points in the same order as first appearance.
    """
    seen = set()          
    res = nb.typed.List()      
    for p in lst:
        t = (p[0], p[1]) 
        if t not in seen:
            seen.add(t)
            res.append(p)
    return res

@nb.njit
def monotonic_chain_algorithm(points):
    """
    Computes the convex hull of a set of 2D points using the monotonic chain algorithm.

    The function constructs the convex hull from a set of (x, y) coordinates by first
    removing duplicates, sorting the points by x-coordinate and then y-coordinate,
    and then building the lower and upper hulls. Collinear points are filtered to
    produce a minimal boundary representation.

    Parameters
    ----------
    points : np.ndarray
        Array of (x, y) coordinates representing obstacle cells or boundary points.

    Returns
    -------
    hull_array : np.ndarray
        Ordered convex hull points as an (N, 2) integer array.
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0,2), dtype=np.int32)
    if n <= 2:
        return points

    # Remove duplicate points
    points = unique_rows_numba(points)

    # sort x, then y
    idx = np.argsort(points[:,0])
    pts = points[idx]

    start = 0
    for i in range(1, pts.shape[0]+1):
        if i == pts.shape[0] or pts[i,0] != pts[start,0]:
            sub_idx = np.argsort(pts[start:i,1])
            pts[start:i] = pts[start:i][sub_idx]
            start = i

    lower = nb.typed.List()
    for i in range(pts.shape[0]):
        p = pts[i]
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
            lower.pop()
        lower.append(p)

    upper = nb.typed.List()
    for i in range(pts.shape[0]-1, -1, -1):
        p = pts[i]
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
            upper.pop()
        upper.append(p)

    full = nb.typed.List()
    for i in range(len(lower)):
        full.append(lower[i])
    for i in range(1, len(upper)): 
        full.append(upper[i])

    # Remove collinear redundancy in hull
    hull = nb.typed.List()
    hull.append(full[0])
    last = full[0]
    i = 1
    while i < len(full)-1:
        j = i
        while j < len(full)-1 and cross(last, full[j], full[j+1]) == 0:
            j += 1
        hull.append(full[j])
        last = full[j]
        i = j + 1
    hull.append(full[-1])

    hull = remove_duplicates_set(hull)

    hull_array = np.empty((len(hull),2), dtype=np.int32)
    for k in range(len(hull)):
        hull_array[k,0] = hull[k][0]
        hull_array[k,1] = hull[k][1]

    return hull_array

@nb.njit
def monotonic_chain(obstacle):
    """
    Computes the convex hull of an obstacle using the monotonic chain algorithm.

    The obstacle is first dilated to obtain an expanded boundary, ensuring that
    the resulting convex hull is constructed around the object rather than directly
    on its surface. Boundary points are then extracted from the dilated volume,
    and a convex hull is computed.

    Parameters
    ----------
    obstacle : np.ndarray
        Point set representing all grid cells belonging to obstacle.

    Returns
    -------
    hull_pts : np.ndarray
        Ordered convex hull points.
    """
    # Dilate obstacle to obtain a convex hull around the object
    dialated_obsatcle = dilate(obstacle)

    # Extract boundary points from obstacle
    points = boundary_points(dialated_obsatcle)

    # Compute convex hull from extracted boundary using monotonic chain algorithm
    hull_pts = monotonic_chain_algorithm(points)

    return hull_pts
