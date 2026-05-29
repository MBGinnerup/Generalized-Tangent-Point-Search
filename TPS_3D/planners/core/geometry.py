import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
import numpy as np
from scipy.spatial import ConvexHull
import numba as nb
from TPS_3D.planners.core import planners_utils as utils

@nb.njit(fastmath=True)
def dot(a, b):
    """Return Dot product of two 3D vectors."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@nb.njit
def point_in_hull(q, faces, face_normals, tol):
    """
    Check whether a point lies inside a convex polyhedral hull.

    The function evaluates each face using the half-space representation:
        (q - p0) · n <= tol

    where n is the outward-facing normal of the face.

    If this condition is violated (i.e. the point lies on or outside the
    face plane in the direction of the normal), the face is marked as
    violated.

    Parameters
    ----------
    q : np.ndarray
        Query point (3D).
    faces : list/array of arrays
        Each entry contains the vertices of a face.
    face_normals : np.ndarray
        Outward-facing normal vector for each face.
    tol : float
        Numerical tolerance for boundary inclusion.

    Returns
    -------
    inside : bool
        True if the point lies inside all half-spaces (convex hull).
    violated_faces : np.ndarray
        Indices of faces whose half-space constraint is violated.
    """
    n_faces = len(faces)
    tangent_faces = np.empty(n_faces, dtype=np.int32)
    inside = True
    count = 0 

    for f_idx in range(n_faces):
        verts = faces[f_idx]
        n = face_normals[f_idx]
        w = q - verts[0]
        d = dot(n, w)

        if d >= -tol:
            inside = False
            tangent_faces[count] = f_idx  
            count += 1

    return inside, tangent_faces[:count] 

@nb.njit
def add_tangent(found, verts, nx, ny, nz, added, count, tangents_array, labels):
    """
    Add valid tangent vertices to a list if they are inside bounds,
    free in the grid, and not already added.

    Parameters
    ----------
    found : bool
        Flag indicating whether at least one valid tangent has been found.
    verts : np.ndarray
        Vertex coordinates (N x 3).
    nx, ny, nz : int
        Grid dimensions.
    added : np.ndarray (bool/int)
        Boolean mask tracking already added tangent points.
    count : int
        Current number of stored tangent points.
    tangents_array : np.ndarray
        Preallocated array for storing tangent points.
    labels : np.ndarray
        3D occupancy grid.

    Returns
    -------
    found : bool
        Updated flag indicating if any valid tangent exists.
    count : int
        Updated number of stored tangent points.
    """
    for k in range(verts.shape[0]):

        v = verts[k]
        vr = int(v[0])
        vc = int(v[1])
        vz = int(v[2])

        if 0 <= vr < nx and 0 <= vc < ny and 0 <= vz < nz and labels[vr, vc, vz] == 0:
            found = True
            if added[vr, vc, vz] == 0:
                tangents_array[count,0] = vr
                tangents_array[count,1] = vc
                tangents_array[count,2] = vz
                added[vr, vc, vz] = 1
                count += 1

    return found, count

@nb.njit
def tangent_points(N, faces, face_normals, face_centers, q, labels, label_l, tol, max_divisions):
    """
    Compute valid tangent points from a query point to a convex obstacle hull.

    If the query point lies outside the hull, vertices from violated faces
    are returned directly as tangent candidates.

    If the query point lies inside the hull, the function progressively
    samples points from face centers toward face vertices and performs
    line-of-sight checks to identify reachable tangent vertices.

    Parameters
    ----------
    N : int
        Maximum number of tangent points.
    faces : list/array
        Hull faces defined by their vertices.
    face_normals : np.ndarray
        Outward-facing normal vector for each face.
    face_centers : np.ndarray
        Center point of each face.
    q : tuple
        Query point (x, y, z).
    labels : np.ndarray
        3D occupancy grid.
    label_l : int
        Target obstacle label used in collision checks.
    tol : float
        Tolerance used for hull membership testing.
    max_divisions : int
        Maximum subdivision level when sampling face points.

    Returns
    -------
    np.ndarray
        Valid tangent points.
    bool
        True if at least one tangent point was found.
    """

    nx, ny, nz = labels.shape
    
    tangents_array = np.empty((N, 3), dtype=np.int32)
    count = 0

    added = np.zeros_like(labels, dtype=np.uint8)

    q_x = int(q[0])
    q_y = int(q[1])
    q_z = int(q[2])

    added[q_x, q_y, q_z] = 1

    inside, tangent_faces = point_in_hull(q, faces, face_normals, tol=tol)

    found = False

    # CASE: q outside hull 
    if not inside:      
        for t_idx in range(tangent_faces.shape[0]):

            f_idx = tangent_faces[t_idx]
            verts = faces[f_idx]

            found, count = add_tangent(found, verts, nx, ny, nz, added, count, tangents_array, labels)

        if found:
            return tangents_array[:count]

    # CASE: q inside hull 
    pts_to_check = np.empty((N, 3), dtype=np.int32)

    for div in range(1, max_divisions + 1):
        div_inv = 1.0 / div

        for f_idx in range(len(faces)):

            verts = faces[f_idx]
            center = face_centers[f_idx]

            pt_count = 0

            if div == 1:
                cx = int(center[0])
                cy = int(center[1])
                cz = int(center[2])
                if 0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz:
                    pts_to_check[pt_count, 0] = cx
                    pts_to_check[pt_count, 1] = cy
                    pts_to_check[pt_count, 2] = cz
                    pt_count += 1
            elif div < max_divisions:
                for v in verts:
                    for t_i in range(1, div):
                        t = t_i * div_inv
                        px = int(center[0] + (v[0] - center[0]) * t)
                        py = int(center[1] + (v[1] - center[1]) * t)
                        pz = int(center[2] + (v[2] - center[2]) * t)
                        if 0 <= px < nx and 0 <= py < ny and 0 <= pz < nz:
                            pts_to_check[pt_count, 0] = px
                            pts_to_check[pt_count, 1] = py
                            pts_to_check[pt_count, 2] = pz
                            pt_count += 1
            else:
                for v in verts:
                    x = int(v[0])
                    y = int(v[1])
                    z = int(v[2])
                    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                        pts_to_check[pt_count, 0] = x
                        pts_to_check[pt_count, 1] = y
                        pts_to_check[pt_count, 2] = z
                        pt_count += 1

            # Check points
            for i in range(pt_count):

                px = pts_to_check[i, 0]
                py = pts_to_check[i, 1]
                pz = pts_to_check[i, 2]

                if px == q[0] and py == q[1] and pz == q[2]:
                    found, count = add_tangent(found, verts, nx, ny, nz, added, count, tangents_array, labels)
                    if found:
                        break
                    continue

                if utils.collision_label(labels, q, (px, py, pz), label_l) is None:
                    found, count = add_tangent(found, verts, nx, ny, nz, added, count, tangents_array, labels)
                    if found:
                        break

        if found:
            break

    return tangents_array[:count]

def tangents(hull, q, labels, label_l, tol=1e-9, max_divisions=3):
    """
    Return tangent points from a query point to a convex hull obstacle.

    Parameters
    ----------
    hull : tuple
        Hull representation containing vertices, faces, normals,
        and face centers.
    q : tuple
        Query point.
    labels : np.ndarray
        3D occupancy grid.
    label_l : int
        Target obstacle label used for collision checking.
    tol : float, optional
        Tolerance for hull membership testing.
    max_divisions : int, optional
        Maximum subdivision depth for interior tangent search.

    Returns
    -------
    list
        List of tangent points as (x, y, z) tuples.
    """
    N, faces, merged_normals, merged_centers = hull

    tangents_array = tangent_points(N, faces, merged_normals, merged_centers, np.array(q, dtype=np.int32), labels, label_l, tol=tol, max_divisions=max_divisions)

    tangents_tuples = [(x, y, z) for x, y, z in tangents_array]

    return tangents_tuples

@nb.njit
def dilate(points):
    """
    Perform one-step 3D morphological dilation on a set of voxel points.

    Each point expands to its 26 neighboring voxels and itself.
    Duplicate points are removed using a local occupancy grid.

    Parameters
    ----------
    points : np.ndarray
        Array of voxel coordinates with shape (N, 3).

    Returns
    -------
    np.ndarray
        Dilated voxel coordinates.
    """
    n = points.shape[0]

    # Worst-case output size: each point contributes up to 27 voxels
    out = np.empty((n * 27, 3), dtype=np.int32)
    k = 0

    # Bounding box
    xmin = points[0, 0]
    ymin = points[0, 1]
    zmin = points[0, 2]
    xmax = points[0, 0]
    ymax = points[0, 1]
    zmax = points[0, 2]

    for i in range(n):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]

        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
        if z < zmin: zmin = z
        if z > zmax: zmax = z

    # Add padding to avoid boundary checks during dilation
    H = xmax - xmin + 3
    W = ymax - ymin + 3
    D = zmax - zmin + 3

    seen = np.zeros((H, W, D), dtype=np.uint8)

    for i in range(n):

        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]

        xx = x - xmin + 1
        yy = y - ymin + 1
        zz = z - zmin + 1

        # center
        if seen[xx, yy, zz] == 0:
            seen[xx, yy, zz] = 1
            out[k, 0] = x
            out[k, 1] = y
            out[k, 2] = z
            k += 1

        # 26-neighbour dilation
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):

                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    nx = xx + dx
                    ny = yy + dy
                    nz = zz + dz

                    if seen[nx, ny, nz] == 0:
                        seen[nx, ny, nz] = 1
                        out[k, 0] = x + dx
                        out[k, 1] = y + dy
                        out[k, 2] = z + dz
                        k += 1

    return out[:k]

@nb.njit
def boundary_points(points):
    """
    Extract boundary points from a set of occupied 3D points.

    The function creates a local occupancy grid around the point set and
    classifies a point as a boundary point if at least one of its
    26 neighboring voxels is empty or outside the bounding box.

    Parameters
    ----------
    points : np.ndarray
        Array of occupied voxel coordinates with shape (N, 3).

    Returns
    -------
    np.ndarray
        Array containing only boundary voxel coordinates.
    """
    n = points.shape[0]

    xmin = points[0,0]
    xmax = points[0,0]
    ymin = points[0,1]
    ymax = points[0,1]
    zmin = points[0,2]
    zmax = points[0,2]

    # Bounding box
    for i in range(n):
        x = points[i,0]
        y = points[i,1]
        z = points[i,2]

        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
        if z < zmin: zmin = z
        if z > zmax: zmax = z

    H = xmax - xmin + 1
    W = ymax - ymin + 1
    D = zmax - zmin + 1

    # Grid
    grid = np.zeros((H, W, D), dtype=np.uint8)

    for i in range(n):
        grid[
            points[i,0] - xmin,
            points[i,1] - ymin,
            points[i,2] - zmin
        ] = 1

    # Boundary extraction
    out = np.empty((n, 3), dtype=np.int32)
    k = 0

    for i in range(n):

        x = points[i,0] - xmin
        y = points[i,1] - ymin
        z = points[i,2] - zmin

        boundary = False

        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):

                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    nx = x + dx
                    ny = y + dy
                    nz = z + dz

                    if nx < 0 or nx >= H or ny < 0 or ny >= W or nz < 0 or nz >= D:
                        boundary = True
                        break

                    if grid[nx, ny, nz] == 0:
                        boundary = True
                        break

                if boundary:
                    break
            if boundary:
                break

        if boundary:
            out[k,0] = points[i,0]
            out[k,1] = points[i,1]
            out[k,2] = points[i,2]
            k += 1

    return out[:k]

@nb.njit(fastmath=True)
def cross2D(o, a, b):
    """Return the signed 2D cross product used for orientation tests."""
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

@nb.njit
def monotone_chain_indices(points):
    """
    Compute the indices of points forming the 2D convex hull using the
    monotone chain algorithm.

    Parameters
    ----------
    points : np.ndarray
        Array of 2D points with shape (N, 2).

    Returns
    -------
    np.ndarray
        Indices of the input points that form the convex hull in CCW order.
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if n <= 2:
        return np.arange(n, dtype=np.int64)

    idx_sort_x = np.argsort(points[:,0])
    pts = points[idx_sort_x]
    idx_map = idx_sort_x.copy()

    # Build lower hull
    lower = nb.typed.List.empty_list(nb.int64)
    for i in range(n):
        while len(lower) >= 2 and cross2D(points[lower[-2]], points[lower[-1]], pts[i]) <= 0:
            lower.pop()
        lower.append(idx_map[i])

    # Build upper hull
    upper = nb.typed.List.empty_list(nb.int64)
    for i in range(n-1, -1, -1):
        while len(upper) >= 2 and cross2D(points[upper[-2]], points[upper[-1]], pts[i]) <= 0:
            upper.pop()
        upper.append(idx_map[i])

    # Merge hulls
    full_idx = nb.typed.List.empty_list(nb.int64)
    for i in range(len(lower)):
        full_idx.append(lower[i])
    for i in range(1, len(upper)-1):
        full_idx.append(upper[i])

    hull_array = np.empty(len(full_idx), dtype=np.int64)
    for i in range(len(full_idx)):
        hull_array[i] = full_idx[i]

    return hull_array

@nb.njit(fastmath=True)
def cross(a, b):
    """Return the 3D cross product a × b."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ], dtype=np.float32)

@nb.njit(fastmath=True)
def normalize(v):
    """
    Normalize a 3D vector.

    If the vector has near-zero magnitude, a zero vector is returned to
    avoid numerical instability.

    Parameters
    ----------
    v : array-like
        Input vector (x, y, z).

    Returns
    -------
    np.ndarray
        Unit vector in the direction of v, or (0, 0, 0) if v is near zero.
    """
    n = dot(v, v)
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    inv = 1.0 / np.sqrt(n)
    return np.array((v[0]*inv, v[1]*inv, v[2]*inv), dtype=np.float32)

@nb.njit(fastmath=True)
def face_normal(a, b, c):
    """
    Compute the normalized normal vector of a triangular face.

    The normal is computed using the cross product of two edge vectors:
        n = (b - a) × (c - a)

    Parameters
    ----------
    a, b, c : array-like
        3D vertices of the triangle face.

    Returns
    -------
    np.ndarray
        Unit normal vector of the face. Returns (0, 0, 0) if degenerate.
    """
    ab = b - a
    ac = c - a
    n = cross(ab, ac)
    return normalize(n)

@nb.njit
def unique_rows_numba_3d(arr, tol=1e-6):
    """
    Remove approximately duplicate 3D rows using a tolerance-based comparison.

    Two rows are considered identical if all coordinate differences are
    smaller than `tol`.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (N, 3).
    tol : float, optional
        Tolerance used to detect duplicate rows.

    Returns
    -------
    np.ndarray
        Array containing unique 3D rows.
    """
    n = arr.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    used = np.zeros(n, dtype=np.uint8)

    count = 0
    for i in range(n):
        if used[i] == 1:
            continue
        used[i] = 1
        out[count] = arr[i]
        count += 1
        for j in range(i+1, n):
            if used[j] == 1:
                continue
            if (abs(arr[i,0]-arr[j,0]) < tol and
                abs(arr[i,1]-arr[j,1]) < tol and
                abs(arr[i,2]-arr[j,2]) < tol):
                used[j] = 1

    return out[:count]

@nb.njit
def merge_faces_by_normal(faces_coords, hull_center, tol=1e-6):
    """
    Merge adjacent faces with approximately parallel normals into single faces.

    Faces are grouped if their normals are nearly parallel, after which their
    vertices are merged and re-triangulated into a consistent polygon using a
    2D projection + convex hull reconstruction.

    The final face orientation is enforced to be consistent with the hull center.

    Parameters
    ----------
    faces_coords : list of np.ndarray
        List of faces, each defined by 3D vertices.
    hull_center : np.ndarray
        Reference center used to ensure consistent face orientation.
    tol : float, optional
        Tolerance for detecting parallel normals.

    Returns
    -------
    merged_faces : list
        Merged and re-ordered face polygons.
    merged_normals : list
        Normal vector for each merged face.
    merged_centers : list
        Geometric center of each merged face.
    """
    n_faces = len(faces_coords)
    normals = nb.typed.List()
    used = nb.typed.List()

    for i in range(n_faces):
        tri = faces_coords[i]
        a = tri[0]
        b = tri[1]
        c = tri[2]
        n = face_normal(a, b, c)
        normals.append(n)
        used.append(False)

    merged_faces = nb.typed.List()
    merged_normals = nb.typed.List()
    merged_centers = nb.typed.List()

    # Merge parallelle faces 
    for i in range(n_faces):
        if used[i]:
            continue
        used[i] = True

        group = nb.typed.List()
        group.append(faces_coords[i])

        for j in range(i+1, n_faces):
            if used[j]:
                continue
            if dot(normals[i], normals[j]) > 1 - tol:
                used[j] = True
                group.append(faces_coords[j])

        total = 0
        for g in group:
            total += g.shape[0]

        all_pts = np.empty((total, 3), dtype=np.float32)
        idx = 0
        for g in group:
            for k in range(g.shape[0]):
                all_pts[idx] = g[k]
                idx += 1

        all_pts = unique_rows_numba_3d(all_pts) 


        if all_pts.shape[0] >= 3:
            a = all_pts[0].astype(np.float32)
            b = all_pts[1].astype(np.float32)
            c = all_pts[2].astype(np.float32)
            n_poly = face_normal(a, b, c)

            # Create local coordinate frame for 3D->2D projection
            if abs(n_poly[0]) < 0.9:
                base = np.array([1.0,0.0,0.0], dtype=np.float32)
            else:
                base = np.array([0.0,1.0,0.0], dtype=np.float32)

            v1 = normalize(cross(n_poly, base))
            v2 = normalize(cross(n_poly, v1))

            # 3D->2D projection
            proj = np.empty((all_pts.shape[0],2), dtype=np.float32)
            for k in range(all_pts.shape[0]):
                pt_f = all_pts[k].astype(np.float32)
                proj[k,0] = dot(pt_f, v1)
                proj[k,1] = dot(pt_f, v2)

            hull_idx = monotone_chain_indices(proj)

            ordered = np.empty((len(hull_idx),3), dtype=np.float32)
            for k in range(len(hull_idx)):
                ordered[k] = all_pts[hull_idx[k]].astype(np.float32)

            # Ensure consistent face orientation (towards hull center)
            center = np.zeros(3, dtype=np.float32)
            for k in range(ordered.shape[0]):
                center += ordered[k]
            center /= ordered.shape[0]

            to_out = center - hull_center
            n_chk = face_normal(ordered[0], ordered[1], ordered[2])

            if dot(n_chk, to_out) < 0:
                rev = np.empty_like(ordered)
                for k in range(ordered.shape[0]):
                    rev[k] = ordered[ordered.shape[0]-1-k]
                ordered = rev
                n_chk = -n_chk

            merged_faces.append(ordered)
            merged_normals.append(n_chk)
            merged_centers.append(center) 

        else:
            merged_faces.append(all_pts)
            merged_normals.append(np.zeros(3, dtype=np.float32))
            center = np.zeros(3, dtype=np.float32)
            for k in range(all_pts.shape[0]):
                center += all_pts[k]
            center /= all_pts.shape[0]
            merged_centers.append(center)

    return merged_faces, merged_normals, merged_centers

@nb.njit
def correct_face_orientation(tri, hull_center):
    """
    Ensure consistent face orientation such that the triangle normal
    points outward from the hull centroid.

    If the computed normal points toward the hull center, the vertex
    order of the triangle is flipped.

    Parameters
    ----------
    tri : np.ndarray
        Triangle vertices with shape (3, 3).
    hull_center : np.ndarray
        Centroid of the hull used as reference for outward orientation.

    Returns
    -------
    np.ndarray
        Triangle with corrected vertex ordering.
    """
    n = face_normal(tri[0], tri[1], tri[2])
    tri_center = (tri[0] + tri[1] + tri[2]) / 3.0

    to_outside = tri_center - hull_center

    # Flip if normal points toward hull center
    if dot(n, to_outside) < 0:
        temp = tri[1].copy()
        tri[1] = tri[2]
        tri[2] = temp

    return tri

@nb.njit
def build_faces_coords(simplices, points, hull_center):
    """
    Construct triangle face coordinates from simplices and ensure consistent orientation.

    Each simplex (triangle indices) is converted into 3D vertex coordinates,
    and face orientation is corrected so normals point outward relative to
    the hull centroid.

    Parameters
    ----------
    simplices : np.ndarray
        Array of triangle indices with shape (M, 3).
    points : np.ndarray
        Array of 3D points with shape (N, 3).
    hull_center : np.ndarray
        Centroid of the hull used for orientation correction.

    Returns
    -------
    list
        List of oriented triangle faces as (3, 3) arrays.
    """
    n = simplices.shape[0]
    faces_coords = nb.typed.List()
    for i in range(n):
        tri_idx = simplices[i]
        tri = np.empty((3,3), dtype=np.float32)
        for j in range(3):
            tri[j] = points[tri_idx[j]]
        tri = correct_face_orientation(tri, hull_center)
        faces_coords.append(tri)
    return faces_coords

def convexhull(obstacle):
    """
    Compute a 3D convex hull representation around a binary obstacle volume.

    The obstacle is first dilated to obtain an expanded boundary, ensuring that
    the resulting convex hull is constructed around the object rather than directly
    on its surface. Boundary points are then extracted from the dilated volume,
    and a convex hull is computed.

    Faces are subsequently built, oriented consistently, and merged based on
    similar normals to produce a compact surface representation.

    Parameters
    ----------
    obstacle : np.ndarray
        Point set representing all grid cells belonging to obstacle.

    Returns
    -------
    tuple
        (num_vertices, merged_faces, merged_normals, merged_centers)
        where:
        - num_vertices : int
            Number of hull vertices.
        - merged_faces : list
            Merged polygonal faces of the hull.
        - merged_normals : list
            Normal vectors for each merged face.
        - merged_centers : list
            Centroid of each merged face.
    """
    # Dilate obstacle to obtain a convex hull around the object
    dilated_obsatcle = dilate(obstacle)

    # Extract surface boundary points
    points = boundary_points(dilated_obsatcle)

    # Compute convex hull from extracted voxel boundary
    hull = ConvexHull(points)

    # Compute centroid of hull
    hull_center = points.mean(axis=0)

    # Build oriented triangle faces
    faces_coords = build_faces_coords(hull.simplices, points, hull_center)

    # Merge coplanar faces and remove duplicates
    merged_faces, merged_normals, merged_centers = merge_faces_by_normal(faces_coords, hull_center)

    return (len(hull.vertices), merged_faces, merged_normals, merged_centers)