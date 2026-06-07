import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
import numpy as np
from tqdm.notebook import tqdm
import heapq
from scipy.ndimage import label, generate_binary_structure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import numba as nb
from TPS_3D.utils import path_queries as pq
from TPS_3D.planners.core import fallback, search, geometry, planners_utils as utils

@nb.njit
def sub_terrains(terrain, sub_size):
    """
    Partition a 3D terrain occupancy grid into labeled sub-regions.

    The grid is divided into blocks of size `sub_size × sub_size` in the
    x–y plane. Each block containing at least one occupied voxel is assigned
    a unique negative label.

    Parameters
    ----------
    terrain : np.ndarray
        3D binary occupancy grid.
    sub_size : int
        Size of spatial subdivision in x and y directions.

    Returns
    -------
    labels : np.ndarray
        3D labeled terrain volume where each terrain region is assigned
        a unique negative label and non-terrain voxels are zero.
    label_count : int
        Number of labeled sub-regions.
    """
    x_len, y_len, z_len = terrain.shape
    labels = np.zeros((x_len, y_len, z_len), dtype=np.int32)

    current_label = -1
    label_count = 0

    for x in range(0, x_len, sub_size):
        for y in range(0, y_len, sub_size):

            x_end = min(x + sub_size, x_len)
            y_end = min(y + sub_size, y_len)

            used_label = False

            for i in range(x, x_end):
                for j in range(y, y_end):
                    for k in range(z_len):

                        if terrain[i, j, k]:
                            labels[i, j, k] = current_label
                            used_label = True

            if used_label:
                label_count += 1
                current_label -= 1

    return labels, label_count

def terrain_to_heightmap(terrain):
    """
    Convert a 3D terrain occupancy grid into a 2D height map.

    For each (x, y) position, the function finds the highest occupied
    voxel along the z-axis and stores its height in the output map.
    Positions without any occupied voxels are assigned a height of 0.

    Parameters
    ----------
    terrain : np.ndarray
        3D boolean occupancy grid with shape (x, y, z).

    Returns
    -------
    H : np.ndarray
        2D height map containing the highest occupied z-coordinate
        at each (x, y) position.
    """
    reverse = terrain[:, :, ::-1]

    first_hit = np.argmax(reverse, axis=2)
    has_terrain = reverse.any(axis=2)

    H = terrain.shape[2] - first_hit - 1
    H[~has_terrain] = 0

    return H.astype(np.float32)

def sub_terrains_watershed(terrain, min_dist=10):
    """
    Segment a 3D terrain into labeled sub-regions using watershed.

    The terrain is first converted into a 2D height map by extracting the
    highest occupied voxel at each (x, y) position. Local maxima in the
    height map are used as watershed markers, and watershed segmentation
    is performed in the x-y plane. The resulting 2D labels are then
    extruded along the z-axis and masked by the original terrain volume.

    Parameters
    ----------
    terrain : np.ndarray
        3D boolean occupancy grid representing the terrain.
    min_dist : int, optional
        Minimum distance between detected peaks used as watershed markers.

    Returns
    -------
    labels : np.ndarray
        3D labeled terrain volume where each terrain region is assigned
        a unique negative label and non-terrain voxels are zero.
    label_count : int
        Number of segmented terrain regions.
    """
    H = terrain_to_heightmap(terrain)

    coords = peak_local_max(H, min_distance=min_dist, exclude_border=False)
    markers = np.zeros_like(H, dtype=np.int32)
    for i, (x, y) in enumerate(coords):
        markers[x, y] = i + 1

    labels_2d = watershed(-H, markers)

    labels_3d = np.repeat(labels_2d[:, :, np.newaxis], terrain.shape[2], axis=2)

    labels_3d[~terrain] = 0

    label_count = labels_2d.max()

    return -labels_3d, label_count

def TPS_preprocess(grid, hull_method = geometry.convexhull, terrain = None, sub_terrain_method = sub_terrains, parameter = 20):
    """
    Preprocess a 3D occupancy grid for Tangent Point Search (TPS).

    The function performs connected-component labeling on obstacle
    regions, computes convex hulls for each obstacle, and optionally
    removes terrain before labeling and reintroduces it afterwards as
    separately labeled terrain regions.

    Terrain segmentation is performed using the supplied
    `sub_terrain_method`, which may be a fixed subdivision method
    or a terrain segmentation method such as watershed.

    Parameters
    ----------
    grid : np.ndarray
        3D binary occupancy grid.
    hull_method : callable, optional
        Function used to compute convex hulls from voxel sets.
    terrain : np.ndarray or None, optional
        Boolean terrain mask to remove and later reinsert.
    sub_terrain_method : callable, optional
        Function used to split the terrain into labeled sub-regions.
        Must return (labels, label_count).
    parameter : int or float, optional
        Parameter passed directly to `sub_terrain_method`.
        For example:
        - sub_size for block-based subdivision
        - min_dist for watershed segmentation

    Returns
    -------
    hulls : dict
        Mapping from label → convex hull representation.
    labels : np.ndarray
        Labeled 3D occupancy grid.
    """
    # Remove ground from occupancy grid before labeling
    if terrain is not None:
        grid = grid.copy()
        grid[terrain] = 0

    # Label all connected obstacle components
    structure = generate_binary_structure(3, 3)
    labels, n = label(grid, structure=structure)

    # Compute convex hull for each obstacle region
    hulls = {}
    for l in range(1, n + 1):
        obsatcle = utils.argwhere(labels, l)
        hull = hull_method(obsatcle)
        hulls[l] = hull

    # Reintroduce terrain regions as separate labeled obstacles
    if terrain is not None:

        # Split terrain into sub-regions for labeling
        labels_terrain, n_terrain = sub_terrain_method(terrain, parameter)
        labels[labels_terrain != 0] = labels_terrain[labels_terrain != 0]

        # Compute convex hull for terrain-derived obstacle regions
        for l_terrain in range(1, n_terrain + 1):
            obstacle_terrain = utils.argwhere(labels_terrain, -l_terrain)
            hull_terrain = hull_method(obstacle_terrain)
            hulls[-l_terrain] = hull_terrain

    return (hulls, labels)

def get_graph(labels, graph, current, obs_goal, goal, cost_so_far, parent, open_set, t_goals_parent, t_goals_cost, t_goals, t_goal_best, t_goal_distance, fallback_method, jump = False):
    """
    Expand precomputed graph edges or trigger fallback tangent handling.

    The function processes cached graph connections between the current
    node and obstacle-related goals. Depending on configuration, it either:
    - Performs Theta*-style rewiring (jump mode)
    - Performs standard A* expansion
    - Triggers fallback control for unresolved tangents

    Parameters
    ----------
    labels : np.ndarray
        Occupancy grid.
    graph : dict
        Cached local connection graph between nodes.
    current : tuple
        Current node.
    obs_goal : tuple
        Obstacle-associated goal key for graph lookup.
    goal : tuple
        Final goal node.
    cost_so_far : dict
        Cost-to-come map.
    parent : dict
        Parent pointer map for path reconstruction.
    open_set : list
        Priority queue for A* expansion.
    t_goals_parent : dict
        Parent map for tangent goals.
    t_goals_cost : dict
        Cost map for tangent goals.
    t_goals : list
        Tangent priority queue.
    t_goal_best : tuple
        Best tangent goal candidate.
    t_goal_distance : float
        Best heuristic distance found so far.
    fallback_method : callable
        Local path planner.
    jump : bool, optional
        If True, applies Theta*-style rewiring.

    Returns
    -------
    tuple
        (t_goal_best, t_goal_distance)
    """
    # Iterate over cached graph connections
    for tangent, path in graph[(current, obs_goal)]:
        # Only process valid precomputed paths
        if path:
            # Theta*-style shortcut with line-of-sight rewiring
            if jump:
                g_new, base_node = search.parent_jump(labels, current, tangent, parent, cost_so_far)
                h_new = utils.dist(goal, tangent)
                f_new = g_new + h_new
                if tangent not in cost_so_far or g_new < cost_so_far[tangent]:
                    cost_so_far[tangent] = g_new
                    parent[tangent] = base_node
                    heapq.heappush(open_set, (f_new, tangent))
            # Standard A* expansion without rewiring
            else:
                g_new = cost_so_far[current] + utils.dist(tangent, current)
                h_new = utils.dist(goal, tangent)
                f_new = g_new + h_new
                if tangent not in cost_so_far or g_new < cost_so_far[tangent]:
                    cost_so_far[tangent] = g_new
                    parent[tangent] = current
                    heapq.heappush(open_set, (f_new, tangent))
        # Trigger fallback for unresolved tangent connections
        elif fallback_method:
            t_goal_best, t_goal_distance = fallback.control_fallback(tangent, t_goals_parent, t_goals_cost, cost_so_far, t_goals, goal, current, t_goal_best, t_goal_distance)
    return t_goal_best, t_goal_distance

def graph_builder(grid, pre, build_method, min_dist=100, iterations=1000, existing_graph=None, start_method=pq.random_start_environment, goal_method=pq.random_goal_environment, fallback_method=fallback.Astar, terrain = True):
    """
    Construct a reusable tangent-connection graph by repeatedly solving
    sampled path planning problems in the environment.

    The function generates a connectivity graph by iteratively sampling
    start and goal states and executing a chosen planning method (e.g.
    TPS, Bidirectional TPS, or Theta-TPS). During each iteration, any
    discovered tangent connections or path edges are optionally stored
    in a shared graph structure, enabling reuse in future planning queries.

    Over many iterations, this builds a sparse but informative roadmap of
    geometrically relevant connections in the environment, focusing on
    obstacle boundaries and visibility structure rather than full grid
    exploration.

    Parameters
    ----------
    grid : np.ndarray
        Binary occupancy grid where:
            0 = free space
            1 = obstacle
    pre : tuple or None
        Precomputed convex hull data used by the planner.
    min_dist : float, optional
        Minimum Euclidean distance between sampled start and goal
        positions to ensure meaningful exploration. Default is 100.
    iterations : int, optional
        Number of sampling and planning iterations used to build
        the graph. Default is 1000.
    existing_graph : dict or None, optional
        Pre-existing graph structure to extend. If None, a new
        graph is created.
    start_method : callable, optional
        Function used to sample or generate start positions.
    goal_method : callable, optional
        Function used to sample or generate goal positions.
    build_method : callable, optional
        Path planning algorithm used to generate trajectories and
        extract tangent connections (e.g. TPS variants).
    fallback_method : callable, optional
        Global fallback planner used when local geometric search
        cannot find a valid continuation (e.g. A*).
    terrain : bool, optional
        If terrain map should be used for sampling in 3D environments. Default is True.
    Returns
    -------
    dict
        A tangent connection graph where nodes represent sampled
        states and edges represent discovered geometric or fallback
        connections between them.
    """
    # Sample initial start state (optionally conditioned on terrain map)
    if terrain:
        start = start_method(grid, terrain)
    else:
        start = start_method(grid)

    # Initialize or reuse existing graph structure
    if existing_graph is None:
        graph = {}
    else:
        graph = existing_graph

    count = 0
    pbar = tqdm(total=iterations, desc="Simulations run")
    
    while count < iterations:

        if terrain:
            goal = goal_method(grid, terrain, start, min_dist=min_dist)
        else:
            goal = goal_method(grid, start, min_dist=min_dist)

        # Run selected planner and optionally store graph edges
        _ = build_method(grid, start, goal, pre=pre, graph=graph, build_graph=True, fallback_method=fallback_method)

        # Next start is previous goal
        start = goal
        count += 1
        pbar.update(1)

    return graph
