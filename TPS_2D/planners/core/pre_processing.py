import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
import numpy as np
from tqdm.notebook import tqdm
import heapq
from scipy.ndimage import label, generate_binary_structure
import numba as nb
from TPS_2D.utils import path_queries as pq
from TPS_2D.planners.core import fallback, search, geometry, planners_utils as utils

@nb.njit(parallel=True)
def hulls_pre(labels, n, hull_points_array, max_pts, hull_method):
    """
    Compute convex hulls for all labeled obstacle components in parallel.

    For each obstacle label, extracts the corresponding occupied cells,
    computes its convex hull, and stores the result in a fixed-size array.

    Parameters
    ----------
    labels : np.ndarray
        Labeled grid of obstacle components.
    n : int
        Number of obstacle components.
    hull_points_array : np.ndarray
        Preallocated array for storing hull points.
    max_pts : int
        Maximum number of points per hull.
    hull_method : callable
        Function used to compute convex hull of a point set.

    Returns
    -------
    hull_points_array : np.ndarray
        Array containing convex hull points for all obstacles.
    """
    for l in nb.prange(n):
        obsatcle = utils.argwhere(labels, l+1)
        hull = hull_method(obsatcle) 
        for i in range(max_pts):
            if i < hull.shape[0]:
                hull_points_array[l, i, 0] = hull[i, 0]
                hull_points_array[l, i, 1] = hull[i, 1]
            else:
                hull_points_array[l, i, 0] = np.nan
                hull_points_array[l, i, 1] = np.nan
    return hull_points_array

def TPS_preprocess(grid, hull_method = geometry.monotonic_chain, max_pts=1000):
    """
    Preprocess a occupancy grid for Tangent Point Search (TPS).

    The function performs connected-component labeling on obstacles and
    computes convex hulls for each obstacle region.

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid.
    hull_method : callable, optional
        Method used to compute convex hulls.
    max_pts : int, optional
        Maximum number of points stored per hull.

    Returns
    -------
    hull_points : dict
        Convex hull points for each labeled obstacle.
    labels : np.ndarray
        Labeled obstacle grid.
    """
    # Label all connected obstacle components
    structure = generate_binary_structure(2, 2)
    labels, n = label(grid, structure=structure)

    hull_points_array = np.empty((n, max_pts, 2), dtype=np.float32)

    # Compute convex hull for each obstacle region
    hull_points_array = hulls_pre(labels, n, hull_points_array, max_pts, hull_method)
    hull_points = {}
    for i in range(n):
        mask = ~np.isnan(hull_points_array[i,:,0])
        hull_points[i+1] = hull_points_array[i, mask, :]

    return (hull_points, labels)

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
        fallback path planner.
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

def graph_builder(grid, pre, build_method, min_dist=100, iterations=1000, existing_graph=None, start_method=pq.random_start, goal_method=pq.random_goal, fallback_method=fallback.Astar):
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
    Returns
    -------
    dict
        A tangent connection graph where nodes represent sampled
        states and edges represent discovered geometric or fallback
        connections between them.
    """
    # Sample initial start state
    start = start_method(grid)

    # Initialize or reuse existing graph structure
    if existing_graph is None:
        graph = {}
    else:
        graph = existing_graph

    count = 0
    pbar = tqdm(total=iterations, desc="Simulations run")
    
    while count < iterations:

        goal = goal_method(grid, start, min_dist=min_dist)

        # Run selected planner and optionally store graph edges
        _ = build_method(grid, start, goal, pre=pre, graph=graph, build_graph=True, fallback_method=fallback_method)

        # Next start is previous goal
        start = goal
        count += 1
        pbar.update(1)

    return graph