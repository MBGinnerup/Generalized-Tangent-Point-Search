import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import numpy as np
import heapq
from TPS_2D.planners.core import fallback, geometry, search, pre_processing, planners_utils as utils

def TPS_Theta(grid, start, goal, pre=None, graph=None, fallback_method=fallback.Astar, hull_method=geometry.monotonic_chain):
    """
    Theta*-enhanced Tangent Point Search (Theta-TPS) using convex hull
    based obstacle navigation.

    The algorithm combines Tangent Point Search (TPS) with Theta*-style
    line-of-sight parent updates to generate shorter and smoother any-angle
    paths. Instead of restricting path expansion to direct parent-child
    connections, Theta-TPS allows nodes to inherit higher-level visible
    ancestors through parent jumping, reducing unnecessary turning points.

    Similar to TPS, the search expands geometrically relevant tangent
    points around obstacle convex hulls rather than exhaustively exploring
    the occupancy grid. Convex hulls are either precomputed or generated
    during execution. The planner additionally supports cached tangent
    reuse and fallback strategies for difficult search regions.

    Parameters
    ----------
    grid : np.ndarray
        Binary occupancy grid where:
            0 = free space
            1 = obstacle
    start : tuple
        Start coordinate in grid space.
    goal : tuple
        Goal coordinate in grid space.
    pre : tuple, optional
        Precomputed preprocessing result:
            (hull_points, labels)
        Returned from:
            pre_processing.TPS_preprocess(...)
        If None, preprocessing is performed internally.
    graph : dict, optional
        Cached tangent connection graph used to reuse previously explored
        obstacle-tangent relationships between searches.
    fallback_method : callable, optional
        Path planning method used when tangent exploration cannot
        continue directly or when the open set becomes empty.
        Default is A*.
    hull_method : callable, optional
        Convex hull algorithm used during preprocessing.
        Default is the monotonic chain convex hull method.
    Returns
    -------
    np.ndarray
        Ordered any-angle path coordinates from start to goal.
    """
    if pre is None:
        hull_points, labels = pre_processing.TPS_preprocess(grid, hull_method)
    else:
        hull_points, labels = pre

    t_goals = []     
    t_goals_parent = {}
    t_goal_best = None
    t_goal_distance = np.inf
    t_goals_cost = {}

    open_set = [(utils.dist(start,goal), start)]

    parent = {}
    cost_so_far = {start: 0}

    closed_set = set()

    while True:

        # Local Fallback
        if fallback_method and not open_set:
            fallback.local_fallback(grid, t_goal_best, t_goals, t_goals_parent, closed_set, start, goal, parent, fallback_method, open_set, cost_so_far, graph, False)

        if not open_set:
            break

        _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)

        # Check direct path to goal
        obs_goal = utils.collision_obs(labels, current, goal)

        if obs_goal is None:
            parent[goal] = current
            break

        # Reuse cached graph connections
        if graph is not None:
            if (current, obs_goal) in graph:
                t_goal_best, t_goal_distance = pre_processing.get_graph(labels, graph, current, obs_goal, goal, cost_so_far, parent, open_set, t_goals_parent, t_goals_cost, t_goals, t_goal_best, t_goal_distance, fallback_method, jump=True)
                continue
        
        # Find convex hull of obstacle:
        hull_points_goal = hull_points[obs_goal]

        # Find tangents from current node:
        tangents_goal = geometry.tangents(hull_points_goal, current, labels, obs_goal)

        obs_search = []

        # Find path to tangent points
        for t_goal in tangents_goal:

            obs_list = [obs_goal]
            tangents_obs = [t_goal]

            while tangents_obs:

                tangent = tangents_obs.pop(0)

                obs_new = utils.collision_obs(labels, current, tangent)
                
                # Direct path
                if obs_new is None:
                    g_new, base_node = search.parent_jump(labels, current, tangent, parent, cost_so_far)
                    h_new = utils.dist(goal, tangent)
                    f_new = g_new + h_new
                    if tangent not in cost_so_far or g_new < cost_so_far[tangent]:
                        cost_so_far[tangent] = g_new
                        parent[tangent] = base_node
                        heapq.heappush(open_set, (f_new, tangent))
                    continue
                
                # Infinite loop protection
                if obs_new in obs_list:
                    continue
                obs_list.append(obs_new)

                # Avoid revisiting discovered obstacles
                if obs_new not in obs_search:

                    # Hull of new obstacle
                    hull_points_new = hull_points[obs_new]

                    tangents_new = geometry.tangents(hull_points_new, current, labels, obs_new)

                    tangents_obs.extend(tangents_new)

                    obs_search.append(obs_new)

            # Control fallback
            if fallback_method:
                t_goal_best, t_goal_distance = fallback.control_fallback(t_goal, t_goals_parent, t_goals_cost, cost_so_far, t_goals, goal, current, t_goal_best, t_goal_distance)

    path = search.reconstruct_path(parent, start, goal)

    return np.array(path)

