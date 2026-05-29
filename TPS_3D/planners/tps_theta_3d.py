import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import numpy as np
import heapq
from TPS_3D.planners.core import fallback, geometry, search, pre_processing, planners_utils as utils

def TPS_Theta(grid, start, goal, max_search=None, pre=None, graph=None, fallback_method=fallback.Astar, hull_method=geometry.convexhull):
    """
    Theta*-based Tangent Point Search (Theta-TPS) using convex hull
    based obstacle navigation.

    Theta-TPS combines Tangent Point Search with Theta*-style any-angle
    path optimization. Instead of strictly connecting nodes through
    intermediate expansions, the algorithm performs line-of-sight based
    parent jumps, allowing nodes to connect directly to higher-quality
    ancestors when visibility permits.

    Convex hulls are precomputed and reused during execution, and the
    algorithm supports cached tangent reuse and optional fallback
    strategies for difficult or disconnected environments.

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
    max_search : int or None, optional
        Limits recursive tangent exploration depth. If None, unlimited.
    pre : tuple or None, optional
        Precomputed TPS preprocessing output:
            (hull_points, labels)
    graph : dict or None, optional
        Cached tangent connection graph used to reuse previously
        discovered obstacle-tangent relationships.
    fallback_method : callable or None, optional
        Global fallback planner used when TPS expansion fails to reach
        the goal efficiently (e.g. A*).
    hull_method : callable, optional
        Function used to compute obstacle convex hulls during preprocessing.
    Returns
    -------
    np.ndarray
        Ordered any-angle path from start to goal.

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
            depth_count = 1

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

                # Limit recursive tangent exploration.
                if max_search is not None:
                    if depth_count >= max_search:
                        continue
                    depth_count += 1
                
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

