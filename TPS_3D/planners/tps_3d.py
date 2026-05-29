import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import numpy as np
import heapq
from TPS_3D.planners.core import fallback, geometry, search, pre_processing, planners_utils as utils

def TPS(grid, start, goal, max_search=None, pre=None, graph=None, build_graph=False, fallback_method=fallback.Astar, hull_method=geometry.convexhull):
    """
    Tangent Point Search (TPS) using convex hull based obstacle navigation.

    TPS is a geometric path planning algorithm that avoids full grid
    exploration by expanding only obstacle-relevant tangent points
    derived from convex hull boundaries. The method incrementally
    constructs a path through visibility-constrained tangent expansions
    rather than cell-based search.

    Convex hulls of obstacles are precomputed (or generated during
    preprocessing) and used to efficiently compute tangent points between
    the current state and obstacle boundaries. The algorithm supports
    optional recursive tangent exploration, cached graph reuse, and
    fallback strategies when local geometric expansion is insufficient.

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
        Limits recursive tangent expansion depth. If None, no limit is applied.
    pre : tuple or None, optional
        Precomputed TPS preprocessing output:
            (hull_points, labels)
    graph : dict or None, optional
        Cached tangent connection graph used to reuse previously discovered
        obstacle-tangent relationships.
    build_graph : bool, optional
        If True, discovered tangent connections are stored in `graph`
        for reuse in future queries.
    fallback_method : callable or None, optional
        Global fallback planner used when tangent-based expansion fails
        to progress (e.g. A*).
    hull_method : callable, optional
        Function used to compute obstacle convex hulls during preprocessing.
    Returns
    -------
    np.ndarray
        Ordered path from start to goal.
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
            fallback.local_fallback(grid, t_goal_best, t_goals, t_goals_parent, closed_set, start, goal, parent, fallback_method, open_set, cost_so_far, graph, build_graph)

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
                t_goal_best, t_goal_distance = pre_processing.get_graph(labels, graph, current, obs_goal, goal, cost_so_far, parent, open_set, t_goals_parent, t_goals_cost, t_goals, t_goal_best, t_goal_distance, fallback_method)
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
                    distance = utils.dist(tangent, current)
                    g_new = cost_so_far[current] + distance
                    h_new = utils.dist(goal, tangent)
                    f_new = g_new + h_new
                    if tangent not in cost_so_far or g_new < cost_so_far[tangent]:
                        cost_so_far[tangent] = g_new
                        parent[tangent] = current
                        heapq.heappush(open_set, (f_new, tangent))
                    if build_graph and current != start:
                        graph.setdefault((current, obs_goal), set()).add((tangent, True))
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
                if build_graph and current != start:
                    graph.setdefault((current, obs_goal), set()).add((t_goal, False))

    path = search.reconstruct_path(parent, start, goal)

    return np.array(path)

