import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
import numpy as np
import heapq
from TPS_3D.planners.core import fallback, geometry, search, pre_processing, planners_utils as utils

def TPS_Bidirectional(grid, start, goal, max_search=None, pre=None, graph=None, build_graph=False, fallback_method=fallback.Astar, hull_method=geometry.convexhull):
    """
    Bidirectional Tangent Point Search (Bi-TPS) using convex hull based
    obstacle navigation.

    Bi-TPS extends Tangent Point Search by running two simultaneous
    geometric searches from both the start and goal positions. Each
    search expands only through visibility-constrained tangent points
    derived from obstacle convex hulls, reducing exploration compared
    to traditional grid-based bidirectional search.

    The two search fronts gradually propagate through the environment
    and terminate when they meet or when a direct line-of-sight
    connection is established. Convex hulls are precomputed and reused
    throughout the search to efficiently generate tangent candidates.

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
        Limits recursive tangent exploration depth. If None, no limit.
    pre : tuple or None, optional
        Precomputed TPS preprocessing output:
            (hull_points, labels)
    graph : dict or None, optional
        Cached tangent connection graph used to reuse previously
        discovered obstacle-tangent relationships.
    build_graph : bool, optional
        If True, discovered tangent connections are stored in `graph`
        for reuse in future planning queries.
    fallback_method : callable or None, optional
        Global fallback planner used when bidirectional tangent
        exploration cannot progress (e.g. A*).
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
    t_starts = []     
    t_starts_parent = {}
    t_start_best = None
    t_start_distance = np.inf
    t_goals_cost = {}
    t_starts_cost = {}

    open_set_fwd = [(utils.dist(start,goal), start)]
    open_set_bwd = [(utils.dist(goal,start), goal)]

    parent_fwd = {}
    cost_so_far_fwd = {start: 0}
    parent_bwd = {}
    cost_so_far_bwd = {goal: 0}

    closed_set_fwd = set()
    closed_set_bwd = set()

    meeting_node = None

    while True:

        # Local Fallback 
        if fallback_method and not open_set_fwd and not open_set_bwd:
            meeting_node = fallback.local_fallback_bidirectional(grid, t_start_best, t_goal_best, t_goals, t_goals_parent, closed_set_fwd, t_starts, t_starts_parent, closed_set_bwd, start, goal, parent_bwd, parent_fwd, fallback_method, open_set_fwd, cost_so_far_fwd, open_set_bwd, cost_so_far_bwd, graph, build_graph)

        if not open_set_fwd and not open_set_bwd:
            break

        # -- Forward search --
        if open_set_fwd:
            _, current_fwd = heapq.heappop(open_set_fwd)

            if current_fwd in closed_set_fwd:
                continue
            closed_set_fwd.add(current_fwd)

            if current_fwd in closed_set_bwd:
                meeting_node = current_fwd
                break

            # Check direct path to goal
            obs_goal = utils.collision_obs(labels, current_fwd, goal)
            if obs_goal is None:
                meeting_node = goal
                parent_fwd[goal] = current_fwd
                break

            # Reuse cached graph connections
            if graph is not None:
                if (current_fwd, obs_goal) in graph:
                    t_goal_best, t_goal_distance = pre_processing.get_graph(labels, graph, current_fwd, obs_goal, goal, cost_so_far_fwd, parent_fwd, open_set_fwd, t_goals_parent, t_goals_cost, t_goals, t_goal_best, t_goal_distance, fallback_method)
                    continue

            # Find convex hull of obstacle:
            hull_points_goal = hull_points[obs_goal]

            # Find tangents from current node:
            tangents_goal = geometry.tangents(hull_points_goal, current_fwd, labels, obs_goal)

            obs_search_fwd = []

            # Find path to tangent points
            for t_goal in tangents_goal:

                obs_list_fwd = [obs_goal]
                tangents_obs_fwd = [t_goal]
                depth_count_fwd = 1

                while tangents_obs_fwd:

                    tangent_fwd = tangents_obs_fwd.pop(0)

                    obs_new_fwd = utils.collision_obs(labels, current_fwd, tangent_fwd)
                    
                    # Direct path
                    if obs_new_fwd is None:
                        distance_fwd = utils.dist(tangent_fwd, current_fwd)
                        g_new_fwd = cost_so_far_fwd[current_fwd] + distance_fwd
                        h_new_fwd = utils.dist(goal, tangent_fwd)
                        f_new_fwd = g_new_fwd + h_new_fwd
                        if tangent_fwd not in cost_so_far_fwd or g_new_fwd < cost_so_far_fwd[tangent_fwd]:
                            cost_so_far_fwd[tangent_fwd] = g_new_fwd
                            parent_fwd[tangent_fwd] = current_fwd
                            heapq.heappush(open_set_fwd, (f_new_fwd, tangent_fwd))
                        if build_graph and current_fwd != start:
                            graph.setdefault((current_fwd, obs_goal), set()).add((tangent_fwd, True))
                        continue

                    # Limit recursive tangent exploration.
                    if max_search is not None:
                        if depth_count_fwd >= max_search:
                            continue
                        depth_count_fwd += 1

                    # Infinite loop protection
                    if obs_new_fwd in obs_list_fwd:
                        continue
                    obs_list_fwd.append(obs_new_fwd)

                    # Avoid revisiting discovered obstacles
                    if obs_new_fwd not in obs_search_fwd:

                        # Hull of new obstacle
                        hull_points_new_fwd = hull_points[obs_new_fwd]

                        tangents_new_fwd = geometry.tangents(hull_points_new_fwd, current_fwd, labels, obs_new_fwd)

                        tangents_obs_fwd.extend(tangents_new_fwd)

                        obs_search_fwd.append(obs_new_fwd)

                # Control fallback
                if fallback_method:
                    t_goal_best, t_goal_distance = fallback.control_fallback(t_goal, t_goals_parent, t_goals_cost, cost_so_far_fwd, t_goals, goal, current_fwd, t_goal_best, t_goal_distance)
                    if build_graph and current_fwd != start:
                        graph.setdefault((current_fwd, obs_goal), set()).add((t_goal, False))

        # --- Backward search --
        if open_set_bwd:

            _, current_bwd = heapq.heappop(open_set_bwd)

            if current_bwd in closed_set_bwd:
                continue
            closed_set_bwd.add(current_bwd)

            if current_bwd in closed_set_fwd:
                meeting_node = current_bwd
                break

            # Check direct path to goal
            obs_start = utils.collision_obs(labels, current_bwd, start)
            if obs_start is None:
                meeting_node = start
                parent_bwd[start] = current_bwd
                break
                
            # Reuse cached graph connections
            if graph is not None:
                if (current_bwd, obs_start) in graph:
                    t_start_best, t_start_distance = pre_processing.get_graph(labels, graph, current_bwd, obs_start, start, cost_so_far_bwd, parent_bwd, open_set_bwd, t_starts_parent, t_starts_cost, t_starts, t_start_best, t_start_distance, fallback_method)
                    continue

            # Find convex hull of obstacle:
            hull_points_start = hull_points[obs_start]

            # Find tangents from current node:
            tangents_start = geometry.tangents(hull_points_start, current_bwd, labels, obs_start)

            obs_search_bwd = []

            # Find path to tangent points
            for t_start in tangents_start:

                obs_list_bwd = [obs_start]
                tangents_obs_bwd = [t_start]
                depth_count_bwd = 1

                while tangents_obs_bwd:

                    tangent_bwd = tangents_obs_bwd.pop(0)

                    obs_new_bwd = utils.collision_obs(labels, current_bwd, tangent_bwd)
                    
                    # Direct path
                    if obs_new_bwd is None:
                        distance_bwd = utils.dist(tangent_bwd, current_bwd)
                        g_new_bwd = cost_so_far_bwd[current_bwd] + distance_bwd
                        h_new_bwd = utils.dist(start, tangent_bwd)
                        f_new_bwd = g_new_bwd + h_new_bwd
                        if tangent_bwd not in cost_so_far_bwd or g_new_bwd < cost_so_far_bwd[tangent_bwd]:
                            cost_so_far_bwd[tangent_bwd] = g_new_bwd
                            parent_bwd[tangent_bwd] = current_bwd
                            heapq.heappush(open_set_bwd, (f_new_bwd, tangent_bwd))
                        if build_graph and current_bwd != goal:
                            graph.setdefault((current_bwd, obs_start), set()).add((tangent_bwd, True))
                        continue

                    # Limit recursive tangent exploration.
                    if max_search is not None:
                        if depth_count_bwd >= max_search:
                            continue
                        depth_count_bwd += 1

                    # Infinite loop protection
                    if obs_new_bwd in obs_list_bwd:
                        continue
                    obs_list_bwd.append(obs_new_bwd)

                    # Avoid revisiting discovered obstacles
                    if obs_new_bwd not in obs_search_bwd:

                        # Hull of new obstacle
                        hull_points_new_bwd = hull_points[obs_new_bwd]

                        tangents_new_bwd = geometry.tangents(hull_points_new_bwd, current_bwd, labels, obs_new_bwd)

                        tangents_obs_bwd.extend(tangents_new_bwd)

                        obs_search_bwd.append(obs_new_bwd)
                    
                # Control fallback
                if fallback_method:
                    t_start_best, t_start_distance = fallback.control_fallback(t_start, t_starts_parent, t_starts_cost, cost_so_far_bwd, t_starts, start, current_bwd, t_start_best, t_start_distance)
                    if build_graph and current_bwd != goal:
                        graph.setdefault((current_bwd, obs_start), set()).add((t_start, False))
        
    path = search.reconstruct_path_bidirectional(parent_fwd, parent_bwd, start, goal, meeting_node)

    return np.array(path)