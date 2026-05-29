import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
import numpy as np
import heapq
from TPS_2D.planners.core import planners_utils as utils

def control_fallback(t_goal, t_goals_parent, t_goals_cost, cost_so_far, t_goals, goal, current, t_goal_best, t_goal_distance):
    """
    Update and evaluate a tangent goal candidate within an A*-like search.

    The function computes the g-cost (path cost from start via current node),
    h-cost (heuristic to final goal), and f-cost (combined cost). It updates
    cost tracking structures and maintains a priority queue of tangent goals.
    The best tangent goal is selected based on heuristic distance to the goal.

    Parameters
    ----------
    t_goal : tuple
        Candidate tangent goal node.
    t_goals_parent : dict
        Parent mapping for tangent goals.
    t_goals_cost : dict
        Best known cost for each tangent goal.
    cost_so_far : dict
        Cost-to-come values for explored nodes.
    t_goals : list
        Priority queue (heap) of tangent goals.
    goal : tuple
        Final goal node.
    current : tuple
        Current search node.
    t_goal_best : tuple
        Currently best tangent goal (by heuristic).
    t_goal_distance : float
        Best heuristic distance found so far.

    Returns
    -------
    tuple
        (t_goal_best, t_goal_distance)
    """
    g_cost = cost_so_far[current] + utils.dist(current, t_goal)
    h_cost = utils.dist(t_goal, goal)
    f_cost = g_cost + h_cost

    # Update if this path is better or unseen
    if (t_goal not in t_goals_cost) or (g_cost < t_goals_cost[t_goal]):

        t_goals_cost[t_goal] = g_cost
        t_goals_parent[t_goal] = current

        # Push new entry
        heapq.heappush(t_goals, (f_cost, t_goal))

        # Track best heuristic candidate
        if h_cost < t_goal_distance:
            t_goal_best = t_goal
            t_goal_distance = h_cost
    
    return t_goal_best, t_goal_distance

def global_fallback(grid, start, goal, t_goal_best, t_goals_parent, closed_set, parent, fallback_method):
    """
    Execute a global fallback path planning step when local tangent-based
    planning is insufficient or fails.

    The function selects an appropriate fallback start point based on the
    current tangent goal state and computes a full global path using a
    provided fallback planner. The resulting path is stored in the parent
    structure for reconstruction.

    Priority of fallback start:
    1. Best tangent goal if it has been expanded (in closed set)
    2. Parent of best tangent goal if available
    3. Start node as last resort

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid used for planning.
    start : tuple
        Start coordinate.
    goal : tuple
        Goal coordinate.
    t_goal_best : tuple or None
        Best tangent goal candidate.
    t_goals_parent : dict
        Parent mapping for tangent goals.
    closed_set : set
        Set of already expanded nodes.
    parent : dict
        Path reconstruction structure.
    fallback_method : callable
        Function to compute fallback paths.

    Returns
    -------
    None
    """    
    # If best tangent goal was already expanded, use it directly
    if t_goal_best in closed_set:
        global_fallback_path = fallback_method(grid, t_goal_best, goal)

        if global_fallback_path is not None: 
            parent[goal] = [tuple(p) for p in global_fallback_path[:-1]]
        
        return
    
    # Otherwise fallback to parent of best tangent goal
    elif t_goal_best is not None:
        t_goal_best_current = t_goals_parent[t_goal_best]
        global_fallback_path = fallback_method(grid, t_goal_best_current, goal)

        if global_fallback_path is not None: 
            parent[goal] = [tuple(p) for p in global_fallback_path[:-1]]

        return 
    
    # If no valid tangent goal exists, fallback to start
    else:
        global_fallback_path = fallback_method(grid, start, goal)

        if global_fallback_path is not None: 
            parent[goal] = [tuple(p) for p in global_fallback_path[:-1]]

        return

def global_fallback_bidirectional(grid, start, goal, t_goal_best, t_goals_parent, closed_set_fwd, parent_fwd, t_start_best, t_starts_parent, closed_set_bwd, parent_bwd, fallback_method):
    """
    Execute a global fallback path planning step in a bidirectional search.

    The function selects fallback connection points from both the forward
    and backward search trees and computes a full global path between them
    using a provided fallback planner.

    Priority for forward/backward selection:
    1. Use tangent goal/start if already expanded
    2. Otherwise use its parent in the search tree
    3. If unavailable, fallback point is None

    If both forward and backward fallback points exist, a global path is
    computed between them and stored for path reconstruction.

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid used for planning.
    start : tuple
        Start node.
    goal : tuple
        Goal node.
    t_goal_best : tuple or None
        Best tangent goal in forward search.
    t_goals_parent : dict
        Parent mapping for forward tangents.
    closed_set_fwd : set
        Expanded nodes in forward search.
    parent_fwd : dict
        Forward path reconstruction map.
    t_start_best : tuple or None
        Best tangent start in backward search.
    t_starts_parent : dict
        Parent mapping for backward tangents.
    closed_set_bwd : set
        Expanded nodes in backward search.
    parent_bwd : dict
        Backward path reconstruction map.
    fallback_method : callable
        Function to compute fallback paths.

    Returns
    -------
    tuple or None
        The selected backward fallback node if a valid fallback path
        is found, otherwise None.
    """
    # Select forward fallback point based on search state
    if t_goal_best in closed_set_fwd:
        t_best_fwd = t_goal_best
    elif t_goal_best is not None:
        t_best_fwd = t_goals_parent[t_goal_best]
    else:
        t_best_fwd = None

    # Select backward fallback point based on search state
    if t_start_best in closed_set_bwd:
        t_best_bwd = t_start_best
    elif t_start_best is not None:
        t_best_bwd = t_starts_parent[t_start_best]
    else:
        t_best_bwd = None
    
    # Proceed if both forward and backward fallback points exist
    if t_best_fwd and t_best_bwd:
        global_fallback_path = fallback_method(grid, t_best_fwd, t_best_bwd)

        if global_fallback_path is not None: 
            parent_fwd[t_best_bwd] = [tuple(p) for p in global_fallback_path[:-1]]

        return t_best_bwd
    
    # Proceed if just forward fallback point exist
    elif t_best_fwd:
        global_fallback_path = fallback_method(grid, t_best_fwd, goal)

        if global_fallback_path is not None: 
            parent_fwd[goal] = [tuple(p) for p in global_fallback_path[:-1]]

        return goal
    
    # Proceed if just backward fallback point exist
    elif t_best_bwd:
        global_fallback_path = fallback_method(grid, t_best_bwd, start)

        if global_fallback_path is not None: 
            parent_bwd[start] = [tuple(p) for p in global_fallback_path[:-1]]

        return start
    
    # If no valid fallback points exists, fallback to start
    else:
        global_fallback_path = fallback_method(grid, start, goal)

        if global_fallback_path is not None: 
            parent_fwd[goal] = [tuple(p) for p in global_fallback_path[:-1]]

        return goal

def get_best(heap, closed_set):
    """
    Retrieve the best valid (non-closed) element from a priority queue.

    Parameters
    ----------
    heap : list
        Priority queue implemented as a heap (heapq format).
    closed_set : set
        Set of nodes that have already been processed/expanded.

    Returns
    -------
    object or None
        The first valid node not in `closed_set`, or None if no valid
        entries remain in the heap.
    """
    while heap:
        _, t = heapq.heappop(heap)
        if t not in closed_set:
            return t
    return None

def create_fallback_path(grid, t_best, t_goals_parent, fallback_method, parent, cost_so_far, open_set, goal, start, graph, build_graph):
    """
    Create or retrieve a fallback path between tangent nodes in a search graph.

    This function either reuses a cached local fallback path or computes a new
    one between a parent tangent node and the selected best tangent node.
    The resulting path is stored in the parent structure and optionally cached
    in a graph for reuse.

    If the path reaches the goal, the function signals early termination.

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid used for local path planning.
    t_best : tuple
        Current best tangent node.
    t_goals_parent : dict
        Parent mapping for tangent goal nodes.
    fallback_method : callable
        Function used to compute fallback paths.
    parent : dict
        Path reconstruction structure.
    cost_so_far : dict
        Cost-to-come values for nodes.
    open_set : list
        Priority queue for A* expansion.
    goal : tuple
        Goal node.
    start : tuple
        Start node.
    graph : dict or None
        Cache of previously computed local fallback paths.
    build_graph : bool
        Whether to store newly computed paths in cache.

    Returns
    -------
    None

    """   
    # Get parent node of best tangent
    t_best_current = t_goals_parent[t_best]
    
    # Use cached local path if available
    if graph is not None and (t_best_current, t_best) in graph:
        local_path = graph[(t_best_current, t_best)]
    else:
        local_path = fallback_method(grid, t_best_current, t_best)
        if build_graph and t_best_current != start:
            graph[(t_best_current, t_best)] = local_path

    if local_path is None:
        return

    # Link local path to parent structure (exclude last node)
    local_path = [tuple(p) for p in local_path]

    # Link hele stien til parent (uden goal)
    parent[t_best] = local_path[:-1]

    # Stop early if goal is reached
    if t_best == goal:
        return

    path_length = sum(
        utils.dist(local_path[i+1], local_path[i])
        for i in range(len(local_path)-1)
    )

    # Update cost-to-come with local path length
    cost_so_far[t_best] = cost_so_far[t_best_current] + path_length

    # Insert node into open set for further expansion
    heapq.heappush(open_set, (0, t_best))

    return

def local_fallback(grid, t_goal_best, t_goals, t_goals_parent, closed_set, start, goal, parent, fallback_method, open_set, cost_so_far, graph, build_graph):
    """
    Attempt to construct a local fallback connection.

    The function first tries to extract a valid tangent goal from the open
    set that has not been expanded yet. If such a node exists, a local
    fallback path is constructed. Otherwise, the algorithm falls back to
    a global planner.

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid used for planning.
    t_goal_best : tuple or None
        Best tangent goal candidate.
    t_goals : list
        Priority queue of tangent goals.
    t_goals_parent : dict
        Parent mapping for tangent goals.
    closed_set : set
        Expanded nodes.
    start : tuple
        Start node.
    goal : tuple
        Goal node.
    parent : dict
        Path reconstruction structure.
    fallback_method : callable
        Function to compute fallback paths.
    open_set : list
        Priority queue for main A* search.
    cost_so_far : dict
        Cost-to-come values.
    graph : dict or None
        Cached local fallback paths.
    build_graph : bool
        Whether to store computed paths.

    Returns
    -------
    None
    """
    if t_goals:

        # Try to get a valid tangent node from t_goals:
        t_best = get_best(t_goals, closed_set)
        
        # Build local fallback path from selected tangent
        if t_best is not None:
            return create_fallback_path(grid, t_best, t_goals_parent, fallback_method, parent, cost_so_far, open_set, goal, start, graph, build_graph)
    
    # No valid local option → fallback to global planner
    return global_fallback(grid, start, goal, t_goal_best, t_goals_parent, closed_set, parent, fallback_method)    

def local_fallback_bidirectional(grid, t_start_best, t_goal_best, t_goals, t_goals_parent, closed_set_fwd, t_starts, t_starts_parent, closed_set_bwd, start, goal, parent_bwd, parent_fwd, fallback_method, open_set_fwd, cost_so_far_fwd, open_set_bwd, cost_so_far_bwd, graph, build_graph):
    """
    Perform bidirectional local fallback connection.

    The function attempts to connect forward and backward search trees using
    local fallback paths from both tangent goal and tangent start candidates.

    Priority:
    1. Forward local fallback (tangent goals)
    2. Backward local fallback (tangent starts)
    3. Global bidirectional fallback if local connection fails

    If any local fallback succeeds, the resulting path is integrated into
    the respective search trees.

    Parameters
    ----------
    grid : np.ndarray
        Occupancy grid.
    t_start_best : tuple or None
        Best tangent start candidate.
    t_goal_best : tuple or None
        Best tangent goal candidate.
    t_goals : list
        Forward tangent priority queue.
    t_goals_parent : dict
        Forward tangent parent map.
    closed_set_fwd : set
        Forward closed set.
    t_starts : list
        Backward tangent priority queue.
    t_starts_parent : dict
        Backward tangent parent map.
    closed_set_bwd : set
        Backward closed set.
    start : tuple
        Start node.
    goal : tuple
        Goal node.
    parent_bwd : dict
        Backward path reconstruction map.
    parent_fwd : dict
        Forward path reconstruction map.
    fallback_method : callable
        Function to compute fallback paths.
    open_set_fwd : list
        Forward A* open set.
    cost_so_far_fwd : dict
        Forward cost map.
    open_set_bwd : list
        Backward A* open set.
    cost_so_far_bwd : dict
        Backward cost map.
    graph : dict or None
        Cached fallback paths.
    build_graph : bool
        Whether to cache computed paths.

    Returns
    -------
    int or tuple
        None or result from global fallback.
    """
    # Flag
    changed = False

    # Try local connection from forward search (tangent goals)
    if t_goals:

        t_best_fwd = get_best(t_goals, closed_set_fwd)

        if t_best_fwd is not None:
            create_fallback_path(grid, t_best_fwd, t_goals_parent, fallback_method, parent_fwd, cost_so_far_fwd, open_set_fwd, goal, start, graph, build_graph)
            changed = True
    
    # Try local connection from backward search (tangent starts)
    if t_starts:

        t_best_bwd = get_best(t_starts, closed_set_bwd)
            
        if t_best_bwd is not None:
            create_fallback_path(grid, t_best_bwd, t_starts_parent, fallback_method, parent_bwd, cost_so_far_bwd, open_set_bwd, start, goal, graph, build_graph)
            changed = True
    
    if changed:
        return

    # No local connection possible → fallback to global bidirectional planner
    return global_fallback_bidirectional(grid, start, goal, t_goal_best, t_goals_parent, closed_set_fwd, parent_fwd, t_start_best, t_starts_parent, closed_set_bwd, parent_bwd, fallback_method)

def Astar(grid, start, goal):
    """
    A* path planning on a 2D occupancy grid with 8-connectivity.

    Parameters
    ----------
    grid : ndarray
        Occupancy grid (0 = free, non-zero = obstacle).
    start : tuple
        Start cell (x, y).
    goal : tuple
        Goal cell (x, y).

    Returns
    -------
    path : ndarray
        Planned path from start to goal.
    """
    x_width, y_width = grid.shape

    # Motions for 8-connectivity 
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, np.sqrt(2)],
              [-1, 1, np.sqrt(2)],
              [1, -1, np.sqrt(2)],
              [1, 1, np.sqrt(2)]]

    open_set = [(utils.dist(start,goal),start)]

    parent = {}

    cost_so_far = {start: 0} 

    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            break

        cx, cy = current

        for dx, dy, move_cost in motion:

            # Neighbor:
            nx, ny = cx + dx, cy + dy

            if 0 <= nx < x_width and 0 <= ny < y_width:
                # Check diogonal corners:
                if dx != 0 and dy != 0:
                    # Make sure you cannot pass trough two diagonal obsatcles:
                    if grid[cx, ny] != 0 and grid[nx, cy] != 0:
                        continue
                if grid[nx, ny] != 0:
                    continue
                g = cost_so_far[current] + move_cost
                if (nx, ny) not in cost_so_far or g < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = g
                    f = g + utils.dist((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    parent[(nx, ny)] = current

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent.get(node) 
        if node is None:
            return []
    path.append(start)
    path = np.array(path[::-1])

    return path