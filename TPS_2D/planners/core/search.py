import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
from TPS_2D.planners.core import planners_utils as utils

def parent_jump(labels, current, tangent, parent, cost_so_far):
    """
    Compute Theta*-style jump cost and optimal parent selection for a tangent node.

    The function attempts to shortcut the path by checking line-of-sight between
    the parent of the current node and a tangent node. If a direct path is free
    of collision, the parent is rewired to enable path smoothing (Theta* behavior).
    Otherwise, the current node is used as the base.

    Parameters
    ----------
    labels : np.ndarray
        occupancy grid.
    current : tuple
        Current node in the search tree.
    tangent : tuple
        Target tangent node.
    parent : dict
        Parent mapping of nodes.
    cost_so_far : dict
        Cost-to-come values.

    Returns
    -------
    tuple
        (g_new, base_node)
        g_new : float
            Updated cost to reach tangent.
        base_node : tuple
            Selected parent node used for cost computation.
    """
    parent_node = parent.get(current)

    # Ensure parent node is a valid single node (not a list)
    if isinstance(parent_node, list):
        if len(parent_node) == 0 or parent_node[0] is None:
            parent_node = None
        else:
            parent_node = parent_node[0]

    # Use parent if direct path to tangent is collision-free
    if parent_node is not None and utils.collision_obs(labels, parent_node, tangent) is None:
        base_node = parent_node
    else:
        base_node = current

    g_new = cost_so_far[base_node] + utils.dist(base_node, tangent)
    return g_new, base_node

def reconstruct_path(parent, start, goal):
    """
    Reconstruct a path from goal to start.

    The function backtracks from the goal node to the start node using a
    parent dictionary. It supports both single-node parents and list-based
    segment parents (e.g., fallback.)

    If a valid path cannot be reconstructed, an empty list is returned.

    Parameters
    ----------
    parent : dict
        Mapping from node to its parent (or list of path segments).
    start : tuple
        Start node.
    goal : tuple
        Goal node.

    Returns
    -------
    list
        Reconstructed path from start to goal, or empty list if invalid.
    """
    path = [goal]
    node = goal
    while node != start:
        p = parent.get(node)
        if p is None:
            return []
        # Handle precomputed path segment (list-based parent)
        if isinstance(p, list):
            if len(p) == 0 or p[0] is None:
                return []
            path = p + path
            node = p[0]
        else:
            path.insert(0, p)
            node = p

    return path

def reconstruct_path_bidirectional(parent_fwd, parent_bwd, start, goal, meeting_node):
    """
    Reconstruct a full path in a bidirectional search by merging forward
    and backward parent chains at a meeting node.

    The function reconstructs:
    - Forward path: start → meeting_node using forward parent map
    - Backward path: meeting_node → goal using backward parent map

    Supports both single-node parents and list-based segment parents.

    Parameters
    ----------
    parent_fwd : dict
        Forward search parent map.
    parent_bwd : dict
        Backward search parent map.
    start : tuple
        Start node.
    goal : tuple
        Goal node.
    meeting_node : tuple
        Node where forward and backward searches connect.

    Returns
    -------
    list
        Full reconstructed path from start to goal, or empty list if invalid.
    """   
    # Direct reconstruction if meeting node equals goal
    if meeting_node == goal:
        return reconstruct_path(parent_fwd, start, goal)
    
    # Direct reconstruction if meeting node equals start
    if meeting_node == start:
        return reconstruct_path(parent_bwd, goal, start)

    # Reconstruct forward path: start → meeting_node
    path_fwd = [meeting_node]
    node = meeting_node

    while node != start:
        p = parent_fwd.get(node)
        if p is None:
            return []
        # Handle precomputed path segment (list-based parent)
        if isinstance(p, list):
            if len(p) == 0 or p[0] is None:
                return []
            path_fwd = p + path_fwd
            node = p[0]
        else:
            path_fwd.insert(0, p)
            node = p

    # Reconstruct backward path: meeting_node → goal
    path_bwd = []
    node = meeting_node

    while node != goal:
        p = parent_bwd.get(node)
        if p is None:
            return []  # ingen sti

        if isinstance(p, list):
            if len(p) == 0 or p[0] is None:
                return []
            path_bwd += p[::-1]
            node = p[0]
        else:
            path_bwd.append(p)
            node = p

    # Combine
    full_path = path_fwd + path_bwd

    return full_path