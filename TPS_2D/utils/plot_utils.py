import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import networkx as nx

def visualize(occupancy_map, path=None, start=None, goal=None, fig_size=6, linewidth=1.5, markersize=3,s=40):
    """
    Visualize a 2D occupancy grid and a path.

    The function plots occupied cells as a binary map and optionally
    overlays a path together with start and goal positions. A light grid
    can also be displayed to improve spatial perception.

    Parameters:
        occupancy_map : ndarray
            2D occupancy grid where occupied cells are nonzero.
        path : array-like, optional
            Sequence of 2D coordinates representing the path.
        start : tuple, optional
            Start coordinate.
        goal : tuple, optional
            Goal coordinate.
        fig_size : int, optional
            Figure size in inches.
        grid : bool, optional
            If True, overlay grid lines.
        linewidth : float, optional
            Width of the plotted path.
        markersize : float, optional
            Marker size along the path.
        s : float, optional
            Marker size for start and goal points.
    """

    # -------------------------------------------------
    # Prepare occupancy map
    # -------------------------------------------------
    occupancy_map = occupancy_map.T
    occupancy_map = 1 - occupancy_map

    light_gray = (0.7, 0.7, 0.7)

    cmap = ListedColormap([light_gray,(1, 1, 1)])

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    ax.imshow(occupancy_map, cmap=cmap, origin='lower')

    nx, ny = occupancy_map.shape

    for x in range(ny + 1):

        ax.vlines(x - 0.5, -0.5, nx - 0.5, color='black', linewidth=0.5, alpha=0.05)
            
    for y in range(nx + 1):

        ax.hlines(y - 0.5, -0.5, ny - 0.5, color='black', linewidth=0.5, alpha=0.05)

    if path is not None:

        path = np.asarray(path)

        if len(path) > 1:

            ax.plot(path[:, 0], path[:, 1], color='steelblue', marker='o', markersize=markersize, linewidth=linewidth)

    if start is not None:

        ax.scatter(start[0], start[1], color='red', s=s, zorder=10)

    if goal is not None:

        ax.scatter(goal[0], goal[1], color='green', s=s, zorder=10)

    plt.show()

