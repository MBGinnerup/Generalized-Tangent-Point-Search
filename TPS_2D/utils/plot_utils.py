import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import networkx as nx

def visualize(occupancy_map, path=None, start=None, goal=None, graph=None, fig_size=6, linewidth=1.5, markersize=3, s=20):
    """
    Visualize a 2D occupancy grid, path, and graph structure.

    The function plots occupied cells as a binary map and optionally
    overlays a path together with start and goal positions. Addiotnally a graph can be plotted.

    Parameters:
        occupancy_map : ndarray
            2D occupancy grid where occupied cells are nonzero.
        path : array-like, optional
            Sequence of 2D coordinates representing the path.
        start : tuple, optional
            Start coordinate.
        goal : tuple, optional
            Goal coordinate.
        graph : dict, optional
            Graph structure to visualize.
        fig_size : int, optional
            Figure size in inches.
        linewidth : float, optional
            Width of the plotted path.
        markersize : float, optional
            Marker size along the path.
        s : float, optional
            Marker size for start and goal points.
    """

    occupancy_map = occupancy_map.T
    occupancy_map = 1 - occupancy_map

    light_gray = (0.7, 0.7, 0.7)

    cmap = ListedColormap([light_gray, (1, 1, 1)])

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    ax.imshow(occupancy_map, cmap=cmap, origin='lower')

    nx_size, ny_size = occupancy_map.shape

    for x in range(ny_size + 1):

        ax.vlines(x - 0.5, -0.5, nx_size - 0.5, color='black', linewidth=0.5, alpha=0.05)

    for y in range(nx_size + 1):

        ax.hlines(y - 0.5, -0.5, ny_size - 0.5, color='black', linewidth=0.5,alpha=0.05)

    if path is not None:

        path = np.asarray(path)

        if len(path) > 1:

            ax.plot(path[:, 0], path[:, 1], marker='o', markersize=markersize, linewidth=linewidth)

    if start is not None:

        ax.scatter(start[0], start[1], color='red', s=s, zorder=10)

    if goal is not None:

        ax.scatter(goal[0], goal[1], color='green', s=s, zorder=10)

    if graph is not None:

        fallback_paths = []

        G = nx.Graph()

        for (current, obs_goal), neighbors in graph.items():

            if  isinstance(neighbors, set):

                for tangent, distance in neighbors:

                    if distance:
                        G.add_edge(current, tangent)
                        
            else:
                pth = neighbors
                fallback_paths.append(pth)

        pos = {}

        for node in G.nodes:

            if len(node) == 3:

                pos[node] = (node[0], node[1])

            else:

                pos[node] = node

        for u, v in G.edges:

            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]

            ax.plot(x, y, color='slateblue', linewidth=0.7, alpha=0.6)

        xs = [pos[n][0] for n in G.nodes]
        ys = [pos[n][1] for n in G.nodes]

        ax.scatter(xs, ys, color='slateblue', s=5, alpha=0.7)

        for pth in fallback_paths:

            pth = np.asarray(pth)

            if len(pth) > 1:

                ax.plot(pth[:, 0], pth[:, 1], color='slateblue', linewidth=0.7, alpha=0.6)

    plt.show()
