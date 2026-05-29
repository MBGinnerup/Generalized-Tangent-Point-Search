import matplotlib.pyplot as plt
import numpy as np

def plot_tube(ax, path, radius=0.25, resolution=12, color='steelblue', alpha=1.0):
    """
    Plot a 3D tube along a path.

    The function visualizes a discrete 3D path as a sequence of connected
    cylindrical segments. For each pair of consecutive points, a local
    orthogonal coordinate system is constructed and used to generate a
    cylindrical surface mesh around the path direction vector.

    Parameters:
        ax : matplotlib 3D axis
            Axis used for plotting.

        path : array-like of shape (N, 3)
            Sequence of 3D path coordinates.

        radius : float, optional
            Radius of the tube.

        resolution : int, optional
            Number of angular samples used for the cylinder mesh.

        color : str, optional
            Tube color.

        alpha : float, optional
            Tube transparency.
    """
    path = np.asarray(path, dtype=float) + 0.5

    for i in range(len(path) - 1):

        p0 = path[i]
        p1 = path[i + 1]

        v = p1 - p0
        L = np.linalg.norm(v)

        if L == 0:
            continue

        v = v / L

        # orthogonal basis
        not_v = np.array([1,0,0]) if abs(v[0]) < 0.9 else np.array([0,1,0])

        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)

        n2 = np.cross(v, n1)

        theta = np.linspace(0, 2*np.pi, resolution)
        t = np.linspace(0, L, 2)

        theta, t = np.meshgrid(theta, t)

        X = (p0[0] + v[0]*t + radius*np.cos(theta)*n1[0] + radius*np.sin(theta)*n2[0])

        Y = (p0[1] + v[1]*t + radius*np.cos(theta)*n1[1] + radius*np.sin(theta)*n2[1])

        Z = (p0[2] + v[2]*t + radius*np.cos(theta)*n1[2] + radius*np.sin(theta)*n2[2])

        ax.plot_surface(X, Y, Z, color=color, linewidth=0, antialiased=True, shade=True, alpha=alpha)


def visualize(voxel_map, path=None, start=None, goal=None, fig_size=6, alpha_fig=0.5, radius=0.25, s=40):
    """
    Visualize a 3D voxel environment and a path.

    The function plots occupied cells as semi-transparent voxels and optionally
    renders a 3D path as a smooth tube. Start and goal positions can also be
    displayed.

    Parameters:
        voxel_map : ndarray
            3D occupancy grid where occupied cells are nonzero.
        path : array-like, optional
            Sequence of 3D coordinates representing the path.
        start : tuple, optional
            Start coordinate.
        goal : tuple, optional
            Goal coordinate.
        fig_size : int, optional
            Figure size in inches.
        alpha_fig : float, optional
            Transparency of the voxel environment.
        radius : float, optional
            Radius of the rendered path tube.
        s : float, optional
            Marker size for start and goal points.
    """
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')

    filled = voxel_map > 0

    light_gray = (0.6, 0.6, 0.6)

    ax.voxels(filled, facecolors=light_gray, edgecolor=light_gray, linewidth=0.1, alpha=alpha_fig)

    if path is not None:

        path = np.asarray(path)

        if len(path) > 1:

            plot_tube(ax, path, radius=radius, color='steelblue')

    if start is not None:

        ax.scatter(start[0] + 0.5, start[1] + 0.5, start[2] + 0.5, color='red', s=s, depthshade=False)

    if goal is not None:

        ax.scatter(goal[0] + 0.5, goal[1] + 0.5, goal[2] + 0.5, color='green', s=s, depthshade=False)

    nx, ny, nz = voxel_map.shape

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)

    ax.set_box_aspect([nx, ny, nz])

    for x in range(nx + 1):
        ax.plot([x, x], [0, ny], [0, 0], color='gray', alpha=0.1)

        ax.plot([x, x], [0, ny], [nz, nz], color='gray', alpha=0.1)

    for y in range(ny + 1):
        ax.plot([0, nx], [y, y], [0, 0], color='gray', alpha=0.1)

        ax.plot([0, nx], [y, y], [nz, nz], color='gray', alpha=0.1)

    for x in range(nx + 1):
        ax.plot([x, x], [0, 0], [0, nz], color='gray', alpha=0.1)

        ax.plot([x, x], [ny, ny], [0, nz], color='gray', alpha=0.1)

    for z in range(nz + 1):
        ax.plot([0, nx], [0, 0], [z, z], color='gray', alpha=0.1)

        ax.plot([0, nx], [ny, ny], [z, z],color='gray', alpha=0.1)

    for y in range(ny + 1):
        ax.plot([0, 0], [y, y], [0, nz], color='gray', alpha=0.1)

        ax.plot([nx, nx], [y, y], [0, nz], color='gray', alpha=0.1)

    for z in range(nz + 1):
        ax.plot([0, 0], [0, ny], [z, z], color='gray', alpha=0.1)

        ax.plot([nx, nx], [0, ny], [z, z], color='gray', alpha=0.1)

    plt.show()
