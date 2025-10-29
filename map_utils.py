import numpy as np
import matplotlib.pyplot as plt

def create_real_map(N):
    """
    Creates the Ground Truth map (1.0 = free, 0.0 = wall).
    """
    real_map = np.ones((N, N), dtype=float)
    real_map[0, :] = 0.0
    real_map[-1, :] = 0.0
    real_map[:, 0] = 0.0
    real_map[:, -1] = 0.0
    real_map[2:4, 2:4] = 0.0
    real_map[7, 3] = 0.0
    real_map[5:8, 7] = 0.0
    return real_map

def create_robot_map(N):
    """
    Creates the robot's map (tinySLAM).
    0 = free, >0 = occupied.
    """
    # We could use float,
    # but int seems closer to the "counter" idea.
    return np.zeros((N, N), dtype=int)

def draw_bw(ax, arr, title, show_ticks=True):
    """
    Draws the map (0=black, 1=white, <0=gray).
    """
    img = np.where(arr < 0, 0.5, arr)  # unknown -> gray
    ax.imshow(
        img,
        vmin=0.0, vmax=1.0,
        cmap="gray",
        interpolation="nearest",
        origin="lower"
    )
    ax.set_title(title)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        N = arr.shape[0]
        ticks = np.arange(0, N, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax.grid(True, linestyle=':', alpha=0.6, color='grey')