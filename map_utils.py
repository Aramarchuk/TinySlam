import numpy as np
import matplotlib.pyplot as plt

def create_real_map(N):
    """
    Creates the Ground Truth map (1.0 = free, 0.0 = wall).
    """
    real_map = np.ones((N, N), dtype=float)
    real_map[5:8, 5:10] = 0.0
    real_map[12:15, 8:11] = 0.0
    real_map[8:11, 20:23] = 0.0
    real_map[20:24, 12:16] = 0.0
    real_map[15:18, 30:34] = 0.0
    real_map[28:31, 25:29] = 0.0
    real_map[35:38, 15:19] = 0.0
    real_map[25:28, 38:42] = 0.0
    real_map[40:43, 30:35] = 0.0
    real_map[32:35, 5:9] = 0.0

    real_map[0, :] = 0.0
    real_map[:, 0] = 0.0
    real_map[N-1, :] = 0.0
    real_map[:, N-1] = 0.0

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

def bresenham(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    points = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points