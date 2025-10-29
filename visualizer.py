import matplotlib.pyplot as plt
import os
from map_utils import draw_bw

def save_grid_based_snapshot(t, output_dir, cfg, gt, robot_map, paths, pose_gt, pose_est, hit_points):
    """
    Draws and saves one frame (snapshot) of the current simulation state.

    :param t: Current step (int)
    :param output_dir: Folder to save to (str)
    :param cfg: Config module
    :param gt: Ground Truth map (np.array)
    :param robot_map: Robot's map (np.array)
    :param paths: Dict with 'gt_x', 'gt_y', 'est_x', 'est_y' (dict)
    :param pose_gt: (x, y) - GT pose
    :param pose_est: (x_est, y_est) - Estimated pose
    :param hit_points: List of [(x, y), ...] lidar points
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    x, y = pose_gt
    x_est, y_est = pose_est

    # --- Left: GT map + GT path + GT pose + LIDAR ---
    draw_bw(ax1, gt, f"GT map (step {t})")
    ax1.plot(paths['gt_x'], paths['gt_y'], linestyle="--", linewidth=1,
             color=cfg.GT_COLOR)
    ax1.plot(x, y, marker="o", markersize=8, linestyle="None",
             color=cfg.GT_COLOR)

    # Drawing lidar rays
    for (hit_x, hit_y) in hit_points:
        ax1.plot([x, hit_x], [y, hit_y],
                 color="red", linestyle="-", linewidth=0.7, alpha=0.8)
        ax1.plot(hit_x, hit_y,
                 marker="x", color="red", markersize=4)

    # --- Right: Robot map + Estimated path + Estimated pose ---
    draw_bw(ax2, robot_map, "Robot map + estimated pose")
    ax2.plot(paths['est_x'], paths['est_y'], linestyle="--", linewidth=1,
             color=cfg.EST_COLOR)
    ax2.plot(x_est, y_est, marker="x", markersize=10, linestyle="None",
             color=cfg.EST_COLOR)

    plt.tight_layout()

    filename = os.path.join(output_dir, f"step_{t:03d}.png")
    plt.savefig(filename)
    plt.close(fig)


def save_ekf_snapshot(t, output_dir, cfg, gt_grid_map, gt_landmark_map, robot_map, paths, pose_gt, pose_est):
    """
    Draws and saves a frame for EKF (landmark-based) simulation.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    x, y = pose_gt
    x_est, y_est = pose_est
    draw_bw(ax1, gt_grid_map, f"GT Map + Landmarks (step {t})")
    # --- Left: GT map (Landmarks) + GT path + GT pose ---
    ax1.set_title(f"GT Landmark Map (step {t})")

    # Draw GT landmarks (e.g., red stars)
    if gt_landmark_map:
        lm_x, lm_y = zip(*gt_landmark_map)
        ax1.plot(lm_x, lm_y, 'r*', markersize=12, label="GT Landmarks",
                 color='darkred')

    # Draw GT path and pose
    ax1.plot(paths['gt_x'], paths['gt_y'], linestyle="--", linewidth=1,
             color=cfg.GT_COLOR)
    ax1.plot(x, y, marker="o", markersize=8, linestyle="None",
             color=cfg.GT_COLOR)

    ax1.plot(paths['gt_x'], paths['gt_y'], linestyle="--", linewidth=1,
             color=cfg.GT_COLOR)
    ax1.plot(x, y, marker="o", markersize=8, linestyle="None",
             color=cfg.GT_COLOR)
    # --- Right: Robot map (empty) + Estimated path + Estimated pose ---

    draw_bw(ax2, robot_map, "Robot map (empty) + estimated pose")
    ax2.plot(paths['est_x'], paths['est_y'], linestyle="--", linewidth=1,
             color=cfg.EST_COLOR)
    ax2.plot(x_est, y_est, marker="x", markersize=10, linestyle="None",
             color=cfg.EST_COLOR)

    plt.tight_layout()

    filename = os.path.join(output_dir, f"step_{t:03d}.png")
    plt.savefig(filename)
    plt.close(fig)