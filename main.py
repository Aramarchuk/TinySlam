import argparse
import os

import config as cfg
import logger
from map_utils import create_real_map
from robot import clamp_inside_walls, integrate_motion, get_noisy_odometry
import lidar

from slam_core import BaseLandmarkSLAM
from slam_implementations import GroundTruthSLAM
import visualizer
import landmark_utils

SLAM_ALGORITHMS = {
    "gt": GroundTruthSLAM,
    # YOUR CODE HERE
}


def initialize_state(slam_):
    """
    Initializes the world (both maps), robot (GT), SLAM algorithm, and path storage.
    """
    # Physical map (walls, obstacles)
    gt_grid_map = create_real_map(cfg.N) #
    # Landmark map (auto-generated from wall map)
    gt_landmark_map = landmark_utils.create_landmarks_from_map(gt_grid_map)

    # Initialise the robot and SLAM
    x, y, theta = cfg.x_start, cfg.y_start, cfg.theta_start
    initial_pose = (cfg.x_start, cfg.y_start, cfg.theta_start)
    slam_algo = slam_(initial_pose, cfg.N)

    # Determine the type of map based on the SLAM class.
    if isinstance(slam_algo, BaseLandmarkSLAM):
        map_type = "landmark"
    else:
        map_type = "grid"

    # Lists for storing history
    paths = {
        'gt_x': [], 'gt_y': [],
        'est_x': [], 'est_y': []
    }

    print(f"Using SLAM Algorithm: {slam_algo.__class__.__name__}")
    print(f"Map type: {map_type}")

    return gt_grid_map, gt_landmark_map, slam_algo, (x, y, theta), paths, map_type


def update_world_state(pose_gt, cfg):
    """
    Performs one motion step for the real robot in the world.
    """
    x, y, theta = pose_gt

    x_new, y_new, theta_new = integrate_motion(
        x, y, theta, cfg.v, cfg.omega, cfg.dt
    )
    x, y = clamp_inside_walls(x_new, y_new, cfg.N, cfg.MARGIN)
    theta = theta_new

    return x, y, theta


def run_simulation(slam_):
    output_dir, log_filepath = logger.setup_run_environment()

    log_file, log_writer = logger.setup_logging(log_filepath, slam_)

    # Get both maps from the initializer
    gt_grid_map, gt_landmark_map, slam_algo, pose_gt, paths, map_type = initialize_state(slam_)

    x, y, theta = pose_gt

    print(f"Starting simulation... Saving images to '{output_dir}/'")

    for t in range(cfg.steps + 1):
        pose_gt = (x, y, theta)

        # --- Simulate Sensors (map type dependent) ---
        hit_points = []
        local_hit_points = []
        observations = []

        if map_type == "landmark":
            observations = landmark_utils.detect_landmarks(
                pose_gt, gt_landmark_map, gt_grid_map #
            )
        else:
            hit_points = lidar.cast_all_rays(
                x, y, theta, gt_grid_map, cfg.N, #
                cfg.LIDAR_ANGLES_RELATIVE,
                cfg.LIDAR_MAX_RANGE,
                cfg.LIDAR_STEP_SIZE
            )
            local_hit_points = lidar.get_local_scan(
                x, y, theta, gt_grid_map, cfg.N, #
                cfg.LIDAR_ANGLES_RELATIVE,
                cfg.LIDAR_MAX_RANGE,
                cfg.LIDAR_STEP_SIZE
            )

        # --- SLAM Update Cycle ---
        v_noisy, omega_noisy = get_noisy_odometry(
            cfg.v, cfg.omega,
            cfg.MOTION_NOISE_V, cfg.MOTION_NOISE_OMEGA
        )
        odometry_input = (v_noisy, omega_noisy, cfg.dt)

        if map_type == "landmark":
            slam_algo.update(odometry_input, observations, gt_pose=pose_gt) #
        else:
            if isinstance(slam_algo, TinySLAM): #
                lidar_data_for_slam = local_hit_points
            else:
                lidar_data_for_slam = hit_points
            slam_algo.update(odometry_input, lidar_data_for_slam, gt_pose=pose_gt) #

        # --- Get Pose Estimate ---
        x_est, y_est, theta_est = slam_algo.get_pose()
        pose_est = (x_est, y_est, theta_est)

        # --- Logging ---

        gt_map_for_log = gt_landmark_map if map_type == "landmark" else gt_grid_map #

        logger.log_current_state(log_writer, t, pose_gt, pose_est, gt_map_for_log, #
                                 hit_points, local_hit_points, observations,
                                 map_type, cfg.N)

        # --- Store Path History ---
        paths['gt_x'].append(x)
        paths['gt_y'].append(y)
        paths['est_x'].append(x_est)
        paths['est_y'].append(y_est)

        # --- Visualization (map type dependent) ---
        if t % cfg.snapshot_every == 0:
            robot_map = slam_algo.get_map()

            if map_type == "landmark":
                visualizer.save_ekf_snapshot( #
                    t, output_dir, cfg,
                    gt_grid_map, gt_landmark_map, robot_map,
                    paths,
                    (x, y), (x_est, y_est)
                )
            else:
                visualizer.save_grid_based_snapshot(
                    t, output_dir, cfg,
                    gt_grid_map, robot_map,
                    paths,
                    (x, y), (x_est, y_est),
                    hit_points
                )
            print(f"Saved snapshot for step {t}")

        # --- Move (Update REAL world) ---
        if t < cfg.steps:
            x, y, theta = update_world_state(pose_gt, cfg)

    log_file.close()
    print(f"Simulation finished. {len(os.listdir(output_dir))} files saved in '{output_dir}/'.")
    print(f"Log data saved to '{log_filepath}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SLAM simulation.")

    parser.add_argument(
        "--slam",
        type=str,
        default="gt",
        choices=SLAM_ALGORITHMS.keys(),
        help="Name of the SLAM algorithm to run."
    )

    args = parser.parse_args()
    slam_name = args.slam
    slam_class_to_run = SLAM_ALGORITHMS[slam_name]

    print(f"--- Running simulation with algorithm: {slam_class_to_run.__name__} ---")

    run_simulation(slam_class_to_run)