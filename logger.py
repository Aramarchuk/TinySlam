import os
import csv
import datetime
import numpy as np

import config as cfg
import landmark_utils
from slam_core import BaseLandmarkSLAM
from map_utils import create_real_map  # <-- НОВЫЙ ИМПОРТ



def setup_run_environment():
    """Creates a unique output directory and log file path based on a timestamp."""
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(cfg.OUTPUT_DIR, f"run_{timestamp_str}")

    log_file_base = os.path.splitext(cfg.LOG_FILE)[0]
    log_file_ext = os.path.splitext(cfg.LOG_FILE)[1]
    log_file_name = f"{log_file_base}_{timestamp_str}{log_file_ext}"

    os.makedirs(output_dir, exist_ok=True)
    log_filepath = os.path.join(output_dir, log_file_name)

    return output_dir, log_filepath


def setup_logging(log_filepath, slam_class):
    """Opens the CSV file and writes the headers."""
    log_file = open(log_filepath, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)

    headers = [
        "step",
        "x_gt", "y_gt", "theta_gt",
        "x_est", "y_est", "theta_est",
        "is_in_obstacle"
    ]

    if issubclass(slam_class, BaseLandmarkSLAM):

        print("Logger: Auto-detecting number of landmarks...")
        temp_grid_map = create_real_map(cfg.N)
        temp_lm_map = landmark_utils.create_landmarks_from_map(temp_grid_map)
        num_landmarks = len(temp_lm_map)
        print(f"Logger: Found {num_landmarks} landmarks. Creating headers.")

        for i in range(num_landmarks):
            headers.extend([f"obs_{i}_id", f"obs_{i}_range", f"obs_{i}_bearing"])
    else:

        for i in range(len(cfg.LIDAR_ANGLES_RELATIVE)):
            headers.extend([f"hit_{i}_x", f"hit_{i}_y"])
        for i in range(len(cfg.LIDAR_ANGLES_RELATIVE)):
            headers.extend([f"local_hit_{i}_x", f"local_hit_{i}_y"])

    log_writer.writerow(headers)
    return log_file, log_writer



def log_current_state(log_writer, t, pose_gt, pose_est, gt_map_data,
                      hit_points, local_hit_points, observations,
                      map_type, N):
    """Writes one row of data to the CSV log."""
    x, y, theta = pose_gt
    x_est, y_est, theta_est = pose_est

    is_in_obstacle = False
    if map_type == "grid":
        ix, iy = int(round(x)), int(round(y))
        ix_c, iy_c = np.clip(ix, 0, N - 1), np.clip(iy, 0, N - 1)
        # 0.0 is a wall in the ground truth map
        is_in_obstacle = (gt_map_data[iy_c, ix_c] == 0.0)

    row_data = [t, x, y, theta, x_est, y_est, theta_est, is_in_obstacle]

    if map_type == "landmark":
        # gt_map_data in "landmark" mode is the list of landmarks
        num_landmarks = len(gt_map_data)

        # Create a "blank" list for all possible observations
        obs_data_full = [""] * (num_landmarks * 3)

        # Fill in the data at the correct index based on the landmark ID
        for (lm_id, lm_range, lm_bearing) in observations:
            base_idx = lm_id * 3

            # Check if the ID is valid
            if 0 <= base_idx < len(obs_data_full):
                obs_data_full[base_idx] = lm_id
                obs_data_full[base_idx + 1] = lm_range
                obs_data_full[base_idx + 2] = lm_bearing

        row_data.extend(obs_data_full)

    else:
        # Logic for grid-based SLAM
        flat_hits = [coord for point in hit_points for coord in point]
        local_hit_points = [coord for point in local_hit_points for coord in point]
        row_data.extend(flat_hits)
        row_data.extend(local_hit_points)

    log_writer.writerow(row_data)