import numpy as np
import config as cfg
from lidar import _cast_one_ray


def wrap_to_pi(angle):
    """Normalizes the angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def create_landmarks_from_map(gt_grid_map):
    """
    Automatically creates an “ideal” (ground truth) map of landmarks
    by finding inner corners on the obstacle map.

    :param gt_grid_map: 2D numpy array (1.0=free, 0.0=obstacle)
    :return: List [(lx, ly), ...] of landmark coordinates.

    """
    gt_landmark_map = []
    N = gt_grid_map.shape[0]

    # Iterate through all “internal” cells of the map without
    # going beyond the boundaries.

    for y in range(1, N - 1):
        for x in range(1, N - 1):

            # 1. A landmark can only be on the obstacle itself.
            if gt_grid_map[y, x] != 0.0:
                continue

            # 2. Checking neighbors to find the “inside corner”

            is_free_up = (gt_grid_map[y + 1, x] == 1.0)
            is_free_down = (gt_grid_map[y - 1, x] == 1.0)
            is_free_right = (gt_grid_map[y, x + 1] == 1.0)
            is_free_left = (gt_grid_map[y, x - 1] == 1.0)

            # --- We identify 4 types of internal angles ---
            if is_free_up and is_free_right and (not is_free_left) and (not is_free_down):
                gt_landmark_map.append((float(x) + 0.5, float(y) + 0.5))

            elif is_free_up and is_free_left and (not is_free_right) and (not is_free_down):
                gt_landmark_map.append((float(x) - 0.5, float(y) + 0.5))

            elif is_free_down and is_free_right and (not is_free_left) and (not is_free_up):
                gt_landmark_map.append((float(x) + 0.5, float(y) - 0.5))

            elif is_free_down and is_free_left and (not is_free_right) and (not is_free_up):
                gt_landmark_map.append((float(x) - 0.5, float(y) - 0.5))

    print(f"Auto-generated {len(gt_landmark_map)} landmarks from grid map.")
    return gt_landmark_map


def detect_landmarks(pose_gt, gt_landmark_map, gt_grid_map, rng=None):
    """
    Simulates an “honest” landmark sensor.
    Checks range, angle (FOV), and LINE OF SIGHT (occlusion).

    :param pose_gt: Ideal robot pose (x, y, theta)
    :param gt_landmark_map: List [(lx, ly), ...] of ideal landmark coordinates
    :param gt_grid_map: 2D array of wall/obstacle map (for occlusion checking)
    :param rng: (Optional) numpy.random.Generator for reproducible noise
    :return: List of observations [(id, range, bearing), ...]
    """
    x_r, y_r, theta_r = pose_gt
    observations = []
    N = gt_grid_map.shape[0]

    for i, (lx, ly) in enumerate(gt_landmark_map):

        dx = lx - x_r
        dy = ly - y_r
        # Ideal (GT) distance to landmark
        dist_to_landmark = np.sqrt(dx ** 2 + dy ** 2)
        # Ideal (GT) global angle to the landmark
        angle_to_lm_global = np.arctan2(dy, dx)

        # Check the sensor range
        if dist_to_landmark > cfg.LM_SENSOR_MAX_RANGE:
            continue

        # Check Field of View (FOV)
        bearing = wrap_to_pi(angle_to_lm_global - theta_r)
        if abs(bearing) > (cfg.LM_SENSOR_FOV / 2):
            continue

        # Checking the ‘Line of Sight’ (Occlusion)
        # Shoot a beam from the robot in the direction of the landmark
        (hit_x, hit_y) = _cast_one_ray(
            x_r, y_r, angle_to_lm_global,
            gt_grid_map, N,
            cfg.LM_SENSOR_MAX_RANGE,
            cfg.LIDAR_STEP_SIZE
        )

        # Calculate the distance to the first obstacle on the path
        dist_to_wall = np.sqrt((hit_x - x_r) ** 2 + (hit_y - y_r) ** 2)

        # Compare.
        if dist_to_wall < (dist_to_landmark - cfg.LIDAR_STEP_SIZE):
            continue  # The landmark is blocked by a wall

        # Landmark in sight. Add noise.

        if cfg.ENABLE_NOISE and rng is not None:
            range_noise = rng.normal(0.0, cfg.LM_NOISE_RANGE)
            bearing_noise = rng.normal(0.0, cfg.LM_NOISE_BEARING)
        else:
            range_noise = 0.0
            bearing_noise = 0.0

        noisy_range = dist_to_landmark + range_noise
        noisy_bearing = bearing + bearing_noise
        noisy_bearing = wrap_to_pi(noisy_bearing)

        # Add to ‘seen’ list
        observations.append((i, noisy_range, noisy_bearing))

    return observations