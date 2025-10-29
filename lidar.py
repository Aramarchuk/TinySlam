import numpy as np


def get_local_scan(x, y, theta, gt_map, N, lidar_angles_relative, lidar_max_range, lidar_step_size):
    """
    Returns a list [(x_local, y_local), ...] of points in the robot's coordinate system.
    For rays that have gone ‘into infinity’, returns (np.inf, np.inf).
    """
    local_hit_points = []


    global_angles = theta + lidar_angles_relative

    for rel_angle, angle_global in zip(lidar_angles_relative, global_angles):


        hit_x_world, hit_y_world = _cast_one_ray(
            x, y, angle_global,
            gt_map, N,
            lidar_max_range,
            lidar_step_size
        )

        dist = np.sqrt((hit_x_world - x) ** 2 + (hit_y_world - y) ** 2)

        is_max_range = np.isclose(dist, lidar_max_range, atol=lidar_step_size)

        if is_max_range:
            x_local = np.inf
            y_local = np.inf
        else:
            x_local = dist * np.cos(rel_angle)
            y_local = dist * np.sin(rel_angle)

        local_hit_points.append((x_local, y_local))

    return local_hit_points

def _cast_one_ray(x_start, y_start, angle_global, gt_map, N, max_range, step_size):
    """
    Casts a single ray from (x_start, y_start) at angle_global.
    Steps by step_size up to max_range.
    Returns (hit_x, hit_y) - the collision point or the end point of the ray.
    """

    # Step direction
    dx = np.cos(angle_global) * step_size
    dy = np.sin(angle_global) * step_size

    current_x, current_y = x_start, y_start

    # Iterate along the ray from 0 to max_range
    for _ in np.arange(0, max_range, step_size):

        current_x += dx
        current_y += dy

        ix = int(current_x + 0.5)
        iy = int(current_y + 0.5)

        # 1. Check if we went off the map
        if not (0 <= ix < N and 0 <= iy < N):
            return current_x - dx, current_y - dy # Return previous point

        # 2. Check for collision with a wall (wall == 0)
        if gt_map[iy, ix] == 0:
            return current_x, current_y # Return collision point

    # 3. The ray traveled the full max_range without hitting anything
    # Return the end point of the ray.
    return current_x, current_y


def cast_all_rays(x, y, theta, gt_map, N, lidar_angles_relative, lidar_max_range, lidar_step_size):
    """
    Casts all lidar rays from (x, y, theta) on the gt_map.

    Returns:
        hit_points (list of tuples): List of end points (hit_x, hit_y) for each ray.
    """
    hit_points = []

    # Calculate absolute angles for each ray
    global_angles = theta + lidar_angles_relative

    for angle_global in global_angles:
        hit_x, hit_y = _cast_one_ray(
            x, y, angle_global,
            gt_map, N,
            lidar_max_range,
            lidar_step_size
        )
        hit_points.append((hit_x, hit_y))

    return hit_points