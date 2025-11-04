import numpy as np
import math

from config import TS_HOLE_WIDTH, ALPHA
from slam_core import BaseSLAM, BaseLandmarkSLAM
from robot import integrate_motion, clamp_inside_walls
import config as cfg
from map_utils import bresenham

class GroundTruthSLAM(BaseSLAM):

    def __init__(self, initial_pose, N):
        super().__init__(initial_pose, N)

    def update(self, odometry_input, lidar_hits, **kwargs):

        gt_pose = kwargs.get('gt_pose')
        if gt_pose:
            self.x_est, self.y_est, self.theta_est = gt_pose
        else:
            print("WARNING: GroundTruthSLAM did not receive 'gt_pose'!") # prezenhem

        self._update_map_from_pose()

    def _update_map_from_pose(self):

        ix, iy = int(round(self.x_est)), int(round(self.y_est))
        ix = np.clip(ix, 0, self.N - 1)
        iy = np.clip(iy, 0, self.N - 1)

        if self.robot_map[iy, ix] != 0.0:
            self.robot_map[iy, ix] = 1.0


class OdometryOnlySLAM(BaseSLAM):
    """
    An "honest" but naive SLAM.
    - Localization: Trusts only "noisy" odometry (will drift).
    - Mapping: Simply marks visited cells as free.
    """

    def __init__(self, initial_pose, N):
        super().__init__(initial_pose, N)
        self.direction = 0


    def update(self, odometry_input, lidar_hits, **kwargs):
        v, omega, gt_pose = odometry_input
        self.x_est += math.cos(self.direction) * v
        self.y_est += math.sin(self.direction) * v
        self.direction += omega

        self._update_map_from_pose()

    def _update_map_from_pose(self):
        ix, iy = int(round(self.x_est)), int(round(self.y_est))
        ix = np.clip(ix, 0, self.N - 1)
        iy = np.clip(iy, 0, self.N - 1)

        if self.robot_map[iy, ix] != 0.0:
            self.robot_map[iy, ix] = 1.0


def update_point(old_map, new_map, point, target):
    old = old_map[point[1], point[0]]
    new_map[point[1], point[0]] = old + (target - old) * ALPHA


class TinySLAM(BaseSLAM):
    """
    Implementation of tinySLAM with Monte-Carlo localization.
    """
    def __init__(self, initial_pose, N):
        super().__init__(initial_pose, N)
        self.direction = 0
        self.robot_map = np.full((N, N), 32750)

    def get_map(self):
        return self.robot_map / 65600.0

    def scan_match(self, lidar_hits_local):
        best_dx = 0
        best_dy = 0
        best_dtheta = 0

        search_range_xy = cfg.TS_SEARCH_RANGE_XY
        search_range_theta = cfg.TS_SEARCH_RANGE_THETA

        score = 0
        for el in lidar_hits_local:
            el = (np.clip(el[0], 0, self.N), np.clip(el[1], 0, self.N))
            xp = int(np.clip(self.x_est, 0, self.N - 1))
            yp = int(np.clip(self.y_est, 0, self.N - 1))

            score += self.robot_map[yp, xp]

        best_score = score

        for dx in np.linspace(-search_range_xy, search_range_xy, cfg.TS_N_PARTICLES):
            for dy in np.linspace(-search_range_xy, search_range_xy, cfg.TS_N_PARTICLES):
                for dtheta in np.linspace(-search_range_theta, search_range_theta, cfg.TS_N_PARTICLES):
                    score = 0
                    for el in lidar_hits_local:
                        el = (np.clip(el[0], 0, self.N), np.clip(el[1], 0, self.N))
                        xp = int(np.clip(self.x_est + dx + el[0] * math.cos(dtheta) - el[1] * math.sin(dtheta), 0, self.N - 1))
                        yp = int(np.clip(self.y_est + dy + el[0] * math.sin(dtheta) + el[1] * math.cos(dtheta), 0, self.N - 1))

                        score += self.robot_map[yp, xp]
                    if score > best_score:
                        best_score = score
                        best_dx = dx
                        best_dy = dy
                        best_dtheta = dtheta

        self.x_est = np.clip(self.x_est + best_dx, 0, self.N - 1)
        self.y_est = np.clip(self.y_est + best_dy, 0, self.N - 1)
        self.direction += best_dtheta


    def update(self, odometry_input, lidar_hits_local, **kwargs):

        self.scan_match(lidar_hits_local)

        new_map = self.robot_map.copy()
        old_map = self.robot_map
        for el in lidar_hits_local:
            xp, yp = np.clip(self.x_est + el[0], 0, self.N - 1), np.clip(self.y_est + el[1], 0, self.N - 1)
            points = bresenham(self.x_est, self.y_est, xp, yp)
            for point in points[:-1]:
                update_point(old_map, new_map, point, cfg.TS_NO_OBSTACLE)

            if el[0] != np.inf and el[1] != np.inf:
                update_point(old_map, new_map, points[-1], cfg.TS_OBSTACLE)

                dist = math.sqrt((el[0]) ** 2 + el[1] ** 2)
                add = TS_HOLE_WIDTH / (2 * dist)
                x2, y2 = el[0] * (1 + add), el[1] * (1 + add)

                points = bresenham(
                    xp,
                    yp,
                    np.clip(self.x_est + x2, 0, self.N - 1),
                    np.clip(self.y_est + y2, 0, self.N - 1)
                )
                for point in points[:-1]:
                    update_point(old_map, new_map, point, cfg.TS_OBSTACLE)

        self.x_est += odometry_input[0] * math.cos(self.direction)
        self.y_est += odometry_input[0] * math.sin(self.direction)

        self.direction += odometry_input[1]

        self.robot_map = new_map

class EkfStubSLAM(BaseLandmarkSLAM):
    """
    A stub implementation for EKF SLAM.
    - Localization: Trusts only noisy odometry (like OdometryOnlySLAM).
    - Mapping: Does not build a map.
    """

    def __init__(self, initial_pose, N):
        # YOUR CODE HERE
        raise NotImplementedError

    def update(self, odometry_input, landmark_observations, **kwargs):
        # YOUR CODE HERE
        raise NotImplementedError

    # Don't forget to update map!
