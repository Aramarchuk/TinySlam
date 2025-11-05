import numpy as np
import math

from fontTools.misc.arrayTools import vectorLength

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


def update_point(old_map, new_map, point, target, coeefficient=1.0):
    old = old_map[point[1], point[0]]
    new_map[point[1], point[0]] = old + (target - old) * ALPHA * coeefficient


class TinySLAM(BaseSLAM):
    """
    Implementation of tinySLAM with Monte-Carlo localization.
    """
    def __init__(self, initial_pose, N):
        super().__init__(initial_pose, N)
        self.direction = 0
        self.robot_map = np.full((N, N), 32750)
        self.odometry_input = (0, 0)
        self.scan_match_coefficient = 0

    def get_map(self):
        return self.robot_map / 65600.0

    def scan_match(self, lidar_hits_local, coef):
        best_dx = 0
        best_dy = 0
        best_dtheta = 0

        search_range_xy = cfg.TS_SEARCH_RANGE_XY
        search_range_theta = cfg.TS_SEARCH_RANGE_THETA

        best_score = float("inf")

        for dx in np.linspace(-search_range_xy, search_range_xy, cfg.TS_N_PARTICLES):
            for dy in np.linspace(-search_range_xy, search_range_xy, cfg.TS_N_PARTICLES):
                for dtheta in np.linspace(-search_range_theta, search_range_theta, cfg.TS_N_PARTICLES):
                    theta = self.direction + dtheta
                    new_x = self.x_est + dx
                    new_y = self.y_est + dy
                    score = 0
                    for el in lidar_hits_local:
                        el = (np.clip(el[0], -cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE), np.clip(el[1], -cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE))
                        x_hit = np.clip(new_x + el[0] * math.cos(theta) - el[1] * math.sin(theta), -1, self.N)
                        y_hit = np.clip(new_y + el[0] * math.sin(theta) + el[1] * math.cos(theta), -1, self.N)

                        points = bresenham(new_x, new_y, x_hit, y_hit)[:-1]
                        for point in points:
                            xp, yp = int(np.clip(point[0], 0, self.N - 1)), int(np.clip(point[1], 0, self.N - 1))
                            score += (cfg.TS_NO_OBSTACLE - self.robot_map[yp, xp])
                        score /= len(points)

                        score += self.robot_map[int(np.clip(y_hit, 0, self.N - 1)), int(np.clip(x_hit, 0, self.N - 1))]
                    if score < best_score:
                        best_score = score
                        best_dx = dx
                        best_dy = dy
                        best_dtheta = dtheta

        self.x_est = np.clip(self.x_est + best_dx * coef, 0, self.N - 1)
        self.y_est = np.clip(self.y_est + best_dy * coef, 0, self.N - 1)
        self.direction += best_dtheta * coef

    def get_absolute_lidar_points(self, lidar_hits_local):
        lidar_hits_local = np.clip(lidar_hits_local, -cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE)
        lidar_hits_global = []
        for el in lidar_hits_local:
            x_global = self.x_est + el[0] * math.cos(self.direction) - el[1] * math.sin(self.direction)
            y_global = self.y_est + el[0] * math.sin(self.direction) + el[1] * math.cos(self.direction)
            lidar_hits_global.append((x_global, y_global))
        return lidar_hits_global


    def update(self, odometry_input, lidar_hits_local, **kwargs):

        self.x_est += self.odometry_input[0] * math.cos(self.direction)
        self.y_est += self.odometry_input[0] * math.sin(self.direction)

        self.direction += self.odometry_input[1]

        # self.x_est = kwargs['gt_pose'][0]
        # self.y_est = kwargs['gt_pose'][1]
        # self.direction = kwargs['gt_pose'][2]

        lidar_local_extended = lidar_hits_local.copy()


        self.scan_match(lidar_local_extended, 1.0)
        self.scan_match_coefficient += 1

        new_map = self.robot_map.copy()
        old_map = self.robot_map

        for i in range(len(lidar_hits_local)):
            vector_len = math.sqrt(lidar_hits_local[i][0] ** 2 + lidar_hits_local[i][1] ** 2)
            if vector_len != float("inf") and vector_len > 0.001:
                lidar_local_extended[i] = ((lidar_hits_local[i][0]) / vector_len * (vector_len + cfg.TS_EXTEND_LIDAR), (lidar_hits_local[i][1]) / vector_len * (vector_len + cfg.TS_EXTEND_LIDAR))

        lidar_hits_global = self.get_absolute_lidar_points(lidar_local_extended)

        for i in range(len(lidar_hits_global)):
            xp, yp = lidar_hits_global[i][0], lidar_hits_global[i][1]

            points = bresenham(self.x_est, self.y_est, xp, yp)
            for point in points[:-1]:
                if 0 <= point[0] < self.N and 0 <= point[1] < self.N:
                    update_point(old_map, new_map, point, cfg.TS_NO_OBSTACLE)

            if lidar_hits_local[i][0] != float("inf") and lidar_hits_local[i][1] != float("inf"):
                point = points[-1]
                point_clipped = (np.clip(point[0], 0, self.N - 1), np.clip(point[1], 0, self.N - 1))
                update_point(old_map, new_map, point_clipped, cfg.TS_OBSTACLE)

                dist = math.sqrt(lidar_hits_local[i][0] ** 2 + lidar_hits_local[i][1] ** 2)
                add = TS_HOLE_WIDTH / (2 * dist)
                x2, y2 = lidar_hits_local[i][0] * (1 + add), lidar_hits_local[i][1] * (1 + add)

                global_hole = self.get_absolute_lidar_points([(x2, y2)])[0]

                points = bresenham(
                    xp,
                    yp,
                    np.clip(global_hole[0], -1, self.N),
                    np.clip(global_hole[1], -1, self.N)
                )
                for j in range(len(points)):
                    point = points[j]
                    point_clipped = (np.clip(point[0], 0, self.N - 1), np.clip(point[1], 0, self.N - 1))
                    coefficient = 1.0 - j / len(points)
                    update_point(old_map, new_map, point_clipped, cfg.TS_OBSTACLE, coeefficient=coefficient)


        self.odometry_input = odometry_input

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
