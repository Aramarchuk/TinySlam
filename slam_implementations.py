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

        search_range_v = cfg.MOTION_NOISE_V * 2
        search_range_theta = cfg.MOTION_NOISE_OMEGA

        best_score = float("inf")

        dv = 0.0
        dtheta = 0.0
        theta = self.direction + dtheta
        new_x = self.x_est
        new_y = self.y_est
        score = 0
        for el in lidar_hits_local:
            el = (np.clip(el[0], -cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE),
                  np.clip(el[1], -cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE))
            x_hit = np.clip(new_x + el[0] * math.cos(theta) - el[1] * math.sin(theta), -1, self.N)
            y_hit = np.clip(new_y + el[0] * math.sin(theta) + el[1] * math.cos(theta), -1, self.N)

            points = bresenham(new_x, new_y, x_hit, y_hit)[:-1]
            for point in points:
                xp, yp = int(np.clip(point[0], 0, self.N - 1)), int(np.clip(point[1], 0, self.N - 1))
                score += (cfg.TS_NO_OBSTACLE - self.robot_map[yp, xp])
            score /= len(points)

            if el[0] in (-cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE) or el[1] in (-cfg.LIDAR_MAX_RANGE,
                                                                                 cfg.LIDAR_MAX_RANGE):
                score += (cfg.TS_NO_OBSTACLE - self.robot_map[
                    int(np.clip(y_hit, 0, self.N - 1)), int(np.clip(x_hit, 0, self.N - 1))])
            else:
                score += self.robot_map[int(np.clip(y_hit, 0, self.N - 1)), int(np.clip(x_hit, 0, self.N - 1))]
        if score < best_score:
            best_score = score
            best_dv = dv
            best_dtheta = dtheta
        for dtheta in np.linspace(-search_range_theta, search_range_theta, cfg.TS_N_PARTICLES):
            for dv in np.linspace(-search_range_v, search_range_v, cfg.TS_N_PARTICLES):
                theta = self.direction + dtheta
                new_x = self.x_est + dv * math.cos(theta)
                new_y = self.y_est + dv * math.sin(theta)
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

                    if el[0] in (-cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE) or el[1] in (-cfg.LIDAR_MAX_RANGE, cfg.LIDAR_MAX_RANGE):
                        score += (cfg.TS_NO_OBSTACLE - self.robot_map[int(np.clip(y_hit, 0, self.N - 1)), int(np.clip(x_hit, 0, self.N - 1))])
                    else:
                        score += self.robot_map[int(np.clip(y_hit, 0, self.N - 1)), int(np.clip(x_hit, 0, self.N - 1))]
                if score < best_score:
                    best_score = score
                    best_dv = dv
                    best_dtheta = dtheta

        self.direction += best_dtheta
        self.x_est = np.clip(self.x_est + best_dv * math.cos(self.direction), 0, self.N - 1)
        self.y_est = np.clip(self.y_est + best_dv * math.sin(self.direction), 0, self.N - 1)


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

        iteration = kwargs['t']

        lidar_local_extended = lidar_hits_local.copy()

        #if iteration > 30:
        self.scan_match(lidar_local_extended, 1.0)

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
        super().__init__(initial_pose, N)
        self.direction = 0
        max_landmarks = 50
        self.state_size = 3 + 2 * max_landmarks  # x, y, theta + 2 per landmark
        self.state = np.zeros(self.state_size)
        self.state[0] = initial_pose[0]
        self.state[1] = initial_pose[1]
        self.state[2] = initial_pose[2]

        self.covariance = np.eye(self.state_size) * 0.1 # Covariance matrix
        self.covariance[0:3, 0:3] = np.diag([0.1, 0.1, 0.1])

        self.landmarks = {}
        self.next_landmark_id = 0

        self.Q = np.diag([cfg.MOTION_NOISE_V**2, cfg.MOTION_NOISE_OMEGA**2])
        self.R = np.diag([cfg.LM_NOISE_RANGE**2, cfg.LM_NOISE_BEARING**2])

    def update_landmark(self, landmark_id, range_meas, bearing_meas):
        idx = self.landmarks[landmark_id]
        landmark_x = self.state[3 + 2 * idx]
        landmark_y = self.state[4 + 2 * idx]

        robot_x, robot_y, robot_theta = self.state[0:3]

        dx = landmark_x - robot_x
        dy = landmark_y - robot_y
        predicted_range = np.sqrt(dx ** 2 + dy ** 2)
        predicted_bearing = np.arctan2(dy, dx) - robot_theta
        predicted_bearing = (predicted_bearing + np.pi) % (2 * np.pi) - np.pi

        z_pred = np.array([predicted_range, predicted_bearing])
        z_actual = np.array([range_meas, bearing_meas])
        innovation = z_actual - z_pred
        innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi

        H = np.zeros((2, self.state_size))
        H[0, 0] = -dx / predicted_range
        H[0, 1] = -dy / predicted_range
        H[0, 2] = 0
        H[0, 3 + 2 * idx] = dx / predicted_range
        H[0, 4 + 2 * idx] = dy / predicted_range

        H[1, 0] = dy / (predicted_range ** 2)
        H[1, 1] = -dx / (predicted_range ** 2)
        H[1, 2] = -1
        H[1, 3 + 2 * idx] = dy / (predicted_range ** 2)
        H[1, 4 + 2 * idx] = - dx / (predicted_range ** 2)

        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        self.covariance = (np.eye(self.state_size) - K @ H) @ self.covariance

        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi

    def add_landmark(self, landmark_id, range_meas, bearing_meas):
        robot_x, robot_y, robot_theta = self.state[0:3]
        landmark_x = robot_x + range_meas * np.cos(bearing_meas + robot_theta)
        landmark_y = robot_y + range_meas * np.sin(bearing_meas + robot_theta)

        theta = robot_theta
        s = math.sin(theta + bearing_meas)
        c = math.cos(theta + bearing_meas)
        Gx = np.array([[1, 0, -range_meas * s],
                       [0, 1, range_meas * c]])
        Gz = np.array([[c, -range_meas * s],
                       [s, range_meas * c]])

        P_rr = self.covariance[0:3, 0:3]
        P_ll = Gx @ P_rr @ Gx.T + Gz @ self.R @ Gz.T
        P_rl = P_rr @ Gx.T
        # вставляем P_ll в ковариацию на позицию нового landmark и P_rl в off-diagonal

        idx = self.next_landmark_id
        self.landmarks[landmark_id] = idx
        self.state[3 + 2 * idx] = landmark_x
        self.state[4 + 2 * idx] = landmark_y
        self.next_landmark_id += 1

        # self.covariance[3 + 2 * idx, 3 + 2 * idx] = 1.0
        # self.covariance[4 + 2 * idx, 4 + 2 * idx] = 1.0

        self.covariance[3 + 2 * idx: 5 + 2 * idx, 3 + 2 * idx: 5 + 2 * idx] = P_ll
        self.covariance[0:3, 3 + 2 * idx: 5 + 2 * idx] = P_rl
        self.covariance[3 + 2 * idx: 5 + 2 * idx, 0:3] = P_rl.T

        self.landmarks[landmark_id] = idx


    def prediction(self, odometry_input):
        v, omega, dt = odometry_input

        x, y, theta = self.state[0], self.state[1], self.state[2]

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        self.state[0], self.state[1], self.state[2] = x_new, y_new, theta_new

        # Якобиан по состоянию: F = ∂g/∂v |_(μ_{t-1}, u_t)
        F = np.array([
            [1.0, 0.0, -v * np.sin(theta) * dt],
            [0.0, 1.0, v * np.cos(theta) * dt],
            [0.0, 0.0, 1.0],
        ])

        # Якобиан по шумам управления: L = ∂g/∂w (w = [v, ω])
        L = np.array([
            [np.cos(theta) * dt, 0.0],
            [np.sin(theta) * dt, 0.0],
            [0.0, dt],
        ])

        F_big = np.eye(self.state_size)
        F_big[0:3, 0:3] = F

        L_big = np.zeros((self.state_size, 2))
        L_big[0:3, 0:2] = L

        # Полное обновление ковариации: P ← F P Fᵀ + L Q Lᵀ
        self.covariance = F_big @ self.covariance @ F_big.T + L_big @ self.Q @ L_big.T

    def correction_landmarks(self, observations):
        for obs_id, range_meas, bearing_meas in observations:
            if obs_id in self.landmarks:
                self.update_landmark(obs_id, range_meas, bearing_meas)
            else:
                self.add_landmark(obs_id, range_meas, bearing_meas)

    def update(self, odometry_input, landmark_observations, **kwargs):
        self.prediction(odometry_input)

        self.correction_landmarks(landmark_observations)

        self.x_est, self.y_est, self.theta_est = self.state[0:3]

