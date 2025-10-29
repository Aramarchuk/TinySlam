import numpy as np
import math

from slam_core import BaseSLAM, BaseLandmarkSLAM
from robot import integrate_motion, clamp_inside_walls
import config as cfg

class GroundTruthSLAM(BaseSLAM):

    def __init__(self, initial_pose, N):
        super().__init__(initial_pose, N)

    def update(self, odometry_input, lidar_hits, **kwargs):

        gt_pose = kwargs.get('gt_pose')
        if gt_pose:
            self.x_est, self.y_est, self.theta_est = gt_pose
        else:
            print("WARNING: GroundTruthSLAM did not receive 'gt_pose'!")

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
        self.y_est += math.sin(self.direction   ) * v
        self.direction += omega

        self._update_map_from_pose()

    def _update_map_from_pose(self):
        ix, iy = int(round(self.x_est)), int(round(self.y_est))
        ix = np.clip(ix, 0, self.N - 1)
        iy = np.clip(iy, 0, self.N - 1)

        if self.robot_map[iy, ix] != 0.0:
            self.robot_map[iy, ix] = 1.0


class TinySLAM(BaseSLAM):
    """
    Implementation of tinySLAM with Monte-Carlo localization.
    """

    def __init__(self, initial_pose, N):
        super().__init__(initial_pose, N)

        # YOUR CODE HERE
        raise NotImplementedError

    def get_map(self):
        # YOUR CODE HERE
        raise NotImplementedError


    def update(self, odometry_input, lidar_hits_local, **kwargs):
        # YOUR CODE HERE
        raise NotImplementedError

    # Don't forget to update map!

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
