from abc import ABC, abstractmethod
from map_utils import create_robot_map
import numpy as np


class BaseSLAM(ABC):
    """
    Abstract base class for all grid-based SLAM algorithms.
    It stores the POSE ESTIMATE and the MAP ESTIMATE.
    """

    def __init__(self, initial_pose, N):
        """
        Initializes the SLAM algorithm.

        :param initial_pose: (x_start, y_start, theta_start)
        :param N: Map size (N x N)
        """
        # Internal state of the algorithm
        self.x_est, self.y_est, self.theta_est = initial_pose
        self.N = N

        # Each SLAM algorithm stores and builds its OWN map
        self.robot_map = create_robot_map(N)

    @abstractmethod
    def update(self, odometry_input, lidar_hits, **kwargs):
        """
        The main method. Called at every simulation step.

        :param odometry_input: (v_noisy, omega_noisy, dt) - odometry readings.
        :param lidar_hits: [(hit_x1, hit_y1), ...] - list of lidar hits.
        :param kwargs: Additional arguments
        """

    def get_pose(self):
        """Returns the current pose estimate."""
        return self.x_est, self.y_est, self.theta_est

    def get_map(self):
        """Returns the currently built map."""
        return self.robot_map


class BaseLandmarkSLAM(ABC):
    """
    Abstract base class for all LANDMARK-BASED SLAM algorithms.
    It stores the POSE ESTIMATE.
    """

    def __init__(self, initial_pose, N):
        """
        Initializes the SLAM algorithm.

        :param initial_pose: (x_start, y_start, theta_start)
        :param N: Map size (N x N) - used for visualization bounds
        """
        # Internal state of the algorithm
        self.x_est, self.y_est, self.theta_est = initial_pose
        self.N = N

        # We create an empty map to avoid breaking the visualizer,
        # which expects a 2D array.
        self._empty_map = np.zeros((N, N), dtype=int)

    @abstractmethod
    def update(self, odometry_input, landmark_observations, **kwargs):
        """
        The main method. Called at every simulation step.

        :param odometry_input: (v_noisy, omega_noisy, dt) - odometry readings.
        :param landmark_observations: [(id, range, bearing), ...] - list of landmark observations.
        :param kwargs: Additional arguments
        """
        pass

    def get_pose(self):
        """Returns the current pose estimate."""
        return self.x_est, self.y_est, self.theta_est

    def get_map(self):
        """Returns an empty map so the visualizer doesn't crash."""
        return self._empty_map