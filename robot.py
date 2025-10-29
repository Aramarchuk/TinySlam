import numpy as np


def clamp_inside_walls(x, y, N, margin):
    """Clamps coordinates (x, y) inside the walls of an N x N map with a margin."""
    x = max(margin, min(N - 1 - margin, x))
    y = max(margin, min(N - 1 - margin, y))
    return x, y


def integrate_motion(x, y, theta, v, omega, dt):
    """Updates the pose (x, y, theta) based on the motion model (one step)."""
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt
    return x_new, y_new, theta_new


def get_noisy_odometry(v, omega, v_noise_std, omega_noise_std):
    """
    Simulates "noisy" odometry readings.
    Adds Gaussian noise to the control commands.
    """
    v_noisy = v + np.random.normal(0.0, v_noise_std)
    omega_noisy = omega + np.random.normal(0.0, omega_noise_std)
    return v_noisy, omega_noisy