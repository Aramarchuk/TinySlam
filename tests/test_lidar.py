import numpy as np
from numpy.testing import assert_allclose
import lidar
import config as cfg
from map_utils import create_real_map

THE_IDEAL_DISTANCE = 4.0

def create_empty_map(N):
    real_map = np.ones((N, N), dtype=float)

    # Outer walls (value 0.0)
    real_map[0, :] = 0.0
    real_map[-1, :] = 0.0
    real_map[:, 0] = 0.0
    real_map[:, -1] = 0.0
    return real_map

def create_test_map(N):
    """
    Creates a "reference" N x N map right here,
    so the test doesn't depend on config.py or map_utils.py.
    """

    real_map = create_empty_map(N)
    # Internal obstacles (the same ones from map_utils)
    real_map[2:4, 2:4] = 0.0  # 2x2 block at (2,2)
    real_map[7, 3] = 0.0  # Single block at (3,7)
    real_map[5:8, 7] = 0.0  # 1x3 block at (7,5)-(7,7) <-- Our "problematic" block

    return real_map


def test_lidar_at_start_position():
    """
    Tests the lidar operation at the start position (4.5, 4.5, 0.0),
    using a locally created map.
    """
    print("--- Running test: test_lidar_at_start_position ---")

    # --- 1. Arrange ---

    # --- Parameters that were previously in config.py ---
    N = 10
    pose_x = 4.5
    pose_y = 4.5
    pose_theta = 0.0
    LIDAR_MAX_RANGE = 6.0
    LIDAR_ANGLES_RELATIVE = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    LIDAR_STEP_SIZE = 0.1
    # -------------------------------------------------

    # Create the map locally
    gt_map = create_test_map(N)

    # Reference values from the log (simulation_log_20251021_120028.csv)
    # These are the exact values we got after fixing round()
    expected_hits = [
        (6.5999999999999925, 4.5),  # Ray 0 (forward)
        (4.5, 8.599999999999985),  # Ray 1 (left)
        (0.49999999999999933, 4.5),  # Ray 2 (back)
        (4.5, 0.49999999999999933)  # Ray 3 (right)
    ]

    print(f"  Map: {N}x{N} (created locally)")
    print(f"  Pose: x={pose_x}, y={pose_y}, theta={pose_theta}")
    print(f"  Expecting {len(expected_hits)} hit points...")

    # --- 2. Act ---
    # Call the function under test
    hit_points = lidar.cast_all_rays(
        pose_x, pose_y, pose_theta,
        gt_map, N,
        LIDAR_ANGLES_RELATIVE,
        LIDAR_MAX_RANGE,
        LIDAR_STEP_SIZE
    )

    # --- 3. Assert ---
    try:
        # assert_allclose is a "smart" comparison for
        # floating-point arrays.
        assert_allclose(hit_points, expected_hits, rtol=1e-7, atol=0)

        # \033[92m...\033[0m — are codes for printing green text in the terminal
        print("\n  RESULT: \033[92mPASS\033[0m")
        print("  ✅ Received hit points match expected values.")

    except AssertionError as e:
        # \033[91m...\033[0m — codes for red text
        print("\n  RESULT: \033[91mFAIL\033[0m")
        print("  ❌ Detected points DO NOT MATCH expected values.")
        print(f"\n  Expected (list of {len(expected_hits)} points):")
        print(f"  {expected_hits}")
        print(f"\n  Received (list of {len(hit_points)} points):")
        print(f"  {hit_points}")

    print("--- Test finished ---")


def test_get_local_scan_hits():
    """
    Checks that get_local_scan returns the correct LOCAL
    coordinates when rays HIT walls.
    """
    N = 10
    gt_map = create_empty_map(10)
    x_gt, y_gt, theta_gt = 4.5, 4.5, 0.0
    angles_rel = np.array([0, np.pi / 2, np.pi, -np.pi / 2])

    local_hits = lidar.get_local_scan(
        x_gt, y_gt, theta_gt,
        gt_map, N,
        angles_rel,
        cfg.LIDAR_MAX_RANGE,
        cfg.LIDAR_STEP_SIZE
    )


    assert len(local_hits) == 4
    atol = cfg.LIDAR_STEP_SIZE


    assert np.isclose(local_hits[0][0], THE_IDEAL_DISTANCE, atol=atol)
    assert np.isclose(local_hits[0][1], 0.0, atol=atol)


    assert np.isclose(local_hits[1][0], 0.0, atol=atol)
    assert np.isclose(local_hits[1][1], THE_IDEAL_DISTANCE, atol=atol)


    assert np.isclose(local_hits[2][0], -THE_IDEAL_DISTANCE, atol=atol)
    assert np.isclose(local_hits[2][1], 0.0, atol=atol)


    assert np.isclose(local_hits[3][0], 0.0, atol=atol)
    assert np.isclose(local_hits[3][1], -THE_IDEAL_DISTANCE, atol=atol)


def test_get_local_scan_misses():
    """
    Checks that get_local_scan returns (inf, inf)
    for rays that do NOT hit walls (go to max_range)."""
    N = 20
    gt_map = np.ones((N, N), dtype=float)
    gt_map[0, :] = 0.0
    gt_map[N - 1, :] = 0.0
    gt_map[:, 0] = 0.0
    gt_map[:, N - 1] = 0.0

    x_gt, y_gt, theta_gt = 10.0, 10.0, 0.0

    angles_rel = np.array([0, np.pi / 2, np.pi, -np.pi / 2])

    local_hits = lidar.get_local_scan(
        x_gt, y_gt, theta_gt,
        gt_map, N,
        angles_rel,
        cfg.LIDAR_MAX_RANGE,
        cfg.LIDAR_STEP_SIZE
    )

    # All rays must ‘miss’ and return (inf, inf)
    assert local_hits[0] == (np.inf, np.inf)
    assert local_hits[1] == (np.inf, np.inf)
    assert local_hits[2] == (np.inf, np.inf)
    assert local_hits[3] == (np.inf, np.inf)

if __name__ == "__main__":
    test_lidar_at_start_position()
    test_get_local_scan_hits()
    test_get_local_scan_misses()
