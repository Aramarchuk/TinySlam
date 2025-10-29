import os
import csv
import numpy as np
import pytest
import config as cfg
import landmark_utils
import logger
from slam_implementations import EkfStubSLAM
import map_utils


def test_landmark_sensor_and_logging_logic(tmp_path, monkeypatch):
    """
    Tests the "Sensor + Logger" pipeline for landmark-based SLAM.
    It verifies that:
    1. setup_logging creates the correct headers for N landmarks.
    2. detect_landmarks correctly identifies visible landmarks (respecting FOV).
    3. log_current_state writes observation data to the correct columns in the CSV.
    """

    # Arrange

    # Define a custom test map
    TEST_MAP = [
        (2.0, 2.0),  # ID 0
        (2.0, 7.0),  # ID 1
        (7.0, 7.0),  # ID 2
        (7.0, 2.0)   # ID 3
    ]
    NUM_LANDMARKS = len(TEST_MAP)

    # Define a dummy grid map (all free space)
    # This makes detect_landmarks pass without occlusion tests
    TEST_GRID_MAP = np.ones((cfg.N, cfg.N), dtype=float)

    # Define robot pose
    ROBOT_POSE = (cfg.x_start, cfg.y_start, cfg.theta_start)

    # Patch functions to use test map and grid
    monkeypatch.setattr(map_utils, 'create_real_map', lambda N: TEST_GRID_MAP)
    monkeypatch.setattr(landmark_utils, 'create_landmarks_from_map', lambda grid_map: TEST_MAP)

    # Disable noise for deterministic test
    monkeypatch.setattr(np.random, 'normal', lambda loc, scale: 0.0)

    # Prepare log file path
    log_filepath = os.path.join(tmp_path, "test_log.csv")

    # Initialize logger
    log_file, log_writer = logger.setup_logging(log_filepath, EkfStubSLAM)

    # Act

    # Detect landmarks using the test map
    observations = landmark_utils.detect_landmarks(ROBOT_POSE, TEST_MAP, TEST_GRID_MAP)

    # Log state
    logger.log_current_state(
        log_writer,
        t=0,
        pose_gt=ROBOT_POSE,
        pose_est=(0, 0, 0),
        gt_map_data=TEST_MAP,
        hit_points=[],
        local_hit_points=[],
        observations=observations,
        map_type="landmark",
        N=cfg.N
    )
    log_file.close()

    # Assert

    # Read headers
    with open(log_filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)

    # Check that columns for all landmarks were created
    assert f"obs_{NUM_LANDMARKS - 1}_bearing" in headers
    assert "obs_3_bearing" in headers

    # Read data row
    with open(log_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # Expected values (noise-free)
    expected_range = np.sqrt(2.5 ** 2 + 2.5 ** 2)
    expected_bearing_2 = np.arctan2(2.5, 2.5)
    expected_bearing_3 = np.arctan2(-2.5, 2.5)

    # Assertions

    # Columns 0 and 1 should be empty
    assert row['obs_0_id'] == ""
    assert row['obs_0_range'] == ""
    assert row['obs_0_bearing'] == ""
    assert row['obs_1_id'] == ""
    assert row['obs_1_range'] == ""
    assert row['obs_1_bearing'] == ""

    # Columns 2 and 3 should contain valid data
    assert row['obs_2_id'] == "2"
    assert float(row['obs_2_range']) == pytest.approx(expected_range)
    assert float(row['obs_2_bearing']) == pytest.approx(expected_bearing_2)

    assert row['obs_3_id'] == "3"
    assert float(row['obs_3_range']) == pytest.approx(expected_range)
    assert float(row['obs_3_bearing']) == pytest.approx(expected_bearing_3)
