import os
import csv
import config as cfg
from main import run_simulation
from slam_implementations import GroundTruthSLAM


def test_simulation_with_ground_truth(tmp_path, monkeypatch):
    """
    Runs a full simulation with the 'cheater' GroundTruthSLAM.
    The test checks that:
    1. The simulation runs and creates output files (log and images).
    2. In the log file, the pose estimate (x_est, y_est) *always*
       exactly equals the real pose (x_gt, y_gt).
    """

    # --- 1. Arrange ---

    # `monkeypatch` is a special pytest fixture for safely
    # modifying settings just for this test.

    # Tell the simulator to save everything to 'tmp_path' (a temporary folder)
    # Instead of 'sim_steps_20251021...'
    monkeypatch.setattr(cfg, 'OUTPUT_DIR', str(tmp_path))

    # `setup_environment` in main.py still adds a timestamp.
    # That's fine, we'll just find the necessary files inside tmp_path.

    # Reduce the number of steps to make the test fast
    monkeypatch.setattr(cfg, 'steps', 10)
    monkeypatch.setattr(cfg, 'snapshot_every', 5)

    # --- 2. Act ---

    run_simulation(GroundTruthSLAM)

    # --- 3. Assert ---

    # `setup_environment` created a timestamped folder inside tmp_path,
    # e.g., 'tmp_path/sim_steps_20251021_112030'. Let's find it.
    subdirs = [d for d in os.listdir(tmp_path) if os.path.isdir(os.path.join(tmp_path, d))]
    assert len(subdirs) == 1, "One results folder should have been created"
    output_dir = os.path.join(tmp_path, subdirs[0])

    # 3a. Check that files were created
    files_in_output = os.listdir(output_dir)

    log_files = [f for f in files_in_output if f.endswith('.csv')]
    assert len(log_files) == 1, "One CSV log file should have been created"

    png_files = [f for f in files_in_output if f.endswith('.png')]
    # (steps / snapshot_every) + 1 (for step 0) = (10 / 5) + 1 = 3
    assert len(png_files) == 3, "3 PNG snapshots should have been created"

    # 3b. Check the log file contents
    log_filepath = os.path.join(output_dir, log_files[0])

    rows_processed = 0
    with open(log_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_processed += 1

            x_gt = float(row['x_gt'])
            y_gt = float(row['y_gt'])
            x_est = float(row['x_est'])
            y_est = float(row['y_est'])

            # MAIN CHECK:
            # Since this is GroundTruthSLAM, the estimate MUST be perfect
            assert x_gt == x_est, f"At step {row['step']}, x_gt ({x_gt}) != x_est ({x_est})"
            assert y_gt == y_est, f"At step {row['step']}, y_gt ({y_gt}) != y_est ({y_est})"

    # Check that the log has 11 rows of data (steps 0..10)
    assert rows_processed == (cfg.steps + 1), f"Expected {cfg.steps + 1} rows, but log has {rows_processed}"