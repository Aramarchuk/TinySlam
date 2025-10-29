# Simple 2D SLAM Simulator

This project is a Python-based simulator for experimenting with 2D SLAM (Simultaneous Localization and Mapping) algorithms. It simulates a robot in a 2D environment, supporting both grid-based (like TinySLAM) and landmark-based (like EKFSLAM) algorithms.
## Features

* **2D Grid World:** Simulates a robot in a 2D environment with predefined obstacles.
* **Sensor Simulation:**
    * **Lidar:** Uses raycasting to simulate Lidar hits.
    * **Landmark Sensor**: Simulates detecting known landmarks, providing noisy range and bearing, and checking for occlusions (line-of-sight).
    * **Odometry:** Provides noisy odometry data (linear and angular velocity) to mimic real-world sensor drift.
* **Modular SLAM:** Easily swap different SLAM algorithms using abstract base classes (`BaseSLAM` for grid-based, `BaseLandmarkSLAM` for landmark-based).
* **Visualization:** Saves snapshot images at regular intervals, showing a side-by-side comparison of the Ground Truth (GT) map and the robot's estimated map.
* **Logging:** Logs detailed simulation data (GT pose, estimated pose, sensor data) to a CSV file for analysis.

## Included Algorithms

1.  **`GroundTruthSLAM`**: A "cheater" algorithm that uses the perfect, noise-free ground truth pose for localization. This is useful as a baseline and for testing the integrity of the simulator's mapping and logging.
2.  **`EkfStubSLAM`**: A "stub" implementation for landmark-based SLAM. It correctly receives landmark observations, but only uses odometry for localization (pending EKF logic).

## File Structure

* `main.py`: The main entry point for running the simulation. Handles the simulation loop, sensor updates, and calls the SLAM algorithm.
* `config.py`: Contains all simulation parameters (map size, steps, robot speed, sensor noise, etc.).
* `slam_core.py`: Defines the abstract base classes: BaseSLAM (grid) and BaseLandmarkSLAM (landmark).
* `slam_implementations.py`: Contains the concrete SLAM algorithms.
* `robot.py`: Utility functions for robot motion, including `integrate_motion` (kinematics), `get_noisy_odometry` (noise simulation), and `clamp_inside_walls`.
* `lidar.py`: Implements the Lidar simulation via raycasting (`cast_all_rays`).
* `landmark_utils.py`: Utilities for landmark-based SLAM, including auto-generating landmarks from a grid map and simulating the landmark sensor.
* `map_utils.py`: Helper functions for creating the Ground Truth map (`create_real_map`), the robot's map (`create_robot_map`), and drawing them (`draw_bw`).
* `visualizer.py`: Contains functions to generate output images, with different functions for grid-based (`save_grid_based_snapshot`) and landmark-based (`save_ekf_snapshot`) visualization.
* `logger.py`: Handles setting up the run directory and CSV log file, with dynamic headers based on the SLAM type.

## Tests 

* `test_simulation_integration.py`: A `pytest` integration test to verify the simulator's core loop and logging.
* `test_lidar.py`: Unit tests for lidar functions, verifying correct raycast hits (`cast_all_rays`) and local coordinate calculation (`get_local_scan`).
* `test_landmark_based_approach.py`: An integration test for the landmark-based sensor-to-logger pipeline, verifying sensor FOV logic and correct CSV column mapping.

## How to Run

**Run with GroundTruthSLAM:**
    ```bash
    python main.py --slam gt
    ```

## How to Test

Run the integration test using `pytest`:

```bash
pytest
```

## Output
The simulation creates a new, timestamped directory in the sim_steps/ folder for each run (e.g., sim_steps/run_20251021_140000/).

This directory will contain:

 - simulation_log_...csv: A CSV file with a detailed log of GT pose, estimated pose, and sensor data for each step.
 - step_XXX.png: Snapshot images generated ```snapshot_every```  steps (as defined in config.py).

## What to do?
- Implement **OdometryOnlySLAM** and **TinySLAM** algorithms in `slam_implementations.py` to see how they perform in this simulated environment!
  - **OdometryOnlySLAM** — a naive implementation that trusts the noisy odometry completely. It does not use Lidar data for pose correction, which clearly demonstrates the effect of accumulated "drift" over time.
  - **TinySLAM** — the algorithm we talked about during the first lecture
  - EKF Slam — the algorithm we talked about during the second lecture 
- Adjust `main.py` to run with different SLAM algorithm based on CLI arguments 

