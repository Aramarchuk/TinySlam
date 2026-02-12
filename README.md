# Simple 2D SLAM Simulator

**Course Project: Algorithm Engineering @ NUP**

This project is an educational 2D SLAM (Simultaneous Localization and Mapping) simulator developed for the *Algorithm Engineering* course at **NUP**. It is designed to help students understand and implement core SLAM concepts, covering both grid-based (e.g., TinySLAM) and landmark-based (e.g., EKF SLAM) approaches.

> **Note:** The core simulation mechanism (environment, sensor raycasting, and basic robot kinematics) is largely adapted/borrowed from existing educational resources to focus efforts on algorithm implementation.

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

1.  **`GroundTruthSLAM` (`gt`)**: A baseline "cheater" algorithm that uses the perfect, noise-free ground truth pose for localization. Useful for debugging/baseline.
2.  **`OdometryOnlySLAM` (`oo`)**: Uses only noisy odometry updates. Demonstrates the effect of drift without correction.
3.  **`TinySLAM` (`ts`)**: A grid-based SLAM implementation using particle filters/scan matching.
4.  **`EkfStubSLAM` (`ek`)**: A "stub" implementation for landmark-based SLAM (EKF logic to be implemented).

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

The simulation is controlled via the command line. Use the `--slam` argument to select the algorithm.

### 1. Ground Truth (Baseline)
Use the exact robot position (cheating) to verify map generation.
```bash
python main.py --slam gt
```

### 2. Odometry Only
Relies solely on noisy wheel encoders. Observe how the estimated position drifts from the ground truth over time.
```bash
python main.py --slam oo
```

### 3. TinySLAM (Grid-based)
Runs the TinySLAM (particle filter / scan matching) algorithm.
```bash
python main.py --slam ts
```

### 4. EKF Stub (Landmark-based)
Runs the framework for Extended Kalman Filter SLAM using landmark detections.
```bash
python main.py --slam ek
```

## How to Test

Run the integration test using `pytest`:

```bash
pytest
```

## Output
The simulation creates a new, timestamped directory in the `sim_steps/` folder for each run (e.g., `sim_steps/run_20251021_140000/`).

This directory will contain:

 - `simulation_log_...csv`: A CSV file with a detailed log of GT pose, estimated pose, and sensor data for each step.
 - `step_XXX.png`: Snapshot images generated every `snapshot_every` steps (as defined in `config.py`). 

