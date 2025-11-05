import numpy as np

# --- Simulation ---
N = 50                  # map size
steps = 300           # total steps
snapshot_every = 20    # save frame every N steps
OUTPUT_DIR = "sim_steps" # output folder for results
RANDOM_SEED = 42
LOG_FILE = "simulation_log.csv"

# --- LIDAR ---
LIDAR_MAX_RANGE = 30.0  # maximum ray range (in cells)
# Ray angles relative to the robot: 0=forward, pi/2=left, pi=back, -pi/2=right
NUMBER_OF_RAYS = 8
LIDAR_ANGLES_RELATIVE = np.array([i * np.pi / NUMBER_OF_RAYS * 2 for i in range(NUMBER_OF_RAYS)])
LIDAR_STEP_SIZE = 0.1 # raycasting precision/step
# --- Robot Motion ---
x_start, y_start = 28, 22       # start position
theta_start = 0.0                   # start angle (facing right)
v = 0.35                             # linear velocity (cells/step)
omega = 0.06                        # angular velocity (rad/step)
dt = 1.0                            # time step
MARGIN = 1.2                        # margin from walls for clamping

ALPHA = 0.4                        # map update rate

# --- Visualization ---
GT_COLOR = "magenta"
EST_COLOR = "cyan"

# --- Motion Noise ---
MOTION_NOISE_V = 0.05
MOTION_NOISE_OMEGA = 0.02

TS_OBSTACLE = 0
TS_NO_OBSTACLE = 65500
TS_INIT_VAL = (TS_OBSTACLE + TS_NO_OBSTACLE) // 2
TS_QUALITY = 16
TS_HOLE_WIDTH = 3.0
TS_EXTEND_LIDAR = 0.75

# --- Monte Carlo ---
TS_N_PARTICLES = 11
TS_SEARCH_RANGE_XY = 0.01
TS_SEARCH_RANGE_THETA = 0.01

# --- LandMark Sensor ---

LM_SENSOR_MAX_RANGE = 5.0
LM_SENSOR_FOV = np.pi
LM_NOISE_RANGE = 0.05
LM_NOISE_BEARING = 0.02
ENABLE_NOISE = True