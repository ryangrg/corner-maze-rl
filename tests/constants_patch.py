# constants.py

# Environment constants
# Coordinate transformation helpers
GRID_SIZE = 13
GRID_MAX_INDEX = GRID_SIZE - 1

def rotate_point(point):
    x, y = point
    return (GRID_MAX_INDEX - y, x)

def rotate_points(points):
    return [rotate_point(point) for point in points]

def rotate_dir(direction):
    return (direction + 1) % 4

def rotate_pose(pose):
    x, y, direction = pose
    rx, ry = rotate_point((x, y))
    return (rx, ry, rotate_dir(direction))

def rotate_poses(poses):
    return [rotate_pose(pose) for pose in poses]

# Position variables
CORNER_POSES = rotate_poses([
    (10, 2, 3),
    (10, 2, 0),
    (10, 10, 0),
    (10, 10, 1),
    (2, 10, 1),
    (2, 10, 2),
    (2, 2, 2),
    (2, 2, 3),
])
WELL_ENTRY_POSES_LEFT = rotate_poses([
    (10, 2, 0),
    (10, 10, 1),
    (2, 10, 2),
    (2, 2, 3),
])
WELL_ENTRY_POSES_RIGHT = rotate_poses([
    (10, 2, 3),
    (10, 10, 0),
    (2, 10, 1),
    (2, 2, 2),
])
INTERSECTIONS = rotate_points([
    (1, 1),
    (1, 11),
    (2, 2),
    (6, 2),
    (10, 2),
    (2, 6),
    (6, 6),
    (10, 6),
    (2, 10),
    (6, 10),
    (10, 10),
    (11, 1),
    (11, 11),
])
CORNERS = rotate_points([
    (10, 2),
    (10, 10),
    (2, 10),
    (2, 2),
])
WELL_EXIT_POSES = rotate_poses([
    (1, 1, 1),
    (11, 1, 2),
    (11, 11, 3),
    (1, 11, 0),
])

# Grid variables
BARRIER_LOCATIONS = rotate_points([
    (5, 2),
    (6, 3),
    (7, 2),
    (10, 5),
    (9, 6),
    (10, 7),
    (7, 10),
    (6, 9),
    (5, 10),
    (2, 7),
    (3, 6),
    (2, 5),
    (6, 5),
    (7, 6),
    (6, 7),
    (5, 6),
])
CUE_LOCATIONS = rotate_points([
    (6, 1),
    (11, 6),
    (6, 11),
    (1, 6),
])
TRIGGER_LOCATIONS = rotate_points([
    (4, 2),
    (8, 2),
    (10, 4),
    (10, 8),
    (8, 10),
    (4, 10),
    (2, 8),
    (2, 4),
    (6, 4),
    (8, 6),
    (6, 8),
    (4, 6),
])
WELL_LOCATIONS = rotate_points([
    (11, 1),
    (11, 11),
    (1, 11),
    (1, 1),
])

# Reward scoring variables for RL model
FORWARD_SCR = -0.001 #-0.001
TURN_SCR = -0.005 #-0.005
TURN_INTR_SCR = -0.005 #-0.005
WELL_REWARD_SCR = 1.061 #0.061 is added to net 1 for entering well and leaving well after punishment
WELL_EMPTY_SCR = -0.005 #-0.005
INAPPROPRIATE_ACTION_SCR = -0.005 #-0.005
TIME_OUT_SCR = -1
REVISIT_SCR = 0
SAME_PLACE_SCR = 0
TOO_LONG_SCR = 55
TOO_LONG_IN_PHASE = -0.001

# Session scoring variables to track progress of agent
ACQUISITION_SESSION_TRIALS = 32

# View variables 
AGENT_VIEW_SIZE = 21
AGENT_VIEW_SIZE_SCALE = 1
VIEW_TILE_SIZE = 1
AGENT_VIEW_BEHIND = 7
CELL_VIEW_BEHIND = 7

# Realtime viewing variables
RENDER_FPS = 30

