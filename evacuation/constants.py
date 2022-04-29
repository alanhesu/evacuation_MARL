import math

SCREEN_SIZE = (600, 600)

MAP_HEIGHT = 7
MAP_WIDTH = 7
FPS = 4
STEPS_PER_FRAME = 2
SPACE_STEP_DELTA = 1/(FPS*STEPS_PER_FRAME)
MAX_CYCLES = 10000

WALL_PENALTY = -1
MOVE_PENALTY = -.001
OOB_PENALTY = -1
EXIT_REWARD = 1

NUM_ROBOTS = 10
NUM_EXITS = 1

ROBOT_OBSERV_SHAPE = (MAP_HEIGHT, MAP_WIDTH)

