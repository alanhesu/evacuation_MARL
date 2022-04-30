import math

SCREEN_SIZE = (600, 600)
PIXEL_RESOLUTION = 30

MAP_HEIGHT = 15
MAP_WIDTH = 15
OBSERVE_SIZE = 15  # must be zero or odd
FPS = 4
STEPS_PER_FRAME = 2
SPACE_STEP_DELTA = 1 / (FPS * STEPS_PER_FRAME)
MAX_CYCLES = 10000

WALL_PENALTY = -1
COLLISION_PENALTY = -1
MOVE_PENALTY = -0.1
EXIT_REWARD = 1

NUM_ROBOTS = 3
NUM_EXITS = 1
NUM_PEOPLE = 10

ROBOT_EXIT_RATIO = 2

PERSON_RAND = 0.25

HUMAN_VISION = 100
