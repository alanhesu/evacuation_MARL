import math

SCREEN_SIZE = (600, 600)
PIXEL_RESOLUTION = 30

MAP_HEIGHT = 20
MAP_WIDTH = 20
OBSERVE_SIZE = 0  # must be zero or odd
MAX_CYCLES = 10000

WALL_PENALTY = -1
COLLISION_PENALTY = -1
MOVE_PENALTY = -1
EXIT_REWARD = 1

NUM_ROBOTS = 3
NUM_EXITS = 1
NUM_PEOPLE = 10

ROBOT_EXIT_RATIO = 2
PERSON_RAND = 0.25
COLLECT_DIST = 9

HUMAN_VISION = 7
