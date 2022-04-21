import itertools as it
import math
import os
import copy
from enum import IntEnum, auto

import numpy as np
import pygame as pg
import pymunk as pm
from gym import spaces
from gym.utils import EzPickle, seeding
from pymunk import Vec2d

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import constants as const

class Directions(IntEnum):
    UP = auto()
    UPRIGHT = auto()
    RIGHT = auto()
    DOWNRIGHT = auto()
    DOWN = auto()
    DOWNLEFT = auto()
    LEFT = auto()
    UPLEFT = auto()
    STAY = auto()

class Objects(IntEnum):
    EMPTY = 0
    WALL = 1
    PERSON = 2
    ROBOT = 3
    EXIT = 4

class Robot():
    def __init__(self, pos, num, space):
        self.id = num

        self.reset(pos, space)

    def reset(self, pos, space):
        self.position = np.array(pos)
        space[pos] = Objects.ROBOT

    def update(self, action, space):
        # action = 0-8
        newpos = np.zeros(self.position.shape)
        done = False
        if (action == Directions.UP):
            newpos = self.position + [-1, 0]
        elif (action == Directions.UPRIGHT):
            newpos = self.position + [-1, 1]
        elif (action == Directions.RIGHT):
            newpos = self.position + [0, 1]
        elif (action == Directions.DOWNRIGHT):
            newpos = self.position + [1, 1]
        elif (action == Directions.DOWN):
            newpos = self.position + [1, 0]
        elif (action == Directions.DOWNLEFT):
            newpos = self.position + [1, -1]
        elif (action == Directions.LEFT):
            newpos = self.position + [0, -1]
        elif (action == Directions.UPLEFT):
            newpos = self.position + [-1, -1]
        elif (action == Directions.STAY):
            newpos = self.position + [0, 0]

        if (np.min(newpos) < 0 or newpos[0] >= const.MAP_HEIGHT or newpos[1] >= const.MAP_WIDTH):
            # check out of bounds
            reward = const.OOB_PENALTY
        elif (space[tuple(newpos.astype(int))] != Objects.EMPTY):
            # check collision
            if (space[tuple(newpos.astype(int))] == Objects.EXIT):
                reward = const.EXIT_REWARD
                space[tuple(self.position.astype(int))] = Objects.EMPTY
                done = True
            else:
                reward = const.WALL_PENALTY
        else:
            # move normally
            reward = const.MOVE_PENALTY
            space[tuple(self.position.astype(int))] = Objects.EMPTY
            space[tuple(newpos.astype(int))] = Objects.ROBOT
            self.position = newpos

        return reward, done


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv, EzPickle):
    def __init__(
            self,
            max_cycles = const.MAX_CYCLES
    ):
        EzPickle.__init__(
                self,
                max_cycles = const.MAX_CYCLES
        )

        pg.init()

        self.agents = []
        self.max_cycles = max_cycles

        self.seed()
        self.closed = False

        self.metadata = {
            'render_modes': ['human'],
            'name': 'evacuation_v1',
            'is_parallelizable': True,
            'render_fps': const.FPS,
        }


        self.state_space = spaces.Box(low=0, high=4, shape=((const.MAP_HEIGHT, const.MAP_WIDTH)), dtype=np.uint8)

        # initialize space
        self.space = np.zeros((const.MAP_HEIGHT, const.MAP_WIDTH), dtype='uint8')
        #TODO: initialize walls

        #TODO: initialize exits
        # randomly generate exit locations
        def randexit():
            side = self.rng.randint(4)
            if (side == 0):
                val = self.rng.randint(1, const.MAP_WIDTH)
                pos = [(0, val-1), (0, val)]
            if (side == 1):
                val = self.rng.randint(1, const.MAP_WIDTH)
                pos = [(const.MAP_HEIGHT-1, val-1), (const.MAP_HEIGHT-1, val)]
            if (side == 2):
                val = self.rng.randint(1, const.MAP_HEIGHT)
                pos = [(val-1, 0), (val, 0)]
            if (side == 3):
                val = self.rng.randint(1, const.MAP_HEIGHT)
                pos = [(val-1, const.MAP_WIDTH-1), (val, const.MAP_WIDTH-1)]

            return pos

        num = 0
        while num < const.NUM_EXITS:
            pos = randexit()
            if (self.space[pos[0]] != Objects.EXIT or self.space[pos[1]] != Objects.EXIT):
                num += 1
                self.space[pos[0]] = Objects.EXIT
                self.space[pos[1]] = Objects.EXIT

        self.space_init = copy.deepcopy(self.space)

        # generate random positions for each robot
        robot_positions = self._randpos(const.NUM_ROBOTS)

        # populate agents with robots given positions
        self.robots = {}
        for num, pos in enumerate(robot_positions):
            robot = Robot(pos, num, self.space)
            identifier = f"robot_{num}"
            self.robots[identifier] = robot
            self.agents.append(identifier)

        # populate action spaces for each agent
        self.action_spaces = {}
        for r in self.robots:
            self.action_spaces[r] = spaces.Discrete(9)

        # populate observation spaces for each agent
        self.observation_spaces = {}
        self.last_observation = {}

        for r in self.robots:
            self.last_observation[r] = None
            self.observation_spaces[r] = spaces.Box(low=0, high=4, shape=const.ROBOT_OBSERV_SHAPE, dtype=np.uint8)

        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.reset()

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)

    def observe(self, agent):
        observation = None
        # populate current observation
        observation = self.space

        self.last_observation[agent] = observation

        return observation

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def state(self):
        state = None
        # populate current state
        state = self.space

        return state

    def step(self, action):
        # if self.dones[self.agent_selection]:
            # return self._was_done_step(action)
        agent_id = self.agent_selection
        all_agents_updated = self._agent_selector.is_last()
        self.rewards = {agent: 0 for agent in self.agents}

        if agent_id in self.robots:
            agent = self.robots[agent_id]

        self.rewards[agent_id], self.dones[agent_id] = agent.update(action, self.space)

        # if all_agents_updated:
            # for _ in range(const.STEPS_PER_FRAME):

        self.frame += 1
        if self.frame == self.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))

        if (self.rendering):
            pg.event.pump()

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent_id] = 0
        self._accumulate_rewards()
        # self._dones_step_first()

    def reset(self):
        self.screen = pg.Surface(const.SCREEN_SIZE)
        self.done = False

        self.space = copy.deepcopy(self.space_init)

        robot_positions = self._randpos(const.NUM_ROBOTS)
        for i, r in enumerate(self.robots.values()):
            r.reset(robot_positions[i], self.space)

        self.agents = self.possible_agents[:]
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.rendering = False
        self.frame = 0
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def render(self, mode='human'):
        if mode == 'human':
            print(self.space)
            if not self.rendering:
                self.rendering = True
                pg.display.init()
                self.screen = pg.Surface((const.MAP_WIDTH, const.MAP_HEIGHT))
                self.display_screen = pg.display.set_mode(const.SCREEN_SIZE)

            self.screen.fill((255, 255, 255))
            for r in range(0, const.MAP_HEIGHT):
                for c in range(0, const.MAP_WIDTH):
                    if (self.space[r,c] == Objects.WALL):
                        color = (128, 128, 128)
                    elif (self.space[r,c] == Objects.PERSON):
                        color = (0, 0, 255)
                    elif (self.space[r,c] == Objects.ROBOT):
                        color = (0, 255, 0)
                    elif (self.space[r,c] == Objects.EXIT):
                        color = (255, 0, 0)
                    else:
                        color = (255, 255, 255)
                    self.screen.set_at((c, r), color)

            # scale up to display size
            scaled_win = pg.transform.scale(self.screen, self.display_screen.get_size())
            self.display_screen.blit(scaled_win, (0, 0))
            pg.display.flip()

    def close(self):
        if not self.closed:
            self.closed = True
            if self.rendering:
                pg.event.pump()
                pg.display.quit()
            pg.quit()

    def _randpos(self, num):
        # generate a series of random positions to spawn in empty spots
        positions = []
        while len(positions) < num:
            pos = tuple(self.rng.randint([0, 0], [const.MAP_HEIGHT, const.MAP_WIDTH]))
            if (pos not in positions and self.space[pos] == Objects.EMPTY):
                positions.append(pos)

        return positions
