import itertools as it
from re import A
import time
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
    UP = 0
    UPRIGHT = 1
    RIGHT = 2
    DOWNRIGHT = 3
    DOWN = 4
    DOWNLEFT = 5
    LEFT = 6
    UPLEFT = 7
    STAY = 8


class Objects(IntEnum):
    EMPTY = 0
    WALL = 1
    PERSON = 2
    ROBOT = 3
    EXIT = 4


class Person:
    def __init__(self, pos, num, space):
        self.id = num

        self.reset(pos, space)

    def reset(self, pos, space):
        self.position = np.array(pos)
        space[pos] = Objects.PERSON
        self.last_act = Directions.STAY

    def update(self, space):
        # Find the closest robot and move in the direction of it

        # Find closest robot

        # Get list of positions of robots
        robots_pos = []
        robots_dist = []
        for i in range(space.shape[0]):
            for j in range(space.shape[1]):
                if space[i][j] == Objects.ROBOT:
                    failure = False
                    # Store robot position
                    robot_pos = (i, j)

                    # Get line from person to robot (idx 0 is y, idx 1 is x)
                    m = (robot_pos[0] - self.position[0]) / (
                        robot_pos[1] - self.position[1]
                    )
                    b = robot_pos[0]

                    a = -m
                    c = -b
                    b = 1

                    for k in range(space.shape[0]):
                        for l in range(space.shape[1]):
                            if space[k][l] == Objects.Wall:
                                d = abs(a * l + b * k + c) / ((a ** 2 + b ** 2) ** 0.5)

                                if d < (2 ** 0.5 - 0.01):
                                    failure = True
                                    break
                            if failure:
                                break
                        if failure:
                            break

                    if not failure:
                        # Add robot position to list
                        robots_pos.append(robot_pos)

                        # Get distance between robot and person
                        robot_dist = get_distance(self.position, robot_pos)

                        # Add robot distance to list
                        robots_dist.append(robot_dist)

        if len(robots_pos):
            # Find index of robot with minimum distance
            robot_idx = np.argmin(robots_dist)

            # Get robot position with minimum distance
            robot_pos = robots_pos[robot_idx]
            robot_dist = robots_dist[robot_idx]

        # Choose action that moves person closer to robot

        actions = []
        delta_dists = []
        new_poses = []

        # Iterate through possible actions
        for a in Directions:
            # Get new position of person
            new_pos = get_new_pos(a, self.position)

            # Ensure new position is empty
            if (
                space[tuple(new_pos.astype(int))] == Objects.EMPTY
                or space[tuple(new_pos.astype(int))] == Objects.EXIT
            ):
                # Add action to list
                actions.append(a)

                # Store new positions
                new_poses.append(new_pos)

                # Get change in distance to robot
                delta_dists.append(get_distance(new_pos, robot_pos))

        # Get desired action
        if len(actions) == 0:
            # Done move
            action = Directions.STAY

            # Keep same position
            new_pos = self.position
        else:
            # Find action with minimum distance
            action_idx = np.argmin(delta_dists)
            action = actions[action_idx]
            new_pos = new_poses[action_idx]

        # Update state and action
        self.last_act = action
        space[tuple(self.position.astype(int))] = Objects.EMPTY
        space[tuple(new_pos.astype(int))] = Objects.PERSON
        self.position = new_pos


class Robot:
    def __init__(self, pos, num, space):
        self.id = num

        self.reset(pos, space)
        self.dist_exp = 1
        # self.max_dist = get_distance([0,0], space.shape)
        self.max_dist = np.sqrt(2)

    def reset(self, pos, space):
        self.position = np.array(pos)
        space[pos] = Objects.ROBOT
        self.prev_dist = get_distance(self.position, np.array([0, 2]))
        self.last_act = Directions.STAY

    def update(self, action, space):
        # action = 0-8
        self.last_act = action

        newpos = np.zeros(self.position.shape)
        done = False
        reward = 0
        newpos = get_new_pos(action, self.position)
        if action == None:
            return 0, True

        if (
            np.min(newpos) < 0
            or newpos[0] >= const.MAP_HEIGHT
            or newpos[1] >= const.MAP_WIDTH
        ):
            # check out of bounds
            reward = const.OOB_PENALTY
        elif space[tuple(newpos.astype(int))] != Objects.EMPTY:
            # check collision
            if space[tuple(newpos.astype(int))] == Objects.EXIT:
                reward = const.EXIT_REWARD
                # print('exit')
                space[tuple(self.position.astype(int))] = Objects.EMPTY
                done = True
            else:
                reward = const.WALL_PENALTY
        else:
            # move normally
            space[tuple(self.position.astype(int))] = Objects.EMPTY
            space[tuple(newpos.astype(int))] = Objects.ROBOT
            self.position = newpos

            # add distance to goal to reward
            dist = get_distance(self.position, np.array([0, 2]))
            # R_goal = (1 - dist**self.dist_exp) - (1 - self.prev_dist**self.dist_exp)
            R_goal = (self.prev_dist - dist) / self.max_dist
            self.prev_dist = dist
            reward += R_goal
            reward += const.MOVE_PENALTY

        return reward, done


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    def __init__(self, despawn=True, max_cycles=const.MAX_CYCLES):
        EzPickle.__init__(self, despawn, max_cycles)

        pg.init()

        self.agents = []
        self.despawn = despawn
        self.max_cycles = max_cycles

        self.seed()
        self.closed = False

        self.metadata = {
            "render_modes": ["human"],
            "name": "evacuation_v1",
            "is_parallelizable": True,
            "render_fps": const.FPS,
        }

        self.state_space = spaces.Box(
            low=0, high=4, shape=((const.MAP_HEIGHT, const.MAP_WIDTH)), dtype=np.uint8
        )

        # initialize space
        self.space = np.zeros((const.MAP_HEIGHT, const.MAP_WIDTH), dtype="uint8")

        # TODO: initialize walls
        for r in range(0, const.MAP_HEIGHT):
            for c in range(0, const.MAP_WIDTH):
                if (
                    r == 0
                    or c == 0
                    or r == const.MAP_HEIGHT - 1
                    or c == const.MAP_WIDTH - 1
                ):
                    self.space[r, c] = Objects.WALL

        # randomly generate exit locations
        def randexit():
            side = self.rng.randint(4)
            if side == 0:
                val = self.rng.randint(1, const.MAP_WIDTH)
                pos = [(0, val - 1), (0, val)]
            if side == 1:
                val = self.rng.randint(1, const.MAP_WIDTH)
                pos = [(const.MAP_HEIGHT - 1, val - 1), (const.MAP_HEIGHT - 1, val)]
            if side == 2:
                val = self.rng.randint(1, const.MAP_HEIGHT)
                pos = [(val - 1, 0), (val, 0)]
            if side == 3:
                val = self.rng.randint(1, const.MAP_HEIGHT)
                pos = [(val - 1, const.MAP_WIDTH - 1), (val, const.MAP_WIDTH - 1)]

            return pos

        num = 0
        # while num < const.NUM_EXITS:
        # pos = randexit()
        # if (self.space[pos[0]] != Objects.EXIT or self.space[pos[1]] != Objects.EXIT):
        # num += 1
        # self.space[pos[0]] = Objects.EXIT
        # self.space[pos[1]] = Objects.EXIT
        self.space[0, 1] = Objects.EXIT
        self.space[0, 2] = Objects.EXIT

        self.space_init = copy.deepcopy(self.space)

        # generate random positions for each human
        human_positions = self._randpos(const.NUM_PEOPLE)

        # populate an array of humans (these aren't agents)
        self.humans = {}
        for num, pos in enumerate(human_positions):
            human = Person(pos, num, self.space)
            identifier = f"human_{num}"
            self.humans[identifier] = human

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
            self.observation_spaces[r] = spaces.Box(
                low=0, high=4, shape=const.ROBOT_OBSERV_SHAPE, dtype=np.uint8
            )

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
        if self.despawn:
            if self.dones[self.agent_selection]:
                return self._was_done_step(action)
        agent_id = self.agent_selection
        all_agents_updated = self._agent_selector.is_last()
        self.rewards = {agent: 0 for agent in self.agents}

        if agent_id in self.robots:
            agent = self.robots[agent_id]

        self.rewards[agent_id], self.dones[agent_id] = agent.update(action, self.space)

        if all_agents_updated:
            for human_id in self.humans:
                self.humans[human_id].update(self.space)

        self.frame += 1
        if self.frame == self.max_cycles:
            self.dones = dict(zip(self.agents, [True for _ in self.agents]))

        if self.rendering:
            pg.event.pump()

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent_id] = 0
        self._accumulate_rewards()
        if self.despawn:
            self._dones_step_first()

        # self.render()
        # time.sleep(0.05)

    def reset(self):
        # print('reset')
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

        # print('initial state:')
        # print(self.space)

    def render(self, mode="human"):
        if mode == "human":
            # print(self.space)
            if not self.rendering:
                self.rendering = True
                pg.display.init()
                self.screen = pg.Surface(
                    (
                        const.MAP_WIDTH * const.PIXEL_RESOLUTION,
                        const.MAP_HEIGHT * const.PIXEL_RESOLUTION,
                    )
                )
                self.display_screen = pg.display.set_mode(const.SCREEN_SIZE)

            res = const.PIXEL_RESOLUTION
            self.screen.fill((255, 255, 255))
            for r in range(0, const.MAP_HEIGHT):
                for c in range(0, const.MAP_WIDTH):
                    if self.space[r, c] == Objects.WALL:
                        color = (128, 128, 128)
                        pg.draw.rect(
                            self.screen,
                            color,
                            pg.Rect(c * res, r * res, c * res + res, r * res + res),
                        )
                    elif self.space[r, c] == Objects.EXIT:
                        color = (255, 0, 0)
                        pg.draw.rect(
                            self.screen,
                            color,
                            pg.Rect(c * res, r * res, c * res + res, r * res + res),
                        )
                    else:
                        color = (255, 255, 255)
                        pg.draw.rect(
                            self.screen,
                            color,
                            pg.Rect(c * res, r * res, c * res + res, r * res + res),
                        )

            # draw agents as circles and their last action as a line
            for agent_id in self.agents:
                if agent_id in self.robots and not self.dones[agent_id]:
                    agent = self.robots[agent_id]
                    color = (0, 255, 0)
                    r, c = tuple(agent.position)
                    pg.draw.circle(
                        self.screen,
                        color,
                        (c * res + res // 2, r * res + res // 2),
                        res // 2,
                    )
                    self._draw_agent_action(agent)

            for human_id in self.humans:
                human = self.humans[human_id]
                color = (0, 0, 255)
                r, c = tuple(human.position)
                pg.draw.circle(
                    self.screen,
                    color,
                    (c * res + res // 2, r * res + res // 2),
                    res // 2,
                )
                self._draw_agent_action(human)

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
            if pos not in positions and self.space[pos] == Objects.EMPTY:
                positions.append(pos)

        return positions

    def _draw_agent_action(self, agent):
        res = const.PIXEL_RESOLUTION
        action = agent.last_act
        color = (0, 0, 0)
        p1 = agent.position * res + res // 2
        if action == Directions.UP:
            p2 = p1 + np.array([-1, 0]) * res // 2
        elif action == Directions.UPRIGHT:
            p2 = p1 + np.array([-1, 1]) * res // 4
        elif action == Directions.RIGHT:
            p2 = p1 + np.array([0, 1]) * res // 2
        elif action == Directions.DOWNRIGHT:
            p2 = p1 + np.array([1, 1]) * res // 4
        elif action == Directions.DOWN:
            p2 = p1 + np.array([1, 0]) * res // 2
        elif action == Directions.DOWNLEFT:
            p2 = p1 + np.array([1, -1]) * res // 4
        elif action == Directions.LEFT:
            p2 = p1 + np.array([0, -1]) * res // 2
        elif action == Directions.UPLEFT:
            p2 = p1 + np.array([-1, -1]) * res // 4
        elif action == Directions.STAY:
            p2 = p1 + np.array([0, 0]) * res // 2

        pg.draw.line(self.screen, color, (p1[1], p1[0]), (p2[1], p2[0]))


def get_distance(p1, p2):
    return np.sqrt(np.power(p2[0] - p1[0], 2) + np.power(p2[1] - p1[1], 2))


def get_new_pos(action, position):
    if action == Directions.UP:
        new_pos = position + [-1, 0]
    elif action == Directions.UPRIGHT:
        new_pos = position + [-1, 1]
    elif action == Directions.RIGHT:
        new_pos = position + [0, 1]
    elif action == Directions.DOWNRIGHT:
        new_pos = position + [1, 1]
    elif action == Directions.DOWN:
        new_pos = position + [1, 0]
    elif action == Directions.DOWNLEFT:
        new_pos = position + [1, -1]
    elif action == Directions.LEFT:
        new_pos = position + [0, -1]
    elif action == Directions.UPLEFT:
        new_pos = position + [-1, -1]
    elif action == Directions.STAY:
        new_pos = position + [0, 0]
    elif action == None:
        new_pos = position
    return new_pos
