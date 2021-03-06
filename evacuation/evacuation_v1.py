import itertools as it
from re import A
import time
import math
import os
import copy
from enum import IntEnum, auto
import random

import numpy as np
import pygame as pg
from gym import spaces
from gym.utils import EzPickle, seeding
import cv2

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
    def __init__(self, pos, num, space, exits):
        self.id = num

        self.reset(pos, space, exits)
        self.exits = exits  # store the exits because they never change

    def reset(self, pos, space, exits):
        self.position = np.array(pos)
        space[pos] = Objects.PERSON
        self.last_act = Directions.STAY
        self.abandoned = False
        self.exits = exits  # store the exits because they never change

    def update(self, space, robot_positions):
        # Find the closest robot and move in the direction of it

        # Find closest robot

        # Get list of positions of robots
        min_pos = self.position
        min_dist = np.Inf

        positions = [tuple(x) for x in self.exits] + [
            value for value in robot_positions.values()
        ]

        for robot_pos in positions:
            # see if humans can see depending on their vision range
            if get_distance(self.position, robot_pos) > const.HUMAN_VISION:
                continue

            failure = False
            i, j = robot_pos

            # Get line from person to robot (idx 0 is y, idx 1 is x)
            m = (robot_pos[0] - self.position[0]) / (
                robot_pos[1] - self.position[1] + 1e-12
            )
            b = robot_pos[0] - m * robot_pos[1]

            a = -m
            c = -b
            b = 1

            for k in range(
                min([robot_pos[0], self.position[0]]),
                max([robot_pos[0], self.position[0]]) + 1,
            ):
                for l in range(
                    min([robot_pos[1], self.position[1]]),
                    max([robot_pos[1], self.position[1]]) + 1,
                ):
                    if space[k][l] == Objects.WALL:
                        d = abs(a * l + b * k + c) / ((a ** 2 + b ** 2) ** 0.5)

                        if d < (2 ** 0.5 / 2 - 0.01):
                            failure = True
                            break
                    if failure:
                        break
                if failure:
                    break

            if not failure:
                # Get distance between robot and person
                robot_dist = (
                    get_distance(self.position, robot_pos) + const.ROBOT_EXIT_RATIO
                    if space[i][j] == Objects.ROBOT
                    else get_distance(self.position, robot_pos)
                )

                if robot_dist < min_dist:
                    # Update minimum distance and position
                    min_dist = robot_dist
                    min_pos = robot_pos

        robot_action = Directions.STAY
        robot_delta_dist = 0
        robot_new_pos = self.position

        prob = random.random()
        if prob < const.PERSON_RAND:
            # Take random action

            # Get list of possible actions
            possible_actions = []
            possible_positions = []

            # Iterate through possible robot actions
            for a in Directions:
                # Get new position of person
                new_pos = get_new_pos(a, self.position)

                # Ensure new position is empty
                if space[tuple(new_pos.astype(int))] in [Objects.EMPTY, Objects.EXIT]:
                    possible_actions.append(a)
                    possible_positions.append(new_pos)

                # Allow person to stay
                if a == Directions.STAY:
                    possible_actions.append(a)
                    possible_positions.append(new_pos)

            # Randomly sample the list
            idx = random.randint(0, len(possible_actions) - 1)

            # Update robot_action and robot_new_pos
            robot_action = possible_actions[idx]
            robot_new_pos = possible_positions[idx]

        else:
            # Take best action

            # Iterate through possible robot actions
            for a in Directions:
                # Get new position of person
                new_pos = get_new_pos(a, self.position)

                dist_to_bot = get_distance(new_pos, min_pos)
                # print(dist_to_bot, dist_to_bot < 2 ** 0.5 + 1e-12)
                if (
                    min_dist < np.inf
                    and dist_to_bot < 2 ** 0.5 + 1e-12
                    and space[min_pos] == Objects.ROBOT
                ):
                    continue

                # Ensure new position is empty
                if space[tuple(new_pos.astype(int))] in [Objects.EMPTY, Objects.EXIT]:
                    d = get_distance(new_pos, min_pos) - get_distance(
                        self.position, min_pos
                    )
                    if d < robot_delta_dist:
                        robot_action = a
                        robot_delta_dist = d
                        robot_new_pos = new_pos

        # Update state and action
        done = False
        self.last_act = robot_action
        space[tuple(self.position.astype(int))] = Objects.EMPTY
        if space[tuple(robot_new_pos.astype(int))] == Objects.EMPTY:
            space[tuple(robot_new_pos.astype(int))] = Objects.PERSON
        else:
            done = True
        self.position = robot_new_pos

        return done


class Robot:
    def __init__(self, pos, num, space, exits):
        self.id = num

        self.reset(pos, space, exits)
        self.dist_exp = 1
        self.max_dist = np.sqrt(2)
        self.max_dist_space = np.sqrt(2) * space.shape[0]
        self.exits = exits  # store the exits because they never change

    def reset(self, pos, space, exits):
        self.position = np.array(pos)
        space[pos] = Objects.ROBOT
        self.prev_dist = get_distance(self.position, np.array([0, 2]))
        self.prev_hum_dist = np.sqrt(2) * space.shape[0]
        self.last_act = Directions.STAY
        self.exits = exits  # store the exits because they never change
        self.prev_hum_count = 0

    def update(self, action, space, count_exited, human_positions):
        # action = 0-8
        self.last_act = action

        # REWARD FUNCTION

        # a bunch of weights so we can write the reward function in one line
        w_exit = 0
        w_collide = 0
        w_wall = 0
        w_move_pen = 0
        w_collect = 0
        w_num_follow = 0
        w_goal = 0
        w_count_exited = 0
        w_hum_dist = 0

        newpos = np.zeros(self.position.shape)
        done = False
        reward = 0
        newpos = get_new_pos(action, self.position)
        R_goal = 0
        R_collect = 0
        R_num_follow = 0
        R_delta_hum = 0
        hum_diff = 0
        if action == None:
            return 0, True

        if space[tuple(newpos.astype(int))] != Objects.EMPTY:
            # check collision
            if space[tuple(newpos.astype(int))] == Objects.EXIT:
                w_exit = 1
                # print('exit')
                space[tuple(self.position.astype(int))] = Objects.EMPTY
                done = True
            elif space[tuple(newpos.astype(int))] == Objects.ROBOT:
                w_collide = 1
            else:
                w_wall = 1
        else:
            # move normally
            space[tuple(self.position.astype(int))] = Objects.EMPTY
            space[tuple(newpos.astype(int))] = Objects.ROBOT
            self.position = newpos

            if count_exited > 0:
                w_count_exited = 1

            # add distance to goal to reward
            # get the closest exit
            mindist = np.inf
            for ex in self.exits:
                dist = get_distance(self.position, ex)
                if dist < mindist:
                    mindist = dist
            dist = mindist
            R_goal = (self.prev_dist - dist) / self.max_dist
            self.prev_dist = dist

            # add number of nearby humans and average distance to reward
            close_dists = []
            maxdist_hum = -np.inf
            for pos in human_positions.values():
                dist = get_distance(pos, self.position)
                if dist > maxdist_hum:
                    maxdist_hum = dist
                if dist < const.COLLECT_DIST:
                    close_dists.append(dist)
            num_humans = len(close_dists)

            if num_humans == 0:
                R_num_follow = -1
                R_collect = -1
            else:
                R_num_follow = num_humans / len(human_positions) + 1e-12
                R_collect = np.mean(close_dists) / const.COLLECT_DIST

            # add distance to farthest human to reward
            R_delta_hum = (
                self.max_dist_space - maxdist_hum
            ) / self.max_dist_space - 0.5
            R_delta_hum = np.clip(R_delta_hum, -1, 1)

            w_collect = 0
            w_num_follow = 0
            w_hum_dist = 0

            w_goal = 1
            w_move_pen = 0

        weights = np.array(
            [
                w_exit,
                w_collide,
                w_wall,
                w_move_pen,
                w_collect,
                w_num_follow,
                w_goal,
                w_count_exited,
                w_hum_dist,
            ]
        )

        weights = weights / (np.sum(weights) + 1e-12)

        reward = (
            +weights[0] * const.EXIT_REWARD
            + weights[1] * const.COLLISION_PENALTY
            + weights[2] * const.WALL_PENALTY
            + weights[3] * const.MOVE_PENALTY
            + weights[4] * R_collect
            + weights[5] * R_num_follow
            + weights[6] * R_goal
            + weights[7] * count_exited
            + weights[8] * R_delta_hum
        )

        reward = np.clip(reward, -1, 1)

        return reward, done


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    def __init__(
        self, despawn=True, max_cycles=const.MAX_CYCLES, rand_exit=const.RANDOM_EXIT
    ):
        EzPickle.__init__(self, despawn, max_cycles, rand_exit)

        pg.init()

        self.agents = []
        self.despawn = despawn
        self.max_cycles = max_cycles
        self.rand_exit = rand_exit

        self.seed()
        self.closed = False

        self.metadata = {
            "render_modes": ["human", "none"],
            "name": "evacuation_v1",
            "is_parallelizable": True,
        }

        self.state_space = spaces.Box(
            low=0, high=4, shape=((const.MAP_HEIGHT, const.MAP_WIDTH)), dtype=np.uint8
        )

        # initialize space
        self.space = np.zeros((const.MAP_HEIGHT, const.MAP_WIDTH), dtype="uint8")
        pad = const.OBSERVE_SIZE // 2
        self.padded_space = (
            np.ones((const.MAP_HEIGHT + pad * 2, const.MAP_HEIGHT + pad * 2))
            * Objects.WALL
        )

        # initialize walls based on an image file
        img = cv2.imread("wall_1.jpg", cv2.IMREAD_GRAYSCALE)
        for r in range(0, const.MAP_HEIGHT):
            for c in range(0, const.MAP_WIDTH):
                if (
                    r == 0
                    or c == 0
                    or r == const.MAP_HEIGHT - 1
                    or c == const.MAP_WIDTH - 1
                ):
                    self.space[r, c] = Objects.WALL
                # elif img[r,c] < 200:
                # self.space[r,c] = Objects.WALL

        self.space_init = copy.deepcopy(self.space)

        self.exits = []

        # generate random positions for each human
        human_pos = self._randpos(const.NUM_PEOPLE)

        # populate an array of humans (these aren't agents)
        self.humans = {}
        self.human_positions = {}
        for num, pos in enumerate(human_pos):
            human = Person(pos, num, self.space, self.exits)
            identifier = f"human_{num}"
            self.humans[identifier] = human
            self.human_positions[identifier] = pos

        # generate random positions for each robot
        robot_pos = self._randpos(const.NUM_ROBOTS)

        # populate agents with robots given positions
        self.robots = {}
        for num, pos in enumerate(robot_pos):
            robot = Robot(pos, num, self.space, self.exits)
            identifier = f"robot_{num}"
            self.robots[identifier] = robot
            self.agents.append(identifier)

        # keep a dictionary of robot positions
        self.robot_positions = dict(zip(self.agents, robot_pos))

        # populate action spaces for each agent
        self.action_spaces = {}
        for r in self.robots:
            self.action_spaces[r] = spaces.Discrete(9)

        # populate observation spaces for each agent
        self.observation_spaces = {}
        self.last_observation = {}

        for r in self.robots:
            self.last_observation[r] = None
            if const.OBSERVE_SIZE == 0:
                obs_shape = (const.MAP_HEIGHT, const.MAP_WIDTH)
            else:
                obs_shape = (const.OBSERVE_SIZE, const.OBSERVE_SIZE)
            self.observation_spaces[r] = spaces.Box(
                low=0, high=4, shape=obs_shape, dtype=np.uint8
            )

        self.possible_agents = self.agents[:]
        self.possible_humans = self.humans
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.reset()

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)

    def observe(self, agent):
        observation = None
        # populate current observation
        if const.OBSERVE_SIZE == 0:
            # if 0, then just use the state space
            observation = self.space
        else:
            pad = const.OBSERVE_SIZE // 2
            self.padded_space[pad:-pad, pad:-pad] = self.space

            if agent in self.agents:
                pos = self.robots[agent].position
            r = pos[0]
            c = pos[1]
            observation = self.padded_space[
                r : r + const.OBSERVE_SIZE, c : c + const.OBSERVE_SIZE
            ]

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

        self.rewards[agent_id], self.dones[agent_id] = agent.update(
            action, self.space, self.count_exited, self.human_positions
        )
        if self.dones[agent_id]:
            self.robot_positions[agent_id] = (-1, -1)
        else:
            self.robot_positions[agent_id] = tuple(agent.position.astype(int))

        if all_agents_updated:
            before_count = list(self.human_dones.values()).count(True)
            for human_id in self.humans:
                if not self.human_dones[human_id]:
                    self.human_dones[human_id] = self.humans[human_id].update(
                        self.space, self.robot_positions
                    )
                    self.human_positions[human_id] = tuple(
                        self.humans[human_id].position.astype(int)
                    )
                else:
                    self.human_positions[human_id] = (-1, -1)

            after_count = list(self.human_dones.values()).count(True)

            self.count_exited = after_count - before_count

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

    def reset(self):
        # print('reset')
        self.screen = pg.Surface(const.SCREEN_SIZE)
        self.done = False

        self.count_exited = 0

        self.space = copy.deepcopy(self.space_init)

        # generate exits
        num = 0
        self.exits = []
        if self.rand_exit:
            while num < const.NUM_EXITS:
                pos = self._randexit()
                if (
                    self.space[pos[0]] != Objects.EXIT
                    or self.space[pos[1]] != Objects.EXIT
                ):
                    num += 1
                    self.exits.append([pos[0][0], pos[0][1]])
                    self.exits.append([pos[1][0], pos[1][1]])
        else:
            if const.NUM_EXITS >= 1:
                self.exits.append([0, 1])
                self.exits.append([0, 2])
            if const.NUM_EXITS >= 2:
                self.exits.append([const.MAP_HEIGHT - 1, const.MAP_WIDTH - 3])
                self.exits.append([const.MAP_HEIGHT - 1, const.MAP_WIDTH - 2])

        for ex in self.exits:
            self.space[tuple(ex)] = Objects.EXIT

        robot_pos = self._randpos(const.NUM_ROBOTS)
        self.robot_positions = dict(zip(self.agents, robot_pos))
        for i, r in enumerate(self.robots.values()):
            r.reset(robot_pos[i], self.space, self.exits)

        human_pos = self._randpos(const.NUM_PEOPLE)
        self.human_positions = dict(zip(list(self.humans.keys()), human_pos))
        for i, h in enumerate(self.humans.values()):
            h.reset(human_pos[i], self.space, self.exits)

        self.agents = self.possible_agents[:]
        self.humans = self.possible_humans
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.human_dones = dict(zip(self.humans, [False for _ in self.humans]))
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
                self.display_screen = pg.display.set_mode(const.SCREEN_SIZE, display=0)

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
                if not self.human_dones[human_id]:
                    human = self.humans[human_id]
                    color = (125, 125, 255)
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

        return self.human_dones, self.human_positions, self.exits

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
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

        pg.draw.line(self.screen, color, (p1[1], p1[0]), (p2[1], p2[0]), width=2)

    def _randexit(self):
        # randomly generate exit locations
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
