import time
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO, DQN
import supersuit as ss
import evacuation_v1
import numpy as np
import keyboard
import constants as const


def get_distance(p1, p2):
    return np.sqrt(np.power(p2[0] - p1[0], 2) + np.power(p2[1] - p1[1], 2))


def check_los(p1, p2, space):
    failure = False
    i, j = p2

    # Get line from person to robot (idx 0 is y, idx 1 is x)
    m = (p2[0] - p1[0]) / (p2[1] - p1[1] + 1e-12)
    b = p2[0] - m * p2[1]

    a = -m
    c = -b
    b = 1

    for k in range(
        min([p2[0], p1[0]]),
        max([p2[0], p1[0]]) + 1,
    ):
        for l in range(
            min([p2[1], p1[1]]),
            max([p2[1], p1[1]]) + 1,
        ):
            if space[k][l] == 1:
                d = abs(a * l + b * k + c) / ((a ** 2 + b ** 2) ** 0.5)

                if d < (2 ** 0.5 / 2 - 0.01):
                    failure = True
                    break
            if failure:
                break
        if failure:
            break

    return not failure


next = False
end = False


def get_next(a):
    global next
    next = True


def run_exit(a):
    global end
    end = True


keyboard.on_press_key("n", get_next)
keyboard.on_press_key("q", run_exit)

env = evacuation_v1.env(despawn=False)
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

model = DQN.load("evac_policy_10.zip")

all_steps = []
all_percent_exit = []
for i in range(100):
    env.reset()
    steps = 0
    percent_exited = 0
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        # if reward == -1:
        # print(reward, act)
        human_dones, human_positions, exits = env.render(mode="none")
        time.sleep(0.01)
        space = env.state()
        if not done:
            steps += 1
        if all(value for value in env.dones.values()):
            num_dones = 0
            num_dones += list(human_dones.values()).count(True)
            for human_id in human_positions:
                if human_dones[human_id]:
                    continue
                near_exit = False
                for ex in exits:
                    if get_distance(
                        ex, human_positions[human_id]
                    ) <= const.HUMAN_VISION and check_los(
                        ex, human_positions[human_id], space
                    ):
                        near_exit = True

                if near_exit:
                    num_dones += 1

            percent_exited = num_dones / len(human_dones) * 100
            break
        # print(human_dones)
        # print(list(human_dones.values()).count(False))
        if next or end:
            # if not list(human_dones.values()).count(False) or next:
            # next = False
            # break
            next = False
            break

    all_steps.append(steps)
    all_percent_exit.append(percent_exited)
    # percent_exited = list(human_dones.values()).count(True) / len(human_dones) * 100

    print("-------------------------------------")
    print(f"Percent Exited: {percent_exited}")
    print(f"Steps: {steps}")
    print("-------------------------------------")

    if end:
        break

print("average steps: {}".format(np.mean(all_steps)))
print("average percent exited: {}".format(np.mean(all_percent_exit)))
print("std steps: {}".format(np.std(all_steps)))
print("std percent exited: {}".format(np.std(all_percent_exit)))
print("done")
