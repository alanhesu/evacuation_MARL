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

next = False
end = False


def get_next(a):
    global next
    next = True


def exit(a):
    global end
    end = True


keyboard.on_press_key("n", get_next)
keyboard.on_press_key("q", exit)

env = evacuation_v1.env(despawn=False)
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

model = DQN.load("evac_policy2")

all_steps = []
all_percent_exit = []
for i in range(30):
    env.reset()
    steps = 0
    percent_exited = 0
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        # if reward == -1:
            # print(reward, act)
        human_dones, human_positions, exits = env.render()
        # time.sleep(0.05)
        if not done:
            steps += 1
        if all(value for value in env.dones.values()):
            num_dones = 0
            num_dones += list(human_dones.values()).count(True)
            for human_id in human_positions:
                if (human_dones[human_id]):
                    continue
                near_exit = False
                for ex in exits:
                    if (get_distance(ex, human_positions[human_id]) <= const.HUMAN_VISION):
                        near_exit = True

                if (near_exit):
                    num_dones += 1

            percent_exited = num_dones / len(human_dones) * 100
            break
        # print(human_dones)
        # print(list(human_dones.values()).count(False))
        # if not list(human_dones.values()).count(False) or next:
            # next = False
            # break
        if next:
            next = False
            break

    all_steps.append(steps)
    all_percent_exit.append(percent_exited)
    # percent_exited = list(human_dones.values()).count(True) / len(human_dones) * 100

    print("-------------------------------------")
    print(f"Percent Exited: {percent_exited}")
    print(f"Steps: {steps}")
    print("-------------------------------------")


print("average steps: {}".format(np.mean(all_steps)))
print("average percent exited: {}".format(np.mean(all_percent_exit)))
print("done")
