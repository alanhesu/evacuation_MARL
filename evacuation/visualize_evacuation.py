import time
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO, DQN
from pettingzoo.butterfly import knights_archers_zombies_v9, pistonball_v6
import supersuit as ss
import evacuation_v1
import numpy as np

env = evacuation_v1.env(despawn=False)
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

model = DQN.load("evac_policy1_good")

all_steps = []
for i in range(0, 10):
    env.reset()
    steps = 0
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        if reward == -1:
            print(reward, act)
        env.render()
        time.sleep(0.05)
        if not done:
            steps += 1
        if all(value for value in env.dones.values()):
            break

    all_steps.append(steps)
    print("steps: {}".format(steps))
print("average steps: {}".format(np.mean(all_steps)))
print("done")
