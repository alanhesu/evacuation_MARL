import time
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import knights_archers_zombies_v9, pistonball_v6
import supersuit as ss
import evacuation_v1

env = evacuation_v1.env()
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

model = PPO.load("evac_policy1")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    print(reward)
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
    time.sleep(0.05)

print('done')
