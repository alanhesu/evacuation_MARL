from stable_baselines3 import PPO
from pettingzoo.magent import battlefield_v4
import supersuit as ss

# rendering 
env = battlefield_v4.env()
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

# env.reset()

# while 1:
#     env.render()

model = PPO.load("bf_policy")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()

