import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO, A2C, DQN
<<<<<<< HEAD
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
=======
from stable_baselines3.common.vec_env import VecMonitor
>>>>>>> km_devel
from pettingzoo.butterfly import knights_archers_zombies_v9, pistonball_v6
from pettingzoo.utils import average_total_reward
import supersuit as ss
import evacuation_v1
# from callbacks import SaveOnBestTrainingRewardCallback

log_dir = './log'
<<<<<<< HEAD
timesteps = 1e5
env = evacuation_v1.parallel_env(despawn=False)
# env = Monitor(env, log_dir)
# env = pistonball_v6.env()
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 16, num_cpus=1, base_class='stable_baselines3')
# model = PPO('MlpPolicy', env, verbose=3, learning_rate=1e-4, n_steps=2048, batch_size=4096, tensorboard_log='./log/')
model = DQN('MlpPolicy', env, verbose=3, learning_rate=1e-4, batch_size=4096, tensorboard_log=log_dir, gradient_steps=-1)

# callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=timesteps)
model.save("evac_policy1")

# plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "policy")
# plt.show()

# Rendering

# env = pistonball_v6.env()
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

# model = PPO.load("policy")

# env.reset()
# for agent in env.agent_iter():
    # obs, reward, done, info = env.last()
    # act = model.predict(obs, deterministic=True)[0] if not done else None
    # env.step(act)
    # env.render()
=======
env = evacuation_v1.parallel_env(despawn=False)
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class='stable_baselines3')
env = VecMonitor(env, log_dir)
model = PPO('MlpPolicy', env, verbose=3, learning_rate=1e-4, n_steps=2048, batch_size=256, tensorboard_log=log_dir)
# model = DQN('MlpPolicy', env, verbose=3, learning_rate=1e-4, batch_size=4096, tensorboard_log='./log/')
model.learn(total_timesteps=1e6)
model.save("evac_policy1")
>>>>>>> km_devel

print('done')
