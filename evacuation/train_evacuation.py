import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecMonitor

# from pettingzoo.butterfly import knights_archers_zombies_v9, pistonball_v6
from pettingzoo.utils import average_total_reward
import supersuit as ss
import evacuation_v1

# from callbacks import SaveOnBestTrainingRewardCallback

log_dir = "./log"
timesteps = 8e5
env = evacuation_v1.parallel_env(despawn=False)
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=8, base_class="stable_baselines3")
env = VecMonitor(env, log_dir)
# model = PPO('MlpPolicy', env, verbose=3, learning_rate=1e-4, n_steps=2048, batch_size=256, tensorboard_log=log_dir)
model = DQN(
    "MlpPolicy",
    env,
    verbose=3,
    learning_rate=1e-4,
    batch_size=4096,
    tensorboard_log="./log/",
    exploration_fraction=0.3,
    exploration_final_eps=0.1,
)
model.learn(total_timesteps=timesteps)
model.save("p")

print("done")
