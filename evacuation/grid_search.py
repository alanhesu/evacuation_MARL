from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import knights_archers_zombies_v9, pistonball_v6
import supersuit as ss
import evacuation_v1

learning_rates = [1e-4]
n_steps = [2048]
batch_size= [256]
total_timesteps = [5e4]

env = evacuation_v1.parallel_env(despawn=False)
# env = pistonball_v6.env()
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')
model = PPO('MlpPolicy', env, verbose=3, learning_rate=1e-4, n_steps=2048, batch_size=256, tensorboard_log='./log/')
model.learn(total_timesteps=5e4)
model.save("evac_policy1")

print('done')
