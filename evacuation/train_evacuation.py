from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.butterfly import knights_archers_zombies_v9, pistonball_v6
from pettingzoo.utils import average_total_reward
import supersuit as ss
import evacuation_v1

log_dir = './log'
env = evacuation_v1.parallel_env(despawn=False)
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class='stable_baselines3')
env = VecMonitor(env, log_dir)
model = PPO('MlpPolicy', env, verbose=3, learning_rate=1e-4, n_steps=2048, batch_size=256, tensorboard_log=log_dir)
# model = DQN('MlpPolicy', env, verbose=3, learning_rate=1e-4, batch_size=4096, tensorboard_log='./log/')
model.learn(total_timesteps=1e6)
model.save("evac_policy1")

print('done')
