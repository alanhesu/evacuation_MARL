from stable_baselines3 import PPO
from pettingzoo.magent import battlefield_v4
import supersuit as ss

env = battlefield_v4.parallel_env(map_size=80, minimap_mode=False, step_reward=-0.005, 
    dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, 
    max_cycles=1000, extra_features=False)
# env = ss.color_reduction_v0(env, mode='G')
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')

# set up model with Stable baselines
model = PPO("MlpPolicy", env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
model.learn(total_timesteps=200000)
model.save("bf_policy")
