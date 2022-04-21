from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import knights_archers_zombies_v9, pistonball_v6
import supersuit as ss
import evacuation_v1

env = evacuation_v1.parallel_env(despawn=False)
# env = pistonball_v6.env()
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class='stable_baselines3')
model = PPO('MlpPolicy', env, verbose=3)
model.learn(total_timesteps=20000)
model.save("evac_policy1")

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

print('done')
