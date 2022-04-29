from pettingzoo.test import performance_benchmark, test_save_obs, api_test, parallel_api_test, seed_test, max_cycles_test, render_test, bombardment_test, state_test
from pettingzoo.butterfly import pistonball_v6, prospector_v4, knights_archers_zombies_v9
from pettingzoo.magent import battlefield_v4, gather_v4, battle_v3
from pettingzoo.utils import random_demo
import evacuation_v1
import rps
import rps_2
import ants
# env = pistonball_v6.env()
# env = battlefield_v4.env(map_size=80, minimap_mode=False, step_reward=-0.005,
    # dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
    # max_cycles=1000, extra_features=False)
# env = gather_v4.env()
# env = evacuation_v0.env()
# env = rps_2.env()
env = evacuation_v1.env()
env_par = evacuation_v1.parallel_env()
# env = prospector_v4.env()
# env_par = rps.parallel_env()
# env = battle_v3.env()
performance_benchmark(env)
# test_save_obs(env)
# random_demo(env, render=True, episodes=1)
# api_test(env, num_cycles=1000, verbose_progress=True)
# parallel_api_test(env_par, num_cycles=1000)
# seed_test(evacuation_v1.env, num_cycles=10)
# parallel_seed_test(env_par, num_cycles=10)
# render_test(evacuation_v1.env)
# max_cycles_test(env)
# bombardment_test(env)
# state_test(env, env_par)
