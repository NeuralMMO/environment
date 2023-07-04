# import time
import cProfile
import io
import pstats

import nmmo
from nmmo.core.config import (NPC, AllGameSystems, Combat, Communication,
                              Equipment, Exchange, Item, Medium, Profession,
                              Progression, Resource, Small, Terrain)
from nmmo.task.task_api import nmmo_default_task, make_same_task
from nmmo.task.base_predicates import CountEvent, FullyArmed
from nmmo.systems.skill import Melee
from tests.testhelpers import profile_env_step
from scripted import baselines


# Test utils
def create_and_reset(conf):
  env = nmmo.Env(conf())
  env.reset(map_id=1)

def create_config(base, *systems):
  systems   = (base, *systems)
  name      = '_'.join(cls.__name__ for cls in systems)

  conf                    = type(name, systems, {})()

  conf.TERRAIN_TRAIN_MAPS = 1
  conf.TERRAIN_EVAL_MAPS  = 1
  conf.IMMORTAL = True

  return conf

def benchmark_config(benchmark, base, nent, *systems):
  conf = create_config(base, *systems)
  conf.PLAYER_N = nent
  conf.PLAYERS = [baselines.Random]

  env = nmmo.Env(conf)
  env.reset()

  benchmark(env.step, actions={})

# Small map tests -- fast with greater coverage for individual game systems
def test_small_env_creation(benchmark):
  benchmark(lambda: nmmo.Env(Small()))

def test_small_env_reset(benchmark):
  config = Small()
  config.PLAYERS = [baselines.Random]
  env = nmmo.Env(config)
  benchmark(lambda: env.reset(map_id=1))

def test_fps_base_small_1_pop(benchmark):
  benchmark_config(benchmark, Small, 1)

def test_fps_minimal_small_1_pop(benchmark):
  benchmark_config(benchmark, Small, 1, Terrain, Resource, Combat, Progression)

def test_fps_npc_small_1_pop(benchmark):
  benchmark_config(benchmark, Small, 1, Terrain, Resource, Combat, Progression, NPC)

def test_fps_test_small_1_pop(benchmark):
  benchmark_config(benchmark, Small, 1, Terrain, Resource, Combat, Progression, Item, Exchange)

def test_fps_no_npc_small_1_pop(benchmark):
  benchmark_config(benchmark, Small, 1, Terrain, Resource,
    Combat, Progression, Item, Equipment, Profession, Exchange, Communication)

def test_fps_all_small_1_pop(benchmark):
  benchmark_config(benchmark, Small, 1, AllGameSystems)

def test_fps_base_med_1_pop(benchmark):
  benchmark_config(benchmark, Medium, 1)

def test_fps_minimal_med_1_pop(benchmark):
  benchmark_config(benchmark, Medium, 1, Terrain, Resource, Combat)

def test_fps_npc_med_1_pop(benchmark):
  benchmark_config(benchmark, Medium, 1, Terrain, Resource, Combat, NPC)

def test_fps_test_med_1_pop(benchmark):
  benchmark_config(benchmark, Medium, 1, Terrain, Resource, Combat, Progression, Item, Exchange)

def test_fps_no_npc_med_1_pop(benchmark):
  benchmark_config(benchmark, Medium, 1, Terrain, Resource,
    Combat, Progression, Item, Equipment, Profession, Exchange, Communication)

def test_fps_all_med_1_pop(benchmark):
  benchmark_config(benchmark, Medium, 1, AllGameSystems)

def test_fps_base_med_100_pop(benchmark):
  benchmark_config(benchmark, Medium, 100)

def test_fps_minimal_med_100_pop(benchmark):
  benchmark_config(benchmark, Medium, 100, Terrain, Resource, Combat)

def test_fps_npc_med_100_pop(benchmark):
  benchmark_config(benchmark, Medium, 100, Terrain, Resource, Combat, NPC)

def test_fps_test_med_100_pop(benchmark):
  benchmark_config(benchmark, Medium, 100, Terrain, Resource, Combat, Progression, Item, Exchange)

def test_fps_no_npc_med_100_pop(benchmark):
  benchmark_config(benchmark, Medium, 100, Terrain, Resource, Combat,
    Progression, Item, Equipment, Profession, Exchange, Communication)

def test_fps_all_med_100_pop(benchmark):
  benchmark_config(benchmark, Medium, 100, AllGameSystems)

def set_seed_test():
  random_seed = 5000
  conf = create_config(Medium, Terrain, Resource, Combat, NPC)
  conf.PLAYER_N = 10
  conf.PLAYERS = [baselines.Random]

  env = nmmo.Env(conf)

  env.reset(seed=random_seed)
  for _ in range(1024):
    env.step({})

def set_seed_test_complex():
  tasks = nmmo_default_task(range(128))
  tasks += make_same_task(CountEvent, range(128),
                          pred_kwargs={'event': 'EAT_FOOD', 'N': 10})
  tasks += make_same_task(FullyArmed, range(128),
                          pred_kwargs={'combat_style': Melee, 'level': 3, 'num_agent': 1})
  profile_env_step(tasks=tasks)

if __name__ == '__main__':
  with open('profile.run','a', encoding="utf-8") as f:
    pr = cProfile.Profile()
    pr.enable()
    set_seed_test_complex()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr,stream=s).sort_stats('tottime')
    ps.print_stats()
    f.write(s.getvalue())

'''
def benchmark_env(benchmark, env, nent):
  env.config.PLAYER_N = nent
  env.config.PLAYERS = [nmmo.agent.Random]
  env.reset()

  benchmark(env.step, actions={})
# Reuse large maps since we aren't benchmarking the reset function
def test_large_env_creation(benchmark):
  benchmark(lambda: nmmo.Env(Large()))

def test_large_env_reset(benchmark):
  env = nmmo.Env(Large())
  benchmark(lambda: env.reset(idx=1))

LargeMapsRCP = nmmo.Env(create_config(Large, Resource, Terrain, Combat, Progression))
LargeMapsAll = nmmo.Env(create_config(Large, AllGameSystems))

def test_fps_large_rcp_1_pop(benchmark):
  benchmark_env(benchmark, LargeMapsRCP, 1)

def test_fps_large_rcp_100_pop(benchmark):
  benchmark_env(benchmark, LargeMapsRCP, 100)

def test_fps_large_rcp_1000_pop(benchmark):
  benchmark_env(benchmark, LargeMapsRCP, 1000)

def test_fps_large_all_1_pop(benchmark):
  benchmark_env(benchmark, LargeMapsAll, 1)

def test_fps_large_all_100_pop(benchmark):
  benchmark_env(benchmark, LargeMapsAll, 100)

def test_fps_large_all_1000_pop(benchmark):
  benchmark_env(benchmark, LargeMapsAll, 1000)
'''
