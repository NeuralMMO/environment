from pdb import set_trace as T
import pytest

import nmmo
from nmmo.core.config import Config, Small, Medium, Large, Terrain, Resource, Combat, NPC, Progression, Item, Equipment, Profession, Exchange, Communication, AllGameSystems

# Test utils
def create_and_reset(conf):
    env = nmmo.Env(conf())
    env.reset(idx=1)

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

    env = nmmo.Env(conf)
    env.reset()

    benchmark(env.step, actions={})

def benchmark_env(benchmark, env, nent):
    env.config.PLAYER_N = nent
    env.reset()

    benchmark(env.step, actions={})

# Small map tests -- fast with greater coverage for individual game systems
def test_small_env_creation(benchmark):
    benchmark(lambda: nmmo.Env(Small()))

def test_small_env_reset(benchmark):
    env = nmmo.Env(Small())
    benchmark(lambda: env.reset(idx=1))

def test_fps_base_small_1_pop(benchmark):
    benchmark_config(benchmark, Small, 1) 

def test_fps_minimal_small_1_pop(benchmark):
    benchmark_config(benchmark, Small, 1, Terrain, Resource, Combat, Progression) 

def test_fps_npc_small_1_pop(benchmark):
    benchmark_config(benchmark, Small, 1, Terrain, Resource, Combat, Progression, NPC) 

def test_fps_test_small_1_pop(benchmark):
    benchmark_config(benchmark, Small, 1, Terrain, Resource, Combat, Progression, Item, Exchange) 

def test_fps_no_npc_small_1_pop(benchmark):
    benchmark_config(benchmark, Small, 1, Terrain, Resource, Combat, Progression, Item, Equipment, Profession, Exchange, Communication) 

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
    benchmark_config(benchmark, Medium, 1, Terrain, Resource, Combat, Progression, Item, Equipment, Profession, Exchange, Communication) 

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
    benchmark_config(benchmark, Medium, 100, Terrain, Resource, Combat, Progression, Item, Equipment, Profession, Exchange, Communication) 

def test_fps_all_med_100_pop(benchmark):
    benchmark_config(benchmark, Medium, 100, AllGameSystems) 


'''
# Reuse large maps since we aren't benchmarking the reset function
def test_large_env_creation(benchmark):
    benchmark(lambda: nmmo.Env(Large()))

def test_large_env_reset(benchmark):
    env = nmmo.Env(Large())
    benchmark(lambda: env.reset(idx=1))

LargeMapsRCP = nmmo.Env(create_config(Large, Resource, Combat, Progression))
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
