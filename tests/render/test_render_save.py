'''Manual test for render client connectivity and save replay'''
import nmmo
from nmmo.core.config import (AllGameSystems, Combat, Communication,
                              Equipment, Exchange, Item, Medium, Profession,
                              Progression, Resource, Small, Terrain)
from nmmo.render.render_client import WebsocketRenderer
from nmmo.render.replay_helper import FileReplayHelper
from scripted import baselines

def create_config(base, nent, *systems):
  systems = (base, *systems)
  name = '_'.join(cls.__name__ for cls in systems)
  conf = type(name, systems, {})()
  conf.TERRAIN_TRAIN_MAPS = 1
  conf.TERRAIN_EVAL_MAPS  = 1
  conf.IMMORTAL = True
  conf.PLAYER_N = nent
  conf.PLAYERS = [baselines.Random]
  return conf

no_npc_small_1_pop_conf = create_config(Small, 1, Terrain, Resource,
  Combat, Progression, Item, Equipment, Profession, Exchange, Communication)

no_npc_med_1_pop_conf = create_config(Medium, 1, Terrain, Resource,
  Combat, Progression, Item, Equipment, Profession, Exchange, Communication)

no_npc_med_100_pop_conf = create_config(Medium, 100, Terrain, Resource,
  Combat, Progression, Item, Equipment, Profession, Exchange, Communication)

all_small_1_pop_conf = create_config(Small, 1, AllGameSystems)

all_med_1_pop_conf = create_config(Medium, 1, AllGameSystems)

all_med_100_pop_conf = create_config(Medium, 100, AllGameSystems)

conf_dict = {
  'no_npc_small_1_pop': no_npc_small_1_pop_conf,
  'no_npc_med_1_pop': no_npc_med_1_pop_conf,
  'no_npc_med_100_pop': no_npc_med_100_pop_conf,
  'all_small_1_pop': all_small_1_pop_conf,
  'all_med_1_pop': all_med_1_pop_conf,
  'all_med_100_pop': all_med_100_pop_conf
}

if __name__ == '__main__':
  import random
  from tqdm import tqdm

  TEST_HORIZON = 100
  RANDOM_SEED = random.randint(0, 9999)

  replay_helper = FileReplayHelper()

  # the renderer is external to the env, so need to manually initiate it
  renderer = WebsocketRenderer()

  for conf_name, config in conf_dict.items():
    env = nmmo.Env(config)

    # to make replay, one should create replay_helper
    #   and run the below line
    env.realm.record_replay(replay_helper)

    env.reset(seed=RANDOM_SEED)
    renderer.set_realm(env.realm)

    for tick in tqdm(range(TEST_HORIZON)):
      env.step({})
      renderer.render_realm()

    # NOTE: save the data in uncompressed json format, since
    #   the web client has trouble loading the compressed replay file
    replay_helper.save(f'replay_{conf_name}_seed_{RANDOM_SEED:04d}.json')
