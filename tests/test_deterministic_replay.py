#from pdb import set_trace as T
import unittest

import os
import glob
import pickle
import logging
import random
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

# pylint: disable=import-error
from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv
from tests.testhelpers import observations_are_equal

import nmmo

TEST_HORIZON = 50
LOCAL_REPLAY = 'tests/replay_local.pickle'

def load_replay_file(replay_file):
  # load the pickle file
  with open(replay_file, 'rb') as handle:
    ref_data = pickle.load(handle)

    logging.info('TestDetReplay: Loading the existing replay file with seed %s',
                 str(ref_data['seed']))

    seed = ref_data['seed']
    config = ref_data['config']
    map_src = ref_data['map']
    init_obs = ref_data['init_obs']
    init_npcs = ref_data['init_npcs']
    med_obs = ref_data['med_obs']
    actions = ref_data['actions']
    final_obs = ref_data['final_obs']
    final_npcs = ref_data['final_npcs']

  return seed, config, map_src, init_obs, init_npcs, med_obs, actions, final_obs, final_npcs


def make_actions_picklable(actions: Dict[int, Dict[str, Dict[str, Any]]]):
  for eid in actions:
    for atn, args in actions[eid].items():
      for arg, val in args.items():
        if arg == nmmo.io.action.Price and not isinstance(val, int):
          # <class 'nmmo.io.action.Price'>: <class 'nmmo.io.action.Discrete_1'>
          # convert Discrete_1 to 1
          actions[eid][atn][arg] = val.val
  return actions


def generate_replay_file(replay_file, test_horizon):
  # generate the new data with a new env
  seed = random.randint(0, 10000)
  logging.info('TestDetReplay: Creating a new replay file with seed %s', str(seed))
  config = ScriptedAgentTestConfig()
  env_src = ScriptedAgentTestEnv(config, seed=seed)
  init_obs = env_src.reset()
  init_npcs = env_src.realm.npcs.packet

  # extract the map
  map_src = np.zeros((config.MAP_SIZE, config.MAP_SIZE))
  for r in range(config.MAP_SIZE):
    for c in range(config.MAP_SIZE):
      map_src[r,c] = env_src.realm.map.tiles[r,c].material_id.val

  med_obs, actions = [], []
  logging.info('TestDetReplay: Running %s ticks', str(test_horizon))
  for _ in tqdm(range(test_horizon)):
    nxt_obs, _, _, _ = env_src.step({})
    med_obs.append(nxt_obs)
    actions.append(make_actions_picklable(env_src.actions))
  final_obs = nxt_obs
  final_npcs = env_src.realm.npcs.packet

  # save to the file
  with open(replay_file, 'wb') as handle:
    ref_data = {}
    ref_data['version'] = nmmo.__version__ # just in case
    ref_data['seed'] = seed
    ref_data['config'] = config
    ref_data['map'] = map_src
    ref_data['init_obs'] = init_obs
    ref_data['init_npcs'] = init_npcs
    ref_data['med_obs'] = med_obs
    ref_data['actions'] = actions
    ref_data['final_obs'] = final_obs
    ref_data['final_npcs'] = final_npcs

    pickle.dump(ref_data, handle)

  return seed, config, map_src, init_obs, init_npcs, med_obs, actions, final_obs, final_npcs


class TestDeterministicReplay(unittest.TestCase):

  # CHECK ME: pausing the deterministic replay test while debugging actions/items
  #   because changes there would most likely to change the game play and make the test fail
  __test__ = False

  @classmethod
  def setUpClass(cls):
    """
    First, check if there is a replay file on the repo that starts with 'replay_repo_'
      If there is one, use it.

    Second, check if there a local replay file, which should be named 'replay_local.pickle'
      If there is one, use it. If not create one.

    TODO: allow passing a different replay file
    """
    # first, look for the repo replay file
    replay_files = glob.glob(os.path.join('tests', 'replay_repo_*.pickle'))
    if replay_files:
      # there may be several, but we only take the first one [0]
      cls.seed, cls.config, cls.map_src, cls.init_obs_src, cls.init_npcs_src, \
      cls.med_obs_src,cls.actions, cls.final_obs_src, cls.final_npcs_src = \
        load_replay_file(replay_files[0])
    else:
      # if there is no repo replay file, then go with the default local file
      if os.path.exists(LOCAL_REPLAY):
        cls.seed, cls.config, cls.map_src, cls.init_obs_src, cls.init_npcs_src, \
        cls.med_obs_src, cls.actions, cls.final_obs_src, cls.final_npcs_src = \
          load_replay_file(LOCAL_REPLAY)
      else:
        cls.seed, cls.config, cls.map_src, cls.init_obs_src, cls.init_npcs_src, \
        cls.med_obs_src, cls.actions, cls.final_obs_src, cls.final_npcs_src = \
          generate_replay_file(LOCAL_REPLAY, TEST_HORIZON)
    cls.horizon = len(cls.actions)

    logging.info('TestDetReplay: Setting up the replication env with seed %s', str(cls.seed))
    env_rep = ScriptedAgentTestEnv(cls.config, seed=cls.seed)
    cls.init_obs_rep = env_rep.reset()
    cls.init_npcs_rep = env_rep.realm.npcs.packet

    # extract the map
    cls.map_rep = np.zeros((cls.config.MAP_SIZE, cls.config.MAP_SIZE))
    for r in range(cls.config.MAP_SIZE):
      for c in range(cls.config.MAP_SIZE):
        cls.map_rep[r,c] = env_rep.realm.map.tiles[r,c].material_id.val

    cls.med_obs_rep, cls.actions_rep = [], []
    logging.info('TestDetReplay: Running %s ticks', str(cls.horizon))
    for t in tqdm(range(cls.horizon)):
      nxt_obs_rep, _, _, _ = env_rep.step(cls.actions[t])
      cls.med_obs_rep.append(nxt_obs_rep)
    cls.final_obs_rep = nxt_obs_rep
    cls.final_npcs_rep = env_rep.realm.npcs.packet

  def test_compare_maps(self):
    self.assertEqual(np.sum(self.map_src != self.map_rep), 0)

  def test_compare_init_obs(self):
    self.assertTrue(observations_are_equal(self.init_obs_src, self.init_obs_rep))

  def test_compare_init_npcs(self):
    self.assertTrue(observations_are_equal(self.init_npcs_src, self.init_npcs_rep))

  def test_compare_final_obs(self):
    self.assertTrue(observations_are_equal(self.final_obs_src, self.final_obs_rep))

  def test_compare_final_npcs(self):
    self.assertTrue(observations_are_equal(self.final_npcs_src, self.final_npcs_rep))


if __name__ == '__main__':
  unittest.main()
