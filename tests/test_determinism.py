#from pdb import set_trace as T
import unittest

import logging
import random
from tqdm import tqdm

from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv
from tests.testhelpers import observations_are_equal, actions_are_equal

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = random.randint(0, 10000)


class TestDeterminism(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.horizon = TEST_HORIZON
    cls.rand_seed = RANDOM_SEED
    cls.config = ScriptedAgentTestConfig()
    env = ScriptedAgentTestEnv(cls.config)

    logging.info('TestDeterminism: Setting up the reference env with seed %s', str(cls.rand_seed))
    cls.init_obs_src = env.reset(seed=cls.rand_seed)
    cls.actions_src = []
    logging.info('TestDeterminism: Running %s ticks', str(cls.horizon))
    for _ in tqdm(range(cls.horizon)):
      nxt_obs_src, _, _, _ = env.step({})
      cls.actions_src.append(env.actions)
    cls.final_obs_src = nxt_obs_src
    npcs_src = {}
    for nid, npc in list(env.realm.npcs.items()):
      npcs_src[nid] = npc.packet()
    cls.final_npcs_src = npcs_src

    logging.info('TestDeterminism: Setting up the replication env with seed %s', str(cls.rand_seed))
    cls.init_obs_rep = env.reset(seed=cls.rand_seed)
    cls.actions_rep = []
    logging.info('TestDeterminism: Running %s ticks', str(cls.horizon))
    for _ in tqdm(range(cls.horizon)):
      nxt_obs_rep, _, _, _ = env.step({})
      cls.actions_rep.append(env.actions)
    cls.final_obs_rep = nxt_obs_rep
    npcs_rep = {}
    for nid, npc in list(env.realm.npcs.items()):
      npcs_rep[nid] = npc.packet()
    cls.final_npcs_rep = npcs_rep

  def test_func_are_observations_equal(self):
    self.assertTrue(observations_are_equal(self.init_obs_src, self.init_obs_src))
    self.assertTrue(observations_are_equal(self.final_obs_src, self.final_obs_src))
    self.assertTrue(actions_are_equal(self.actions_src[0], self.actions_src[0]))
    self.assertDictEqual(self.final_npcs_src, self.final_npcs_src)

  def test_compare_initial_observations(self):
    # assertDictEqual CANNOT replace are_observations_equal
    self.assertTrue(observations_are_equal(self.init_obs_src, self.init_obs_rep))
    #self.assertDictEqual(self.init_obs_src, self.init_obs_rep)

  def test_compare_actions(self):
    self.assertEqual(len(self.actions_src), len(self.actions_rep))
    for t, action_src in enumerate(self.actions_src):
      self.assertTrue(actions_are_equal(action_src, self.actions_rep[t]))

  def test_compare_final_observations(self):
    # assertDictEqual CANNOT replace are_observations_equal
    self.assertTrue(observations_are_equal(self.final_obs_src, self.final_obs_rep))
    #self.assertDictEqual(self.final_obs_src, self.final_obs_rep)

  def test_compare_final_npcs(self)        :
    self.assertDictEqual(self.final_npcs_src, self.final_npcs_rep)


if __name__ == '__main__':
  unittest.main()
