# pylint: disable=protected-access,bad-builtin
import unittest
from timeit import timeit
from copy import deepcopy
#import random
import numpy as np

import nmmo
from tests.testhelpers import ScriptedAgentTestConfig

RANDOM_SEED = 3333  # random.randint(0, 10000)
PERF_TEST = True

class TestCythonMasks(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = ScriptedAgentTestConfig()
    cls.config.set("USE_CYTHON", True)
    cls.config.set("COMBAT_SPAWN_IMMUNITY", 5)
    cls.env = nmmo.Env(cls.config, RANDOM_SEED)
    cls.env.reset()
    for _ in range(7):
      cls.env.step({})

    cls.move_mask = cls.env._dummy_obs["ActionTargets"]["Move"]
    cls.attack_mask = cls.env._dummy_obs["ActionTargets"]["Attack"]

  def test_move_mask(self):
    obs = self.env.obs
    for agent_id in self.env.realm.players:
      np_masks = deepcopy(self.move_mask)
      cy_masks = deepcopy(self.move_mask)
      obs[agent_id]._make_move_mask(np_masks, use_cython=False)
      obs[agent_id]._make_move_mask(cy_masks, use_cython=True)
      self.assertTrue(np.array_equal(np_masks["Direction"], cy_masks["Direction"]))
    if PERF_TEST:
      print('---test_move_mask---')
      print('numpy:', timeit(
        lambda: [obs[agent_id]._make_move_mask(np_masks, use_cython=False)
                 for agent_id in self.env.realm.players], number=1000, globals=globals()))
      print('cython:', timeit(
        lambda: [obs[agent_id]._make_move_mask(cy_masks, use_cython=True)
                 for agent_id in self.env.realm.players], number=1000, globals=globals()))

  def test_attack_mask(self):
    obs = self.env.obs
    for agent_id in self.env.realm.players:
      np_masks = deepcopy(self.attack_mask)
      cy_masks = deepcopy(self.attack_mask)
      obs[agent_id]._make_attack_mask(np_masks, use_cython=False)
      obs[agent_id]._make_attack_mask(cy_masks, use_cython=True)
      self.assertTrue(np.array_equal(np_masks["Target"], cy_masks["Target"]))
    if PERF_TEST:
      print('---test_attack_mask---')
      print('numpy:', timeit(
        lambda: [obs[agent_id]._make_attack_mask(np_masks, use_cython=False)
                 for agent_id in self.env.realm.players], number=1000, globals=globals()))
      print('cython:', timeit(
        lambda: [obs[agent_id]._make_attack_mask(cy_masks, use_cython=True)
                 for agent_id in self.env.realm.players], number=1000, globals=globals()))


if __name__ == '__main__':
  unittest.main()
