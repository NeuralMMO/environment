import unittest
import random
from tqdm import tqdm

import numpy as np

from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv

import nmmo

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = random.randint(0, 10000)


class TestMonkeyAction(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = ScriptedAgentTestConfig()
    cls.config.PROVIDE_ACTION_TARGETS = True

  def _make_random_actions(self, ent_obs):
    assert 'ActionTargets' in ent_obs, 'ActionTargets is not provided in the obs'
    actions = {}

    # atn, arg, val
    for atn in sorted(nmmo.Action.edges(self.config)):
      actions[atn] = {}
      for arg in sorted(atn.edges, reverse=True): # intentionally doing wrong
        mask = ent_obs['ActionTargets'][atn][arg]
        actions[atn][arg] = 0
        if np.any(mask):
          actions[atn][arg] += int(np.random.choice(np.where(mask)[0]))

    return actions

  def test_monkey_action(self):
    env = ScriptedAgentTestEnv(self.config)
    obs = env.reset(seed=RANDOM_SEED)

    # the goal is just to run TEST_HORIZON without runtime errors
    # TODO(kywch): add more sophisticate/correct action validation tests
    #   for example, one cannot USE/SELL/GIVE/DESTORY the same item
    #   this will not produce an runtime error, but agents should not do that
    for _ in tqdm(range(TEST_HORIZON)):
      # sample random actions for each player
      actions = {}
      for ent_id in env.realm.players:
        actions[ent_id] = self._make_random_actions(obs[ent_id])
      obs, _, _, _ = env.step(actions)

    # DONE


if __name__ == '__main__':
  unittest.main()
