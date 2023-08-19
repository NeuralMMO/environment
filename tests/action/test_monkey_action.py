import unittest
import random
from tqdm import tqdm

import numpy as np

from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv

import nmmo

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = random.randint(0, 1000000)


def make_random_actions(config, ent_obs):
  assert 'ActionTargets' in ent_obs, 'ActionTargets is not provided in the obs'
  actions = {}

  # atn, arg, val
  for atn in sorted(nmmo.Action.edges(config)):
    actions[atn] = {}
    for arg in sorted(atn.edges, reverse=True): # intentionally doing wrong
      mask = ent_obs["ActionTargets"][atn.__name__][arg.__name__]
      actions[atn][arg] = 0
      if np.any(mask):
        actions[atn][arg] += int(np.random.choice(np.where(mask)[0]))

  return actions

# CHECK ME: this would be nice to include in the env._validate_actions()
def filter_item_actions(actions, use_str_key=False):
  # when there are multiple actions on the same item, select one
  flt_atns = {}
  inventory_atn = {} # key: inventory idx, val: action
  for atn in actions:
    if atn in [nmmo.action.Use, nmmo.action.Sell, nmmo.action.Give, nmmo.action.Destroy]:
      for arg, val in actions[atn].items():
        if arg == nmmo.action.InventoryItem:
          if val not in inventory_atn:
            inventory_atn[val] = [( atn, actions[atn] )]
          else:
            inventory_atn[val].append(( atn, actions[atn] ))
    else:
      flt_atns[atn] = actions[atn]

    # randomly select one action for each inventory item
    for atns in inventory_atn.values():
      if len(atns) > 1:
        picked = random.choice(atns)
        flt_atns[picked[0]] = picked[1]
      else:
        flt_atns[atns[0][0]] = atns[0][1]

    # convert action keys to str
    if use_str_key:
      str_atns = {}
      for atn, args in flt_atns.items():
        str_atns[atn.__name__] = {}
        for arg, val in args.items():
          str_atns[atn.__name__][arg.__name__] = val
      flt_atns = str_atns

    return flt_atns


class TestMonkeyAction(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = ScriptedAgentTestConfig()
    cls.config.PROVIDE_ACTION_TARGETS = True

  @staticmethod
  # NOTE: this can also be used for sweeping random seeds
  def rollout_with_seed(config, seed, use_str_key=False):
    env = ScriptedAgentTestEnv(config)
    obs = env.reset(seed=seed)

    for _ in tqdm(range(TEST_HORIZON)):
      # sample random actions for each player
      actions = {}
      for ent_id in env.realm.players:
        ent_atns = make_random_actions(config, obs[ent_id])
        actions[ent_id] = filter_item_actions(ent_atns, use_str_key)
      obs, _, _, _ = env.step(actions)

  def test_monkey_action(self):
    try:
      self.rollout_with_seed(self.config, RANDOM_SEED)
    except: # pylint: disable=bare-except
      assert False, f"Monkey action failed. seed: {RANDOM_SEED}"

  def test_monkey_action_with_str_key(self):
    self.rollout_with_seed(self.config, RANDOM_SEED, use_str_key=True)

if __name__ == '__main__':
  unittest.main()
