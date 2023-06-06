# pylint: disable=protected-access,bad-builtin
import unittest
from timeit import timeit
import numpy as np

import nmmo
from nmmo.core.tile import TileState
from nmmo.core.observation import Observation
from nmmo.core import action as Action
from nmmo.lib import material as Material

TileAttr = TileState.State.attr_name_to_col

class TestActionTargets(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = nmmo.config.Default()
    cls.env = nmmo.Env(cls.config)
    cls.env.reset()
    for _ in range(3):
      cls.env.step({})

  def test_tile_attr(self):
    self.assertDictEqual(TileAttr, {'row': 0, 'col': 1, 'material_id': 2})

  def test_move_mask_correctness(self):
    obs = self.env._compute_observations()

    center = self.config.PLAYER_VISION_RADIUS
    tile_dim = self.config.PLAYER_VISION_DIAMETER

    def correct_move_mask(agent_obs: Observation):
      # pylint: disable=not-an-iterable
      return np.array([agent_obs.tile(*d.delta).material_id in Material.Habitable
                       for d in Action.Direction.edges], dtype=np.int8)

    for agent_obs in obs.values():
      # check if the coord conversion is correct
      row_map = agent_obs.tiles[:,TileAttr['row']].reshape(tile_dim,tile_dim)
      col_map = agent_obs.tiles[:,TileAttr['col']].reshape(tile_dim,tile_dim)
      mat_map = agent_obs.tiles[:,TileAttr['material_id']].reshape(tile_dim,tile_dim)
      agent = agent_obs.agent()
      self.assertEqual(agent.row, row_map[center,center])
      self.assertEqual(agent.col, col_map[center,center])
      self.assertEqual(agent_obs.tile(0,0).material_id, mat_map[center,center])

      mask_ref = correct_move_mask(agent_obs)
      self.assertTrue(np.array_equal(agent_obs._make_move_mask(), mask_ref))

    # pylint: disable=unnecessary-lambda
    print('reference:', timeit(lambda: correct_move_mask(agent_obs),
                               number=1000, globals=globals()))
    print('implemented:', timeit(lambda: agent_obs._make_move_mask(),
                                 number=1000, globals=globals()))

if __name__ == '__main__':
  unittest.main()

  # config = nmmo.config.Default()
  # env = nmmo.Env(config)
  # env.reset()
  # for _ in range(10):
  #   env.step({})

  # obs = env._compute_observations()

  # test_func = [
  #   '_make_move_mask()', # 0.170 -> 0.022
  #   '_make_attack_mask()', # 0.060 -> 0.037
  #   '_make_use_mask()', # 0.0036 ->
  #   '_make_sell_mask()',
  #   '_make_give_target_mask()',
  #   '_make_destroy_item_mask()',
  #   '_make_buy_mask()', # 0.022 -> 0.011
  #   '_make_give_gold_mask()',
  #   '_existing_ammo_listings()',
  #   'agent()',
  #   'tile(1,-1)' # 0.020 (cache off) -> 0.012
  # ]

  # for func in test_func:
  #   print(func, timeit(f'obs[1].{func}', number=1000, globals=globals()))

  # # without ActionTargets: 0.97 (before) -> 0.26 (after)
  # # with ActionTargets: 3.23 (before) -> 2.45 (after)
  # print('obs._to_gym()', timeit(lambda: {a: o.to_gym() for a,o in obs.items()},
  #                               number=100, globals=globals()))
