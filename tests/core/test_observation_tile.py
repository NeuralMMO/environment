# pylint: disable=protected-access,bad-builtin
import unittest
from timeit import timeit
import numpy as np

import nmmo
from nmmo.core.tile import TileState
from nmmo.core.observation import Observation
from nmmo.core import action as Action

TileAttr = TileState.State.attr_name_to_col

class TestObservationTile(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = nmmo.config.Default()
    cls.env = nmmo.Env(cls.config)
    cls.env.reset(seed=1)
    for _ in range(3):
      cls.env.step({})

  def test_tile_attr(self):
    self.assertDictEqual(TileAttr, {'row': 0, 'col': 1, 'material_id': 2})

  def test_tile_correctness(self):
    obs = self.env._compute_observations()
    center = self.config.PLAYER_VISION_RADIUS
    tile_dim = self.config.PLAYER_VISION_DIAMETER

    # pylint: disable=inconsistent-return-statements
    def correct_tile(agent_obs: Observation, r_delta, c_delta):
      agent = agent_obs.agent()
      if (0 <= agent.row + r_delta < self.config.MAP_SIZE) & \
        (0 <= agent.col + c_delta < self.config.MAP_SIZE):
        r_cond = (agent_obs.tiles[:,TileState.State.attr_name_to_col["row"]] == agent.row+r_delta)
        c_cond = (agent_obs.tiles[:,TileState.State.attr_name_to_col["col"]] == agent.col+c_delta)
        return TileState.parse_array(agent_obs.tiles[r_cond & c_cond][0])

    for agent_obs in obs.values():
      # check if the coord conversion is correct
      row_map = agent_obs.tiles[:,TileAttr['row']].reshape(tile_dim,tile_dim)
      col_map = agent_obs.tiles[:,TileAttr['col']].reshape(tile_dim,tile_dim)
      mat_map = agent_obs.tiles[:,TileAttr['material_id']].reshape(tile_dim,tile_dim)
      agent = agent_obs.agent()
      self.assertEqual(agent.row, row_map[center,center])
      self.assertEqual(agent.col, col_map[center,center])
      self.assertEqual(agent_obs.tile(0,0).material_id, mat_map[center,center])

      # pylint: disable=not-an-iterable
      for d in Action.Direction.edges:
        self.assertTrue(np.array_equal(correct_tile(agent_obs, *d.delta),
                                      agent_obs.tile(*d.delta)))

    print('---test_correct_tile---')
    print('reference:', timeit(lambda: correct_tile(agent_obs, *d.delta),
                              number=1000, globals=globals()))
    print('implemented:', timeit(lambda: agent_obs.tile(*d.delta),
                                number=1000, globals=globals()))

if __name__ == '__main__':
  unittest.main()

  # from tests.testhelpers import profile_env_step
  # profile_env_step()

  # config = nmmo.config.Default()
  # env = nmmo.Env(config)
  # env.reset()
  # for _ in range(10):
  #   env.step({})

  # obs = env._compute_observations()

  # NOTE: the most of performance gain in _make_move_mask comes from the improved tile
  # test_func = [
  #   '_make_move_mask()', # 0.170 -> 0.012
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
