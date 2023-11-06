# pylint: disable=protected-access,bad-builtin
import unittest
from timeit import timeit
from collections import defaultdict
import numpy as np

import nmmo
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState
from nmmo.lib.event_log import EventState
from nmmo.core.observation import Observation
from nmmo.core import action as Action
from nmmo.lib import utils
from tests.testhelpers import ScriptedAgentTestConfig

TileAttr = TileState.State.attr_name_to_col
EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
EventAttr = EventState.State.attr_name_to_col


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

  def test_action_target_consts(self):
    self.assertEqual(len(Action.Style.edges), 3)
    self.assertEqual(len(Action.Price.edges), self.config.PRICE_N_OBS)
    self.assertEqual(len(Action.Token.edges), self.config.COMMUNICATION_NUM_TOKENS)

  def test_obs_tile_correctness(self):
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
      # check if the tile obs size
      self.assertEqual(len(agent_obs.tiles), self.config.MAP_N_OBS)

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

  def test_env_visible_tiles_correctness(self):
    def correct_visible_tile(realm, agent_id):
      # Based on numpy datatable window query
      assert agent_id in realm.players, "agent_id not in the realm"
      agent = realm.players[agent_id]
      radius = realm.config.PLAYER_VISION_RADIUS
      return TileState.Query.window(
        realm.datastore, agent.row.val, agent.col.val, radius)

    # implemented in the env._compute_observations()
    def visible_tiles_by_index(realm, agent_id, tile_map):
      assert agent_id in realm.players, "agent_id not in the realm"
      agent = realm.players[agent_id]
      radius = realm.config.PLAYER_VISION_RADIUS
      return tile_map[agent.row.val-radius:agent.row.val+radius+1,
                      agent.col.val-radius:agent.col.val+radius+1,:].reshape(225,3)

    # get tile map, to bypass the expensive tile window query
    tile_map = TileState.Query.get_map(self.env.realm.datastore, self.config.MAP_SIZE)

    obs = self.env._compute_observations()
    for agent_id in self.env.realm.players:
      self.assertTrue(np.array_equal(correct_visible_tile(self.env.realm, agent_id),
                                     obs[agent_id].tiles))

    print('---test_visible_tile_window---')
    print('reference:', timeit(lambda: correct_visible_tile(self.env.realm, agent_id),
                              number=1000, globals=globals()))
    print('implemented:',
          timeit(lambda: visible_tiles_by_index(self.env.realm, agent_id, tile_map),
                 number=1000, globals=globals()))

  def test_make_attack_mask_within_range(self):
    def correct_within_range(entities, attack_range, agent_row, agent_col):
      entities_pos = entities[:,[EntityAttr["row"],EntityAttr["col"]]]
      within_range = utils.linf(entities_pos,(agent_row, agent_col)) <= attack_range
      return within_range

    # implemented in the Observation._make_attack_mask()
    def simple_within_range(entities, attack_range, agent_row, agent_col):
      return np.maximum(
          np.abs(entities[:,EntityAttr["row"]] - agent_row),
          np.abs(entities[:,EntityAttr["col"]] - agent_col)
        ) <= attack_range

    obs = self.env._compute_observations()
    attack_range = self.config.COMBAT_MELEE_REACH

    for agent_obs in obs.values():
      entities = agent_obs.entities.values
      agent = agent_obs.agent()
      self.assertTrue(np.array_equal(
        correct_within_range(entities, attack_range, agent.row, agent.col),
        simple_within_range(entities, attack_range, agent.row, agent.col)))

    print('---test_attack_within_range---')
    print('reference:', timeit(
      lambda: correct_within_range(entities, attack_range, agent.row, agent.col),
      number=1000, globals=globals()))
    print('implemented:', timeit(
      lambda: simple_within_range(entities, attack_range, agent.row, agent.col),
      number=1000, globals=globals()))

  def test_gs_where_in_1d(self):
    config = ScriptedAgentTestConfig()
    env = nmmo.Env(config)
    env.reset(seed=0)
    for _ in range(5):
      env.step({})

    def correct_where_in_1d(event_data, subject):
      flt_idx = np.in1d(event_data[:, EventAttr['ent_id']], subject)
      return event_data[flt_idx]

    def where_in_1d_with_index(event_data, subject, index):
      flt_idx = [row for sbj in subject for row in index.get(sbj,[])]
      return event_data[flt_idx]

    event_data = EventState.Query.table(env.realm.datastore)
    event_index = defaultdict()
    for row, id_ in enumerate(event_data[:,EventAttr['ent_id']]):
      if id_ in event_index:
        event_index[id_].append(row)
      else:
        event_index[id_] = [row]

    # NOTE: the index-based approach returns the data in different order,
    #   and all the operations in the task system don't use the order info
    def sort_event_data(event_data):
      keys = [event_data[:,i] for i in range(1,8)]
      sorted_idx = np.lexsort(keys)
      return event_data[sorted_idx]
    arr1 = sort_event_data(correct_where_in_1d(event_data, [1,2,3]))
    arr2 = sort_event_data(where_in_1d_with_index(event_data, [1,2,3], event_index))
    self.assertTrue(np.array_equal(arr1, arr2))

    print('---test_gs_where_in_1d---')
    print('reference:', timeit(
      lambda: correct_where_in_1d(event_data, [1, 2, 3]),
      number=1000, globals=globals()))
    print('implemented:', timeit(
      lambda: where_in_1d_with_index(event_data, [1, 2, 3], event_index),
      number=1000, globals=globals()))

  def test_habitable(self):
    from nmmo.systems.npc_policy import get_habitable_dir as habitable_impl
    realm_map = self.env.realm.map
    realm_tiles= self.env.realm.map.tiles
    ent = self.env.realm.npcs[-1]
    np_random = self.env._np_random

    def habitable_ref(tiles, ent, np_random):
      r, c  = ent.pos
      cands = []
      if tiles[r-1, c].habitable:
        cands.append(Action.North)
      if tiles[r+1, c].habitable:
        cands.append(Action.South)
      if tiles[r, c-1].habitable:
        cands.append(Action.West)
      if tiles[r, c+1].habitable:
        cands.append(Action.East)

      if len(cands) == 0:
        return Action.North

      return np_random.choice(cands)

    print('---test_habitable---')
    print('reference:', timeit(
      lambda: habitable_ref(realm_tiles, ent, np_random),
      number=1000, globals=globals()))
    print('habitable_impl:', timeit(
      lambda: habitable_impl(realm_map, ent, np_random),
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
