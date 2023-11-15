# pylint: disable=protected-access
import unittest
import numpy as np

import nmmo
import nmmo.core.map
from nmmo.core.tile import Tile, TileState
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.lib import material

class MockRealm:
  def __init__(self):
    self.datastore = NumpyDatastore()
    self.datastore.register_object_type("Tile", TileState.State.num_attributes)
    self.config = nmmo.config.Small()
    self._np_random = np.random
    self.tick = 0
    self.event_log = None

class MockTask:
  def __init__(self, ent_id):
    self.assignee = (ent_id,)

class MockEntity:
  def __init__(self, ent_id):
    self.ent_id = ent_id
    self.my_tasks = None
    if ent_id > 0:  # only for players
      self.my_tasks = [MockTask(ent_id)]

class TestTileSeize(unittest.TestCase):
  # pylint: disable=no-member
  def test_tile(self):
    mock_realm = MockRealm()
    np_random = np.random
    tile = Tile(mock_realm, 10, 20, np_random)

    tile.reset(material.Foilage, nmmo.config.Small(), np_random)

    self.assertEqual(tile.row.val, 10)
    self.assertEqual(tile.col.val, 20)
    self.assertEqual(tile.material_id.val, material.Foilage.index)
    self.assertEqual(tile.seize_history, [])

    mock_realm.tick = 1
    tile.add_entity(MockEntity(1))
    self.assertEqual(tile.occupied, True)
    tile.update_seize()
    self.assertEqual(tile.seize_history[-1], (1, 1))

    # Agent 1 stayed, so no change
    mock_realm.tick = 2
    tile.update_seize()
    self.assertEqual(tile.seize_history[-1], (1, 1))

    # Two agents occupy the tile, so no change
    mock_realm.tick = 3
    tile.add_entity(MockEntity(2))
    self.assertCountEqual(tile.entities.keys(), [1, 2])
    self.assertEqual(tile.occupied, True)
    tile.update_seize()
    self.assertEqual(tile.seize_history[-1], (1, 1))

    mock_realm.tick = 5
    tile.remove_entity(1)
    self.assertCountEqual(tile.entities.keys(), [2])
    self.assertEqual(tile.occupied, True)
    tile.update_seize()
    self.assertEqual(tile.seize_history[-1], (2, 5))  # new seize history

    # Two agents occupy the tile, so no change
    mock_realm.tick = 7
    tile.add_entity(MockEntity(-10))
    self.assertListEqual(list(tile.entities.keys()), [2, -10])
    self.assertEqual(tile.occupied, True)
    tile.update_seize()
    self.assertEqual(tile.seize_history[-1], (2, 5))

    # Should not change when occupied by an npc
    mock_realm.tick = 9
    tile.remove_entity(2)
    self.assertListEqual(list(tile.entities.keys()), [-10])
    #self.assertEqual(tile.occupied, False)
    tile.update_seize()
    self.assertEqual(tile.seize_history[-1], (2, 5))

    tile.harvest(True)
    self.assertEqual(tile.depleted, True)
    self.assertEqual(tile.material_id.val, material.Scrub.index)

  def test_map_seize_targets(self):
    mock_realm = MockRealm()
    config = mock_realm.config
    np_random = mock_realm._np_random
    map_dict = {"map": np.ones((config.MAP_SIZE, config.MAP_SIZE))*2}  # all grass tiles
    center_tile = (config.MAP_SIZE//2, config.MAP_SIZE//2)

    test_map = nmmo.core.map.Map(config, mock_realm, np_random)
    test_map.reset(map_dict, np_random, seize_targets=["center"])
    self.assertListEqual(test_map.seize_targets, [center_tile])
    self.assertDictEqual(test_map.seize_status, {})

    mock_realm.tick = 4
    test_map.tiles[center_tile].add_entity(MockEntity(5))
    test_map.step()
    self.assertDictEqual(test_map.seize_status, {center_tile: (5, 4)})  # ent_id, tick

    mock_realm.tick = 6
    test_map.tiles[center_tile].remove_entity(5)
    test_map.step()
    self.assertDictEqual(test_map.seize_status, {center_tile: (5, 4)})  # should not change

    mock_realm.tick = 9
    test_map.tiles[center_tile].add_entity(MockEntity(6))
    test_map.tiles[center_tile].add_entity(MockEntity(-7))
    test_map.step()
    self.assertDictEqual(test_map.seize_status, {center_tile: (5, 4)})  # should not change

    mock_realm.tick = 11
    test_map.tiles[center_tile].remove_entity(6)  # so that -7 is the only entity
    test_map.step()
    self.assertDictEqual(test_map.seize_status, {center_tile: (5, 4)})  # should not change

    mock_realm.tick = 14
    test_map.tiles[center_tile].remove_entity(-7)
    test_map.tiles[center_tile].add_entity(MockEntity(10))
    test_map.step()
    self.assertDictEqual(test_map.seize_status, {center_tile: (10, 14)})

if __name__ == '__main__':
  unittest.main()
