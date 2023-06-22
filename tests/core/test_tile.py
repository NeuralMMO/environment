import unittest
import numpy as np

import nmmo
from nmmo.core.tile import Tile, TileState
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.lib import material

class MockRealm:
  def __init__(self):
    self.datastore = NumpyDatastore()
    self.datastore.register_object_type("Tile", TileState.State.num_attributes)
    self.config = nmmo.config.Small()
    self._np_random = np.random

class MockEntity():
  def __init__(self, ent_id):
    self.ent_id = ent_id

class TestTile(unittest.TestCase):
  # pylint: disable=no-member
  def test_tile(self):
    mock_realm = MockRealm()
    np_random = np.random
    tile = Tile(mock_realm, 10, 20, np_random)

    tile.reset(material.Foilage, nmmo.config.Small(), np_random)

    self.assertEqual(tile.row.val, 10)
    self.assertEqual(tile.col.val, 20)
    self.assertEqual(tile.material_id.val, material.Foilage.index)

    tile.add_entity(MockEntity(1))
    tile.add_entity(MockEntity(2))
    self.assertCountEqual(tile.entities.keys(), [1, 2])
    tile.remove_entity(1)
    self.assertCountEqual(tile.entities.keys(), [2])

    tile.harvest(True)
    self.assertEqual(tile.depleted, True)
    self.assertEqual(tile.material_id.val, material.Scrub.index)

if __name__ == '__main__':
  unittest.main()
