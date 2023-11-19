import unittest
import numpy as np

import nmmo
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.systems.item import Hat, Top, ItemState
from nmmo.systems.exchange import Exchange

class MockRealm:
  def __init__(self):
    self.config = nmmo.config.Default()
    self.datastore = NumpyDatastore()
    self.items = {}
    self.exchange = Exchange(self)
    self.datastore.register_object_type("Item", ItemState.State.num_attributes)
    self.players = {}

# pylint: disable=no-member
class TestItem(unittest.TestCase):
  def test_item(self):
    realm = MockRealm()

    hat_1 = Hat(realm, 1)
    self.assertTrue(ItemState.Query.by_id(realm.datastore, hat_1.id.val) is not None)
    self.assertEqual(hat_1.type_id.val, Hat.ITEM_TYPE_ID)
    self.assertEqual(hat_1.level.val, 1)
    self.assertEqual(hat_1.mage_defense.val, 10)

    hat_2 = Hat(realm, 10)
    self.assertTrue(ItemState.Query.by_id(realm.datastore, hat_2.id.val) is not None)
    self.assertEqual(hat_2.level.val, 10)
    self.assertEqual(hat_2.melee_defense.val, 100)

    self.assertDictEqual(realm.items, {hat_1.id.val: hat_1, hat_2.id.val: hat_2})

    # also test destroy
    ids = [hat_1.id.val, hat_2.id.val]
    hat_1.destroy()
    hat_2.destroy()
    # after destroy(), the datastore entry is gone, but the class still exsits
    # make sure that after destroy the owner_id is 0, at least
    self.assertTrue(hat_1.owner_id.val == 0)
    self.assertTrue(hat_2.owner_id.val == 0)
    for item_id in ids:
      self.assertTrue(len(ItemState.Query.by_id(realm.datastore, item_id)) == 0)
    self.assertDictEqual(realm.items, {})

    # create a new item with the hat_1's id, but it must still be void
    new_top = Top(realm, 3)
    new_top.id.update(ids[0]) # hat_1's id
    new_top.owner_id.update(100)
    # make sure that the hat_1 is not linked to the new_top
    self.assertTrue(hat_1.owner_id.val == 0)

  def test_owned_by(self):
    realm = MockRealm()

    hat_1 = Hat(realm, 1)
    hat_2 = Hat(realm, 10)

    hat_1.owner_id.update(1)
    hat_2.owner_id.update(1)

    np.testing.assert_array_equal(
      ItemState.Query.owned_by(realm.datastore, 1)[:,0],
      [hat_1.id.val, hat_2.id.val])

    self.assertEqual(Hat.Query.owned_by(realm.datastore, 2).size, 0)

if __name__ == '__main__':
  unittest.main()
