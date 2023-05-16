from types import SimpleNamespace
import unittest
import nmmo
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.systems.exchange import Exchange
from nmmo.systems.item import ItemState
import nmmo.systems.item as item
import numpy as np

class MockRealm:
  def __init__(self):
    self.config = nmmo.config.Default()
    self.config.EXCHANGE_LISTING_DURATION = 3
    self.datastore = NumpyDatastore()
    self.items = {}
    self.datastore.register_object_type("Item", ItemState.State.num_attributes)

class MockEntity:
  def __init__(self) -> None:
    self.items = []
    self.inventory = SimpleNamespace(
      receive = lambda item: self.items.append(item),
      remove = lambda item: self.items.remove(item)
     )
  
class TestExchange(unittest.TestCase):
  def test_listings(self):
    realm = MockRealm()
    exchange = Exchange(realm)

    entity_1 = MockEntity()

    hat_1 = item.Hat(realm, 1)
    hat_2 = item.Hat(realm, 10)
    entity_1.inventory.receive(hat_1)
    entity_1.inventory.receive(hat_2)
    self.assertEqual(len(entity_1.items), 2)

    tick = 0
    exchange._list_item(hat_1, entity_1, 10, tick)
    self.assertEqual(len(exchange._item_listings), 1)
    self.assertEqual(exchange._listings_queue[0], (hat_1.id.val, 0))

    tick = 1
    exchange._list_item(hat_2, entity_1, 20, tick)
    self.assertEqual(len(exchange._item_listings), 2)
    self.assertEqual(exchange._listings_queue[0], (hat_1.id.val, 0))

    tick = 4
    exchange.step(tick)
    # hat_1 should expire and not be listed
    self.assertEqual(len(exchange._item_listings), 1)
    self.assertEqual(exchange._listings_queue[0], (hat_2.id.val, 1))

    tick = 5
    exchange._list_item(hat_2, entity_1, 10, tick)
    exchange.step(tick)
    # hat_2 got re-listed, so should still be listed
    self.assertEqual(len(exchange._item_listings), 1)
    self.assertEqual(exchange._listings_queue[0], (hat_2.id.val, 5))

    tick = 10
    exchange.step(tick)
    self.assertEqual(len(exchange._item_listings), 0)

  def test_for_sale_items(self):
    realm = MockRealm()
    exchange = Exchange(realm)
    entity_1 = MockEntity()

    hat_1 = item.Hat(realm, 1)
    hat_2 = item.Hat(realm, 10)
    exchange._list_item(hat_1, entity_1, 10, 0)
    exchange._list_item(hat_2, entity_1, 20, 10)

    np.testing.assert_array_equal(
      item.Item.Query.for_sale(realm.datastore)[:,0], [hat_1.id.val, hat_2.id.val])

    # first listing should expire
    exchange.step(10)
    np.testing.assert_array_equal(
      item.Item.Query.for_sale(realm.datastore)[:,0], [hat_2.id.val])

    # second listing should expire
    exchange.step(100)
    np.testing.assert_array_equal(
      item.Item.Query.for_sale(realm.datastore)[:,0], [])

if __name__ == '__main__':
    unittest.main()