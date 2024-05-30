from __future__ import annotations
from collections import deque
import math

from typing import Dict

from nmmo.systems.item import Item, Stack
from nmmo.lib.event_code import EventCode

"""
The Exchange class is a simulation of an in-game item exchange.
It has several methods that allow players to list items for sale,
buy items, and remove expired listings.

The _list_item() method is used to add a new item to the
exchange, and the unlist_item() method is used to remove
an item from the exchange. The step() method is used to
regularly check and remove expired listings.

The sell() method allows a player to sell an item, and the buy() method
allows a player to purchase an item. The packet property returns a
dictionary that contains information about the items currently being
sold on the exchange, such as the maximum and minimum price,
the average price, and the total supply of the items.

"""
class ItemListing:
  def __init__(self, item: Item, seller, price: int, tick: int):
    self.item = item
    self.seller = seller
    self.price = price
    self.tick = tick

class Exchange:
  def __init__(self, realm):
    self._listings_queue: deque[(int, int)] = deque() # (item_id, tick)
    self._item_listings: Dict[int, ItemListing] = {}
    self._realm = realm
    self._config = realm.config

  def reset(self):
    self._listings_queue.clear()
    self._item_listings.clear()

  def _list_item(self, item: Item, seller, price: int, tick: int):
    item.listed_price.update(price)
    self._item_listings[item.id.val] = ItemListing(item, seller, price, tick)
    self._listings_queue.append((item.id.val, tick))

  def unlist_item(self, item: Item):
    if item.id.val in self._item_listings:
      self._unlist_item(item.id.val)

  def _unlist_item(self, item_id: int):
    item = self._item_listings.pop(item_id).item
    item.listed_price.update(0)

  def step(self):
    """
    Remove expired listings from the exchange's listings queue
    and item listings dictionary.

    The method starts by checking the oldest listing in the listings
    queue using a while loop. If the current tick minus the
    listing tick is less than or equal to the EXCHANGE_LISTING_DURATION
    in the realm's configuration, the method breaks out of
    the loop as the oldest listing has not expired.
    If the oldest listing has expired, the method removes it from the
    listings queue and the item listings dictionary.

    It then checks if the actual listing still exists and that
    it is indeed expired. If it does exist and is expired,
    it calls the _unlist_item method to remove the listing and update
    the item's listed price. The process repeats until all expired listings
    are removed from the queue and dictionary.
    """
    if self._config.EXCHANGE_SYSTEM_ENABLED is False:
      return

    current_tick = self._realm.tick

    # Remove expired listings
    while self._listings_queue:
      (item_id, listing_tick) = self._listings_queue[0]
      if current_tick - listing_tick <= self._config.EXCHANGE_LISTING_DURATION:
        # Oldest listing has not expired
        break

      # Remove expired listing from queue
      self._listings_queue.popleft()

      # The actual listing might have been refreshed and is newer than the queue record.
      # Or it might have already been removed.
      listing = self._item_listings.get(item_id)
      if listing is not None and \
        current_tick - listing.tick > self._config.EXCHANGE_LISTING_DURATION:
        self._unlist_item(item_id)

  def sell(self, seller, item: Item, price: int, tick: int):
    assert isinstance(
        item, object), f'{item} for sale is not an Item instance'
    assert item in seller.inventory, f'{item} for sale is not in {seller} inventory'
    assert item.quantity.val > 0, f'{item} for sale has quantity {item.quantity.val}'
    assert item.listed_price.val == 0, 'Item is already listed'
    assert item.equipped.val == 0, 'Item has been equiped so cannot be listed'
    assert price > 0, 'Price must be larger than 0'
    self._list_item(item, seller, price, tick)
    self._realm.event_log.record(EventCode.LIST_ITEM, seller, item=item, price=price)

  def buy(self, buyer, item: Item):
    assert item.quantity.val > 0, f'{item} purchase has quantity {item.quantity.val}'
    assert item.equipped.val == 0, 'Listed item must not be equipped'
    assert buyer.gold.val >= item.listed_price.val, 'Buyer does not have enough gold'
    assert buyer.ent_id != item.owner_id.val, 'One cannot buy their own items'

    if not buyer.inventory.space:
      if isinstance(item, Stack):
        if not buyer.inventory.has_stack(item.signature):
          # no ammo stack with the same signature, so cannot buy
          return
      else: # no space, and item is not ammo stack, so cannot buy
        return

    # item is not in the listing (perhaps bought by other)
    if item.id.val not in self._item_listings:
      return

    listing = self._item_listings[item.id.val]
    price = item.listed_price.val

    self.unlist_item(item)
    listing.seller.inventory.remove(item)
    buyer.inventory.receive(item)
    buyer.gold.decrement(price)
    listing.seller.gold.increment(price)

    self._realm.event_log.record(EventCode.BUY_ITEM, buyer, item=item, price=price)
    self._realm.event_log.record(EventCode.EARN_GOLD, listing.seller, amount=price)

  @property
  def packet(self):
    packet = {}
    for listing in self._item_listings.values():
      item = listing.item
      key = f'{item.__class__.__name__}_{item.level.val}'
      max_price = max(packet.get(key, {}).get('max_price', -math.inf), listing.price)
      min_price = min(packet.get(key, {}).get('min_price', math.inf), listing.price)
      supply = packet.get(key, {}).get('supply', 0) + item.quantity.val

      packet[key] = {
        'max_price': max_price,
        'min_price': min_price,
        'price': (max_price + min_price) / 2,
        'supply': supply
      }

    return packet
