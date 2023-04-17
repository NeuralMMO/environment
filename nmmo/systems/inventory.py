from typing import Dict, Tuple

from ordered_set import OrderedSet

from nmmo.systems import item as Item
class EquipmentSlot:
  def __init__(self) -> None:
    self.item = None

  def equip(self, item: Item.Item) -> None:
    self.item = item

  def unequip(self) -> None:
    if self.item:
      self.item.equipped.update(0)
    self.item = None

class Equipment:
  def __init__(self):
    self.hat = EquipmentSlot()
    self.top = EquipmentSlot()
    self.bottom = EquipmentSlot()
    self.held = EquipmentSlot()
    self.ammunition = EquipmentSlot()

  def total(self, lambda_getter):
    items = [lambda_getter(e).val for e in self]
    if not items:
      return 0
    return sum(items)

  def __iter__(self):
    for slot in [self.hat, self.top, self.bottom, self.held, self.ammunition]:
      if slot.item is not None:
        yield slot.item

  def conditional_packet(self, packet, slot_name: str, slot: EquipmentSlot):
    if slot.item:
      packet[slot_name] = slot.item.packet

  @property
  def item_level(self):
    return self.total(lambda e: e.level)

  @property
  def melee_attack(self):
    return self.total(lambda e: e.melee_attack)

  @property
  def range_attack(self):
    return self.total(lambda e: e.range_attack)

  @property
  def mage_attack(self):
    return self.total(lambda e: e.mage_attack)

  @property
  def melee_defense(self):
    return self.total(lambda e: e.melee_defense)

  @property
  def range_defense(self):
    return self.total(lambda e: e.range_defense)

  @property
  def mage_defense(self):
    return self.total(lambda e: e.mage_defense)

  @property
  def packet(self):
    packet = {}

    self.conditional_packet(packet, 'hat',        self.hat)
    self.conditional_packet(packet, 'top',        self.top)
    self.conditional_packet(packet, 'bottom',     self.bottom)
    self.conditional_packet(packet, 'held',       self.held)
    self.conditional_packet(packet, 'ammunition', self.ammunition)

    # pylint: disable=R0801
    # Similar lines here and in npc.py
    packet['item_level']    = self.item_level
    packet['melee_attack']  = self.melee_attack
    packet['range_attack']  = self.range_attack
    packet['mage_attack']   = self.mage_attack
    packet['melee_defense'] = self.melee_defense
    packet['range_defense'] = self.range_defense
    packet['mage_defense']  = self.mage_defense

    return packet


class Inventory:
  def __init__(self, realm, entity):
    config           = realm.config
    self.realm       = realm
    self.entity      = entity
    self.config      = config

    self.equipment   = Equipment()
    self.capacity = 0

    if config.ITEM_SYSTEM_ENABLED:
      self.capacity         = config.ITEM_INVENTORY_CAPACITY

      self._item_stacks: Dict[Tuple, Item.Stack] = {}
      self.items: OrderedSet[Item.Item] = OrderedSet([])

  @property
  def space(self):
    return self.capacity - len(self.items)

  def has_stack(self, signature: Tuple) -> bool:
    return signature in self._item_stacks

  def packet(self):
    item_packet = []
    if self.config.ITEM_SYSTEM_ENABLED:
      item_packet = [e.packet for e in self.items]

    return {
          'items':     item_packet,
          'equipment': self.equipment.packet}

  def __iter__(self):
    for item in self.items:
      yield item

  def receive(self, item: Item.Item):
    assert isinstance(item, Item.Item), f'{item} received is not an Item instance'
    assert item not in self.items, f'{item} object received already in inventory'
    assert not item.equipped.val, f'Received equipped item {item}'
    assert not item.listed_price.val, f'Received listed item {item}'
    assert item.quantity.val, f'Received empty item {item}'

    if isinstance(item, Item.Stack):
      signature = item.signature
      if self.has_stack(signature):
        stack = self._item_stacks[signature]
        assert item.level.val == stack.level.val, f'{item} stack level mismatch'
        stack.quantity.increment(item.quantity.val)
        # destroy the original item instance after the transfer is complete
        item.destroy()
        return

      if not self.space:
        # if no space thus cannot receive, just destroy the item
        item.destroy()
        return

      self._item_stacks[signature] = item

    if not self.space:
      # if no space thus cannot receive, just destroy the item
      item.destroy()
      return

    self.realm.log_milestone(f'Receive_{item.__class__.__name__}', item.level.val,
      f'INVENTORY: Received level {item.level.val} {item.__class__.__name__}',
      tags={"player_id": self.entity.ent_id})

    item.owner_id.update(self.entity.id.val)
    self.items.add(item)

  # pylint: disable=protected-access
  def remove(self, item, quantity=None):
    assert isinstance(item, Item.Item), f'{item} removing item is not an Item instance'
    assert item in self.items, f'No item {item} to remove'

    if isinstance(item, Item.Equipment) and item.equipped.val:
      item.unequip(item._slot(self.entity))

    if isinstance(item, Item.Stack):
      signature = item.signature

      assert self.has_stack(item.signature), f'{item} stack to remove not in inventory'
      stack = self._item_stacks[signature]

      if quantity is None or stack.quantity.val == quantity:
        self._remove(stack)
        del self._item_stacks[signature]
        return

      assert 0 < quantity <= stack.quantity.val, \
        f'Invalid remove {quantity} x {item} ({stack.quantity.val} available)'
      stack.quantity.val -= quantity
      return

    self._remove(item)

  def _remove(self, item):
    self.realm.exchange.unlist_item(item)
    item.owner_id.update(0)
    self.items.remove(item)
