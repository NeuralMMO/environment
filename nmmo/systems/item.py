from __future__ import annotations

import math
from abc import ABC
from types import SimpleNamespace
from typing import Dict

from nmmo.datastore.serialized import SerializedState
from nmmo.lib.colors import Tier
from nmmo.lib.event_code import EventCode

# pylint: disable=no-member
ItemState = SerializedState.subclass("Item", [
  "id",
  "type_id",
  "owner_id",

  "level",
  "capacity",
  "quantity",
  "melee_attack",
  "range_attack",
  "mage_attack",
  "melee_defense",
  "range_defense",
  "mage_defense",
  "health_restore",
  "resource_restore",
  "equipped",

  # Market
  "listed_price",
])

# TODO: These limits should be defined in the config.
ItemState.Limits = lambda config: {
  "id": (0, math.inf),
  "type_id": (0, 99),
  "owner_id": (-math.inf, math.inf),
  "level": (0, 99),
  "capacity": (0, 99),
  "quantity": (0, math.inf), # NOTE: Ammunitions can be stacked infinitely
  "melee_attack": (0, 100),
  "range_attack": (0, 100),
  "mage_attack": (0, 100),
  "melee_defense": (0, 100),
  "range_defense": (0, 100),
  "mage_defense": (0, 100),
  "health_restore": (0, 100),
  "resource_restore": (0, 100),
  "equipped": (0, 1),
  "listed_price": (0, math.inf),
}

ItemState.Query = SimpleNamespace(
  table=lambda ds: ds.table("Item").where_neq(
    ItemState.State.attr_name_to_col["id"], 0),

  by_id=lambda ds, id: ds.table("Item").where_eq(
    ItemState.State.attr_name_to_col["id"], id),

  owned_by = lambda ds, id: ds.table("Item").where_eq(
    ItemState.State.attr_name_to_col["owner_id"], id),

  for_sale = lambda ds: ds.table("Item").where_neq(
    ItemState.State.attr_name_to_col["listed_price"], 0),
)

class Item(ItemState):
  ITEM_TYPE_ID = None
  _item_type_id_to_class: Dict[int, type] = {}

  @staticmethod
  def register(item_type):
    assert item_type.ITEM_TYPE_ID is not None
    if item_type.ITEM_TYPE_ID not in Item._item_type_id_to_class:
      Item._item_type_id_to_class[item_type.ITEM_TYPE_ID] = item_type

  @staticmethod
  def item_class(type_id: int):
    return Item._item_type_id_to_class[type_id]

  def __init__(self, realm, level,
              capacity=0,
              melee_attack=0, range_attack=0, mage_attack=0,
              melee_defense=0, range_defense=0, mage_defense=0,
              health_restore=0, resource_restore=0):

    super().__init__(realm.datastore, ItemState.Limits(realm.config))
    self.realm = realm
    self.config = realm.config

    Item.register(self.__class__)

    self.id.update(self.datastore_record.id)
    self.type_id.update(self.ITEM_TYPE_ID)
    self.level.update(level)
    self.capacity.update(capacity)
    # every item instance is created individually, i.e., quantity=1
    self.quantity.update(1)
    self.melee_attack.update(melee_attack)
    self.range_attack.update(range_attack)
    self.mage_attack.update(mage_attack)
    self.melee_defense.update(melee_defense)
    self.range_defense.update(range_defense)
    self.mage_defense.update(mage_defense)
    self.health_restore.update(health_restore)
    self.resource_restore.update(resource_restore)
    realm.items[self.id.val] = self

  def destroy(self):
    # NOTE: we may want to track the item lifecycle and
    #   and see how many high-level items are wasted
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      self.realm.exchange.unlist_item(self)
    if self.owner_id.val in self.realm.players:
      self.realm.players[self.owner_id.val].inventory.remove(self)
    self.realm.items.pop(self.id.val, None)
    self.datastore_record.delete()

  @property
  def packet(self):
    return {'item':             self.__class__.__name__,
            'level':            self.level.val,
            'capacity':         self.capacity.val,
            'quantity':         self.quantity.val,
            'melee_attack':     self.melee_attack.val,
            'range_attack':     self.range_attack.val,
            'mage_attack':      self.mage_attack.val,
            'melee_defense':    self.melee_defense.val,
            'range_defense':    self.range_defense.val,
            'mage_defense':     self.mage_defense.val,
            'health_restore':   self.health_restore.val,
            'resource_restore': self.resource_restore.val,
            }

  def _level(self, entity):
    # this is for armors, ration, and potion
    # weapons and tools must override this with specific skills
    return entity.level

  def level_gt(self, entity):
    return self.level.val > self._level(entity)

  def use(self, entity) -> bool:
    raise NotImplementedError

class Stack:
  @property
  def signature(self):
    return (self.type_id.val, self.level.val)

class Equipment(Item):
  @property
  def packet(self):
    packet = {'color': self.color.packet()}
    return {**packet, **super().packet}

  @property
  def color(self):
    if self.level == 0:
      return Tier.BLACK
    if self.level < 10:
      return Tier.WOOD
    if self.level < 20:
      return Tier.BRONZE
    if self.level < 40:
      return Tier.SILVER
    if self.level < 60:
      return Tier.GOLD
    if self.level < 80:
      return Tier.PLATINUM
    return Tier.DIAMOND

  def unequip(self, equip_slot):
    assert self.equipped.val == 1
    self.equipped.update(0)
    equip_slot.unequip()

  def equip(self, entity, equip_slot):
    assert self.equipped.val == 0
    if self._level(entity) < self.level.val:
      return

    self.equipped.update(1)
    equip_slot.equip(self)

  def _slot(self, entity):
    raise NotImplementedError

  def use(self, entity):
    assert self in entity.inventory, "Item is not in entity's inventory"
    assert self.listed_price == 0, "Listed item cannot be used"
    assert self._level(entity) >= self.level.val, "Entity's level is not sufficient to use the item"

    if self.equipped.val:
      self.unequip(self._slot(entity))
    else:
      # always empty the slot first
      self._slot(entity).unequip()
      self.equip(entity, self._slot(entity))
      self.realm.event_log.record(EventCode.EQUIP_ITEM, entity, item=self)

class Armor(Equipment, ABC):
  def __init__(self, realm, level, **kwargs):
    defense = realm.config.EQUIPMENT_ARMOR_BASE_DEFENSE + \
              level*realm.config.EQUIPMENT_ARMOR_LEVEL_DEFENSE
    super().__init__(realm, level,
                     melee_defense=defense,
                     range_defense=defense,
                     mage_defense=defense,
                     **kwargs)
class Hat(Armor):
  ITEM_TYPE_ID = 2
  def _slot(self, entity):
    return entity.inventory.equipment.hat
class Top(Armor):
  ITEM_TYPE_ID = 3
  def _slot(self, entity):
    return entity.inventory.equipment.top
class Bottom(Armor):
  ITEM_TYPE_ID = 4
  def _slot(self, entity):
    return entity.inventory.equipment.bottom


class Weapon(Equipment):
  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.attack = (
      realm.config.EQUIPMENT_WEAPON_BASE_DAMAGE +
      level*realm.config.EQUIPMENT_WEAPON_LEVEL_DAMAGE)

  def _slot(self, entity):
    return entity.inventory.equipment.held

class Spear(Weapon):
  ITEM_TYPE_ID = 5

  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.melee_attack.update(self.attack)

  def _level(self, entity):
    return entity.skills.melee.level.val
class Bow(Weapon):
  ITEM_TYPE_ID = 6

  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.range_attack.update(self.attack)

  def _level(self, entity):
    return entity.skills.range.level.val
class Wand(Weapon):
  ITEM_TYPE_ID = 7

  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.mage_attack.update(self.attack)

  def _level(self, entity):
    return entity.skills.mage.level.val


class Tool(Equipment):
  def __init__(self, realm, level, **kwargs):
    defense = realm.config.EQUIPMENT_TOOL_BASE_DEFENSE + \
        level*realm.config.EQUIPMENT_TOOL_LEVEL_DEFENSE
    super().__init__(realm, level,
                      melee_defense=defense,
                      range_defense=defense,
                      mage_defense=defense,
                      **kwargs)

  def _slot(self, entity):
    return entity.inventory.equipment.held
class Rod(Tool):
  ITEM_TYPE_ID = 8
  def _level(self, entity):
    return entity.skills.fishing.level.val
class Gloves(Tool):
  ITEM_TYPE_ID = 9
  def _level(self, entity):
    return entity.skills.herbalism.level.val
class Pickaxe(Tool):
  ITEM_TYPE_ID = 10
  def _level(self, entity):
    return entity.skills.prospecting.level.val
class Axe(Tool):
  ITEM_TYPE_ID = 11
  def _level(self, entity):
    return entity.skills.carving.level.val
class Chisel(Tool):
  ITEM_TYPE_ID = 12
  def _level(self, entity):
    return entity.skills.alchemy.level.val


class Ammunition(Equipment, Stack):
  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.attack = (
      realm.config.EQUIPMENT_AMMUNITION_BASE_DAMAGE +
      level*realm.config.EQUIPMENT_AMMUNITION_LEVEL_DAMAGE)

  def _slot(self, entity):
    return entity.inventory.equipment.ammunition

  def fire(self, entity) -> int:
    assert self.equipped.val > 0, 'Ammunition not equipped'
    assert self.quantity.val > 0, 'Used ammunition with 0 quantity'

    self.quantity.decrement()

    if self.quantity.val == 0:
      entity.inventory.remove(self)
      # delete this empty item instance from the datastore
      self.destroy()

    self.realm.event_log.record(EventCode.FIRE_AMMO, entity, item=self)
    return self.damage

class Whetstone(Ammunition):
  ITEM_TYPE_ID = 13

  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.melee_attack.update(self.attack)

  def _level(self, entity):
    return entity.skills.melee.level.val

  @property
  def damage(self):
    return self.melee_attack.val

class Arrow(Ammunition):
  ITEM_TYPE_ID = 14

  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.range_attack.update(self.attack)

  def _level(self, entity):
    return entity.skills.range.level.val

  @property
  def damage(self):
    return self.range_attack.val

class Runes(Ammunition):
  ITEM_TYPE_ID = 15

  def __init__(self, realm, level, **kwargs):
    super().__init__(realm, level, **kwargs)
    self.mage_attack.update(self.attack)

  def _level(self, entity):
    return entity.skills.mage.level.val

  @property
  def damage(self):
    return self.mage_attack.val


# NOTE: Each consumable item (ration, potion) cannot be stacked,
#   so each item takes 1 inventory space
class Consumable(Item):
  def use(self, entity) -> bool:
    assert self in entity.inventory, "Item is not in entity's inventory"
    assert self.listed_price == 0, "Listed item cannot be used"
    assert self._level(entity) >= self.level.val, "Entity's level is not sufficient to use the item"

    self.realm.event_log.record(EventCode.CONSUME_ITEM, entity, item=self)
    self._apply_effects(entity)
    entity.inventory.remove(self)
    self.destroy()
    return True

class Ration(Consumable):
  ITEM_TYPE_ID = 16

  def __init__(self, realm, level, **kwargs):
    restore = 0
    if realm.config.PROFESSION_SYSTEM_ENABLED:
      restore = realm.config.PROFESSION_CONSUMABLE_RESTORE(level)
    super().__init__(realm, level, resource_restore=restore, **kwargs)

  def _apply_effects(self, entity):
    entity.resources.food.increment(self.resource_restore.val)
    entity.resources.water.increment(self.resource_restore.val)

class Potion(Consumable):
  ITEM_TYPE_ID = 17

  def __init__(self, realm, level, **kwargs):
    restore = 0
    if realm.config.PROFESSION_SYSTEM_ENABLED:
      restore = realm.config.PROFESSION_CONSUMABLE_RESTORE(level)
    super().__init__(realm, level, health_restore=restore, **kwargs)

  def _apply_effects(self, entity):
    entity.resources.health.increment(self.health_restore.val)
    entity.poultice_consumed += 1
    entity.poultice_level_consumed = max(
      entity.poultice_level_consumed, self.level.val)

# Item groupings
ARMOR = [Hat, Top, Bottom]
WEAPON = [Spear, Bow, Wand]
TOOL = [Rod, Gloves, Pickaxe, Axe, Chisel]
AMMUNITION = [Whetstone, Arrow, Runes]
CONSUMABLE = [Ration, Potion]
ALL_ITEM = ARMOR + WEAPON + TOOL + AMMUNITION + CONSUMABLE
