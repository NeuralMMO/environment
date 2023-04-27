
import math
from types import SimpleNamespace

import numpy as np

from nmmo.core.config import Config
from nmmo.lib import utils
from nmmo.datastore.serialized import SerializedState
from nmmo.systems import inventory
from nmmo.lib.log import EventCode

# pylint: disable=no-member
EntityState = SerializedState.subclass(
  "Entity", [
    "id",
    "npc_type", # 1 - passive, 2 - neutral, 3 - aggressive
    "row",
    "col",

    # Status
    "damage",
    "time_alive",
    "freeze",
    "item_level",
    "attacker_id",
    "message",

    # Resources
    "gold",
    "health",
    "food",
    "water",

    # Combat
    "melee_level",
    "range_level",
    "mage_level",

    # Skills
    "fishing_level",
    "herbalism_level",
    "prospecting_level",
    "carving_level",
    "alchemy_level",
  ])

EntityState.Limits = lambda config: {
  **{
    "id": (-math.inf, math.inf),
    "npc_type": (0, 4),
    "row": (0, config.MAP_SIZE-1),
    "col": (0, config.MAP_SIZE-1),
    "damage": (0, math.inf),
    "time_alive": (0, math.inf),
    "freeze": (0, 3),
    "item_level": (0, 5*config.NPC_LEVEL_MAX),
    "attacker_id": (-np.inf, math.inf),
    "health": (0, config.PLAYER_BASE_HEALTH),
  },
  **({
    "message": (0, config.COMMUNICATION_NUM_TOKENS),
  } if config.COMMUNICATION_SYSTEM_ENABLED else {}),
  **({
    "gold": (0, math.inf),
    "food": (0, config.RESOURCE_BASE),
    "water": (0, config.RESOURCE_BASE),
  } if config.RESOURCE_SYSTEM_ENABLED else {}),
  **({
    "melee_level": (0, config.PROGRESSION_LEVEL_MAX),
    "range_level": (0, config.PROGRESSION_LEVEL_MAX),
    "mage_level": (0, config.PROGRESSION_LEVEL_MAX),
    "fishing_level": (0, config.PROGRESSION_LEVEL_MAX),
    "herbalism_level": (0, config.PROGRESSION_LEVEL_MAX),
    "prospecting_level": (0, config.PROGRESSION_LEVEL_MAX),
    "carving_level": (0, config.PROGRESSION_LEVEL_MAX),
    "alchemy_level": (0, config.PROGRESSION_LEVEL_MAX),
  } if config.PROGRESSION_SYSTEM_ENABLED else {}),
}

EntityState.Query = SimpleNamespace(
  # Whole table
  table=lambda ds: ds.table("Entity").where_neq(
    EntityState.State.attr_name_to_col["id"], 0),

  # Single entity
  by_id=lambda ds, id: ds.table("Entity").where_eq(
    EntityState.State.attr_name_to_col["id"], id)[0],

  # Multiple entities
  by_ids=lambda ds, ids: ds.table("Entity").where_in(
    EntityState.State.attr_name_to_col["id"], ids),

  # Entities in a radius
  window=lambda ds, r, c, radius: ds.table("Entity").window(
    EntityState.State.attr_name_to_col["row"],
    EntityState.State.attr_name_to_col["col"],
    r, c, radius),
)

class Resources:
  def __init__(self, ent, config):
    self.config = config
    self.health = ent.health
    self.water = ent.water
    self.food = ent.food

    self.health.update(config.PLAYER_BASE_HEALTH)
    if config.RESOURCE_SYSTEM_ENABLED:
      self.water.update(config.RESOURCE_BASE)
      self.food.update(config.RESOURCE_BASE)

  def update(self):
    if not self.config.RESOURCE_SYSTEM_ENABLED:
      return

    regen = self.config.RESOURCE_HEALTH_RESTORE_FRACTION
    thresh = self.config.RESOURCE_HEALTH_REGEN_THRESHOLD

    food_thresh = self.food > thresh * self.config.RESOURCE_BASE
    water_thresh = self.water > thresh * self.config.RESOURCE_BASE

    if food_thresh and water_thresh:
      restore = np.floor(self.health.max * regen)
      self.health.increment(restore)

    if self.food.empty:
      self.health.decrement(self.config.RESOURCE_STARVATION_RATE)

    if self.water.empty:
      self.health.decrement(self.config.RESOURCE_DEHYDRATION_RATE)

  def packet(self):
    data = {}
    data['health'] = self.health.val
    data['food'] = self.food.val
    data['water'] = self.water.val
    return data

class Status:
  def __init__(self, ent):
    self.freeze = ent.freeze

  def update(self):
    if self.freeze.val > 0:
      self.freeze.decrement(1)

  def packet(self):
    data = {}
    data['freeze'] = self.freeze.val
    return data


class History:
  def __init__(self, ent):
    self.actions = {}
    self.attack = None

    self.starting_position = ent.pos
    self.exploration = 0
    self.player_kills = 0

    self.damage_received = 0
    self.damage_inflicted = 0

    self.damage = ent.damage
    self.time_alive = ent.time_alive

    self.last_pos = None

  def update(self, entity, actions):
    self.attack = None
    self.damage.update(0)

    self.actions = {}
    if entity.ent_id in actions:
      self.actions = actions[entity.ent_id]

    exploration = utils.linf(entity.pos, self.starting_position)
    self.exploration = max(exploration, self.exploration)

    self.time_alive.increment()

  def packet(self):
    data = {}
    data['damage'] = self.damage.val
    data['timeAlive'] = self.time_alive.val
    data['damage_inflicted'] = self.damage_inflicted
    data['damage_received'] = self.damage_received

    if self.attack is not None:
      data['attack'] = self.attack

    actions = {}
    for atn, args in self.actions.items():
      atn_packet = {}

      # Avoid recursive player packet
      if atn.__name__ == 'Attack':
        continue

      for key, val in args.items():
        if hasattr(val, 'packet'):
          atn_packet[key.__name__] = val.packet
        else:
          atn_packet[key.__name__] = val.__name__
      actions[atn.__name__] = atn_packet
    data['actions'] = actions

    return data

# pylint: disable=no-member
class Entity(EntityState):
  def __init__(self, realm, pos, entity_id, name):
    super().__init__(realm.datastore, EntityState.Limits(realm.config))

    self.realm = realm
    self.config: Config = realm.config

    self.policy = name
    self.entity_id = entity_id
    self.repr = None

    self.name = name + str(entity_id)

    self.row.update(pos[0])
    self.col.update(pos[1])
    self.id.update(entity_id)

    self.vision = self.config.PLAYER_VISION_RADIUS

    self.attacker = None
    self.target = None
    self.closest = None
    self.spawn_pos = pos

    # Submodules
    self.status = Status(self)
    self.history = History(self)
    self.resources = Resources(self, self.config)
    self.inventory = inventory.Inventory(realm, self)

  @property
  def ent_id(self):
    return self.id.val

  def packet(self):
    data = {}

    data['status'] = self.status.packet()
    data['history'] = self.history.packet()
    data['inventory'] = self.inventory.packet()
    data['alive'] = self.alive
    data['base'] = {
      'r': self.row.val,
      'c': self.col.val,
      'name': self.name,
      'level': self.attack_level,
      'item_level': self.item_level.val,
    }

    return data

  def update(self, realm, actions):
    '''Update occurs after actions, e.g. does not include history'''
    if self.history.damage == 0:
      self.attacker = None
      self.attacker_id.update(0)

    if realm.config.EQUIPMENT_SYSTEM_ENABLED:
      self.item_level.update(self.equipment.total(lambda e: e.level))

    self.status.update()
    self.history.update(self, actions)

  # Returns True if the entity is alive
  def receive_damage(self, source, dmg):
    self.history.damage_received += dmg
    self.history.damage.update(dmg)
    self.resources.health.decrement(dmg)

    if self.alive:
      return True

    # at this point, self is dead
    if source:
      source.history.player_kills += 1
      self.realm.event_log.record(EventCode.PLAYER_KILL, source, target=self)

    # if self is dead, unlist its items from the market regardless of looting
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      for item in list(self.inventory.items):
        self.realm.exchange.unlist_item(item)

    # if self is dead but no one can loot, destroy its items
    if source is None or not source.is_player: # nobody or npcs cannot loot
      if self.config.ITEM_SYSTEM_ENABLED:
        for item in list(self.inventory.items):
          item.destroy()
      return False

    # now, source can loot the dead self
    return False

  # pylint: disable=unused-argument
  def apply_damage(self, dmg, style):
    self.history.damage_inflicted += dmg

  @property
  def pos(self):
    return int(self.row.val), int(self.col.val)

  @property
  def alive(self):
    if self.resources.health.empty:
      return False

    return True

  @property
  def is_player(self) -> bool:
    return False

  @property
  def is_npc(self) -> bool:
    return False

  @property
  def attack_level(self) -> int:
    melee = self.skills.melee.level.val
    ranged = self.skills.range.level.val
    mage = self.skills.mage.level.val

    return int(max(melee, ranged, mage))
