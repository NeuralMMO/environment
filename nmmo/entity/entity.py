
import math
from types import SimpleNamespace

import numpy as np

from nmmo.core.config import Config
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
    "latest_combat_tick",
    "message",

    # Resources
    "gold",
    "health",
    "food",
    "water",

    # Combat Skills
    "melee_level",
    "melee_exp",
    "range_level",
    "range_exp",
    "mage_level",
    "mage_exp",

    # Harvest Skills
    "fishing_level",
    "fishing_exp",
    "herbalism_level",
    "herbalism_exp",
    "prospecting_level",
    "prospecting_exp",
    "carving_level",
    "carving_exp",
    "alchemy_level",
    "alchemy_exp",
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
    "latest_combat_tick": (0, math.inf),
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
    "melee_exp": (0, math.inf),
    "range_level": (0, config.PROGRESSION_LEVEL_MAX),
    "range_exp": (0, math.inf),
    "mage_level": (0, config.PROGRESSION_LEVEL_MAX),
    "mage_exp": (0, math.inf),
    "fishing_level": (0, config.PROGRESSION_LEVEL_MAX),
    "fishing_exp": (0, math.inf),
    "herbalism_level": (0, config.PROGRESSION_LEVEL_MAX),
    "herbalism_exp": (0, math.inf),
    "prospecting_level": (0, config.PROGRESSION_LEVEL_MAX),
    "prospecting_exp": (0, math.inf),
    "carving_level": (0, config.PROGRESSION_LEVEL_MAX),
    "carving_exp": (0, math.inf),
    "alchemy_level": (0, config.PROGRESSION_LEVEL_MAX),
    "alchemy_exp": (0, math.inf),
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
    self.health_restore = 0
    self.resilient = False

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

    org_health = self.health.val
    if food_thresh and water_thresh:
      restore = np.floor(self.health.max * regen)
      self.health.increment(restore)

    if self.food.empty:
      starvation_damage = self.config.RESOURCE_STARVATION_RATE
      if self.resilient:
        starvation_damage *= self.config.RESOURCE_DAMAGE_REDUCTION
      self.health.decrement(int(starvation_damage))

    if self.water.empty:
      dehydration_damage = self.config.RESOURCE_DEHYDRATION_RATE
      if self.resilient:
        dehydration_damage *= self.config.RESOURCE_DAMAGE_REDUCTION
      self.health.decrement(int(dehydration_damage))

    # records both increase and decrease in health due to food and water
    self.health_restore = self.health.val - org_health

  def packet(self):
    data = {}
    data['health'] = { 'val': self.health.val, 'max': self.config.PLAYER_BASE_HEALTH }
    data['food'] = { 'val': self.food.val, 'max': self.config.RESOURCE_BASE }
    data['water'] = { 'val': self.water.val, 'max': self.config.RESOURCE_BASE }
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


# NOTE: History.packet() is actively used in visulazing attacks
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

    self.time_alive.increment()

  def packet(self):
    data = {}
    data['damage'] = self.damage.val
    data['timeAlive'] = self.time_alive.val
    data['damage_inflicted'] = self.damage_inflicted
    data['damage_received'] = self.damage_received

    if self.attack is not None:
      data['attack'] = self.attack

    # NOTE: the client seems to use actions for visualization
    #   but produces errors with the new actions. So we comment out these for now
    # actions = {}
    # for atn, args in self.actions.items():
    #   atn_packet = {}

    #   # Avoid recursive player packet
    #   if atn.__name__ == 'Attack':
    #     continue

    #   for key, val in args.items():
    #     if hasattr(val, 'packet'):
    #       atn_packet[key.__name__] = val.packet
    #     else:
    #       atn_packet[key.__name__] = val.__name__
    #   actions[atn.__name__] = atn_packet
    # data['actions'] = actions
    data['actions'] = {}

    return data

# pylint: disable=no-member
class Entity(EntityState):
  def __init__(self, realm, pos, entity_id, name):
    super().__init__(realm.datastore, EntityState.Limits(realm.config))

    self.realm = realm
    self.config: Config = realm.config
    # TODO: do not access realm._np_random directly
    #   related to the whole NPC, scripted logic
    # pylint: disable=protected-access
    self._np_random = realm._np_random

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
    return self.row.val, self.col.val

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

  @property
  def in_combat(self) -> bool:
    # NOTE: the initial latest_combat_tick is 0, and valid values are greater than 0
    if not self.config.COMBAT_SYSTEM_ENABLED or self.latest_combat_tick.val == 0:
      return False

    return (self.realm.tick - self.latest_combat_tick.val) < self.config.COMBAT_STATUS_DURATION
