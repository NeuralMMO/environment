import numpy as np
from nmmo.entity import entity
from nmmo.core import action as Action
from nmmo.systems import combat, droptable
from nmmo.systems import item as Item
from nmmo.systems import skill
from nmmo.systems.inventory import EquipmentSlot
from nmmo.lib.event_code import EventCode
from nmmo.lib import utils, astar


DIRECTIONS = [ # row delta, col delta, action
      (-1, 0, Action.North),
      (1, 0, Action.South),
      (0, -1, Action.West),
      (0, 1, Action.East)] * 2
DELTA_TO_DIR = {(r, c): atn for r, c, atn in DIRECTIONS}
DELTA_TO_DIR[(0, 0)] = None

def get_habitable_dir(ent):
  r, c = ent.pos
  is_habitable = ent.realm.map.habitable_tiles
  start = ent._np_random.get_direction()  # pylint: disable=protected-access
  for i in range(4):
    delta_r, delta_c, direction = DIRECTIONS[start + i]
    if is_habitable[r + delta_r, c + delta_c]:
      return direction
  return Action.North

def meander_toward(ent, goal, dist_crit=10, toward_weight=3):
  r, c = ent.pos
  delta_r, delta_c = goal[0] - r, goal[1] - c
  abs_dr, abs_dc = abs(delta_r), abs(delta_c)
  dist_l1 = abs_dr + abs_dc
  # If close (less than dist_crit), use expensive aStar
  if dist_l1 <= dist_crit:
    delta = astar.aStar(ent.realm.map, ent.pos, goal)
    return move_action(DELTA_TO_DIR[delta] if delta in DELTA_TO_DIR else None)

  # Otherwise, use a weighted random walk
  cand_dirs = []
  weights = []
  for i in range(4):
    r_offset, c_offset, direction = DIRECTIONS[i]
    if ent.realm.map.habitable_tiles[r + r_offset, c + c_offset]:
      cand_dirs.append(direction)
      weights.append(1)
      if r_offset * delta_r > 0:
        weights[-1] += toward_weight * abs_dr/dist_l1
      if c_offset * delta_c > 0:
        weights[-1] += toward_weight * abs_dc/dist_l1
  if len(cand_dirs) == 0:
    return move_action(Action.North)
  if len(cand_dirs) == 1:
    return move_action(cand_dirs[0])
  weights = np.array(weights)
  # pylint: disable=protected-access
  return move_action(ent._np_random.choice(cand_dirs, p=weights/np.sum(weights)))

def move_action(direction):
  return {Action.Move: {Action.Direction: direction}} if direction else {}


class Equipment:
  def __init__(self, total,
    melee_attack, range_attack, mage_attack,
    melee_defense, range_defense, mage_defense):

    self.level         = total
    self.ammunition    = EquipmentSlot()

    self.melee_attack  = melee_attack
    self.range_attack  = range_attack
    self.mage_attack   = mage_attack
    self.melee_defense = melee_defense
    self.range_defense = range_defense
    self.mage_defense  = mage_defense

  def total(self, getter):
    return getter(self)

  # pylint: disable=R0801
  # Similar lines here and in inventory.py
  @property
  def packet(self):
    packet = {}
    packet["item_level"]    = self.total
    packet["melee_attack"]  = self.melee_attack
    packet["range_attack"]  = self.range_attack
    packet["mage_attack"]   = self.mage_attack
    packet["melee_defense"] = self.melee_defense
    packet["range_defense"] = self.range_defense
    packet["mage_defense"]  = self.mage_defense
    return packet


# pylint: disable=no-member
class NPC(entity.Entity):
  def __init__(self, realm, pos, iden, name, npc_type):
    super().__init__(realm, pos, iden, name)
    self.skills = skill.Combat(realm, self)
    self.realm = realm
    self.last_action = None
    self.droptable = None
    self.spawn_danger = None
    self.equipment = None
    self.npc_type.update(npc_type)

  @property
  def is_npc(self) -> bool:
    return True

  def update(self, realm, actions):
    super().update(realm, actions)

    if not self.alive:
      return

    self.resources.health.increment(1)
    self.last_action = actions

  def can_see(self, target):
    if target is None or target.immortal:
      return False
    distance = utils.linf_single(self.pos, target.pos)
    return distance <= self.vision

  def _move_toward(self, goal):
    delta = astar.aStar(self.realm.map, self.pos, goal)
    return move_action(DELTA_TO_DIR[delta] if delta in DELTA_TO_DIR else None)

  def _meander(self):
    return move_action(get_habitable_dir(self))

  def can_attack(self, target):
    if target is None or not self.config.NPC_SYSTEM_ENABLED or target.immortal:
      return False
    if not self.config.NPC_ALLOW_ATTACK_OTHER_NPCS and target.is_npc:
      return False
    distance = utils.linf_single(self.pos, target.pos)
    return distance <= self.skills.style.attack_range(self.realm.config)

  def _has_target(self, search=False):
    if self.target and (not self.target.alive or not self.can_see(self.target)):
      self.target = None
    # NOTE: when attacked by several agents, this will always target the last attacker
    if self.attacker and self.target is None:
      self.target = self.attacker
    if self.target is None and search is True:
      self.target = utils.identify_closest_target(self)
    return self.target

  def _add_attack_action(self, actions, target):
    actions.update({Action.Attack: {Action.Style: self.skills.style, Action.Target: target}})

  def _charge_toward(self, target):
    actions = self._move_toward(target.pos)
    if self.can_attack(target):
      self._add_attack_action(actions, target)
    return actions

  # Returns True if the entity is alive
  def receive_damage(self, source, dmg):
    if super().receive_damage(source, dmg):
      return True

    # run the next lines if the npc is killed
    # source receive gold & items in the droptable
    # pylint: disable=no-member
    if self.gold.val > 0:
      source.gold.increment(self.gold.val)
      self.realm.event_log.record(EventCode.LOOT_GOLD, source, amount=self.gold.val, target=self)
      self.gold.update(0)

    if self.droptable:
      for item in self.droptable.roll(self.realm, self.attack_level):
        if source.is_player and source.inventory.space:
          # inventory.receive() returns True if the item is received
          # if source does not have space, inventory.receive() destroys the item
          if source.inventory.receive(item):
            self.realm.event_log.record(EventCode.LOOT_ITEM, source, item=item, target=self)
        else:
          item.destroy()

    return False

  @staticmethod
  def default_spawn(realm, pos, iden, np_random, danger=None):
    config = realm.config

    # check the position
    if realm.map.tiles[pos].impassible:
      return None

    # Select AI Policy
    danger = danger or combat.danger(config, pos)
    if danger >= config.NPC_SPAWN_AGGRESSIVE:
      ent = Aggressive(realm, pos, iden)
    elif danger >= config.NPC_SPAWN_NEUTRAL:
      ent = PassiveAggressive(realm, pos, iden)
    elif danger >= config.NPC_SPAWN_PASSIVE:
      ent = Passive(realm, pos, iden)
    else:
      return None

    ent.spawn_danger = danger

    # Select combat focus
    style = np_random.integers(0,3)
    if style == 0:
      style = Action.Melee
    elif style == 1:
      style = Action.Range
    else:
      style = Action.Mage
    ent.skills.style = style

    # Compute level
    level = 0
    if config.PROGRESSION_SYSTEM_ENABLED:
      level_min = config.NPC_LEVEL_MIN
      level_max = config.NPC_LEVEL_MAX
      level     = int(danger * (level_max - level_min) + level_min)

      # Set skill levels
      if style == Action.Melee:
        ent.skills.melee.set_experience_by_level(level)
      elif style == Action.Range:
        ent.skills.range.set_experience_by_level(level)
      elif style == Action.Mage:
        ent.skills.mage.set_experience_by_level(level)

    # Gold
    if config.EXCHANGE_SYSTEM_ENABLED:
      # pylint: disable=no-member
      ent.gold.update(level)

    ent.droptable = droptable.Standard()

    # Equipment to instantiate
    if config.EQUIPMENT_SYSTEM_ENABLED:
      lvl     = level - np_random.random()
      ilvl    = int(5 * lvl)

      offense = int(config.NPC_BASE_DAMAGE + lvl*config.NPC_LEVEL_DAMAGE)
      defense = int(config.NPC_BASE_DEFENSE + lvl*config.NPC_LEVEL_DEFENSE)

      ent.equipment = Equipment(ilvl, offense, offense, offense, defense, defense, defense)

      armor =  [Item.Hat, Item.Top, Item.Bottom]
      ent.droptable.add(np_random.choice(armor))

    if config.PROFESSION_SYSTEM_ENABLED:
      tools =  [Item.Rod, Item.Gloves, Item.Pickaxe, Item.Axe, Item.Chisel]
      ent.droptable.add(np_random.choice(tools))

    return ent

  def packet(self):
    data = super().packet()
    data["skills"]   = self.skills.packet()
    data["resource"] = { "health": {
      "val": self.resources.health.val, "max": self.config.PLAYER_BASE_HEALTH } }
    return data

class Passive(NPC):
  def __init__(self, realm, pos, iden, name=None):
    super().__init__(realm, pos, iden, name or "Passive", 1)

  def decide(self):
    # Move only, no attack
    return self._meander()

class PassiveAggressive(NPC):
  def __init__(self, realm, pos, iden, name=None):
    super().__init__(realm, pos, iden, name or "Neutral", 2)

  def decide(self):
    if self._has_target() is None:
      return self._meander()
    return self._charge_toward(self.target)

class Aggressive(NPC):
  def __init__(self, realm, pos, iden, name=None):
    super().__init__(realm, pos, iden, name or "Hostile", 3)

  def decide(self):
    if self._has_target(search=True) is None:
      return self._meander()
    return self._charge_toward(self.target)

class Soldier(NPC):
  def __init__(self, realm, pos, iden, name, order):
    super().__init__(realm, pos, iden, name or "Soldier", 3)  # Hostile with order
    self.target_entity = None
    self.rally_point = None
    self._process_order(order)

  def _process_order(self, order):
    if order is None:
      return
    if "destroy" in order:  # destroy the specified entity id
      self.target_entity = self.realm.entity(order["destroy"])
    if "rally" in order:
      # rally until spotting an enemy
      self.rally_point = order["rally"]  # (row, col)

  def _is_order_done(self, radius=5):
    if self.target_entity and not self.target_entity.alive:
      self.target_entity = None
    if self.rally_point and utils.linf_single(self.pos, self.rally_point) <= radius:
      self.rally_point = None

  def decide(self):
    self._is_order_done()
    # NOTE: destroying the target entity is the highest priority
    if self.target_entity is None and self._has_target(search=True):
      if self.can_attack(self.target):
        return self._charge_toward(self.target)

    actions = self._decide_move_action()
    self._decide_attack_action(actions)
    return actions

  def _decide_move_action(self):
    # in the order of priority
    if self.target_entity:
      return self._move_toward(self.target_entity.pos)
    if self.target:
      # If it"s close enough, it will use A*. Otherwise, random.
      return meander_toward(self, self.target.pos)
    if self.rally_point:
      return meander_toward(self, self.rally_point)
    return self._meander()

  def _decide_attack_action(self, actions):
    # The default is to attack the target entity, if within range
    if self.target_entity and self.can_attack(self.target_entity):
      self._add_attack_action(actions, self.target_entity)
    elif self.can_attack(self.target):
      self._add_attack_action(actions, self.target)
