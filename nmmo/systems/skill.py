from __future__ import annotations

import abc

import numpy as np
from ordered_set import OrderedSet

from nmmo.lib import material
from nmmo.systems import combat, experience
from nmmo.lib.log import EventCode

### Infrastructure ###
class SkillGroup:
  def __init__(self, realm, entity):
    self.config  = realm.config
    self.realm   = realm
    self.entity = entity

    self.experience_calculator = experience.ExperienceCalculator()
    self.skills  = OrderedSet() # critical for determinism

  def update(self):
    for skill in self.skills:
      skill.update()

  def packet(self):
    data = {}
    for skill in self.skills:
      data[skill.__class__.__name__.lower()] = skill.packet()

    return data

class Skill(abc.ABC):
  def __init__(self, skill_group: SkillGroup):
    self.realm = skill_group.realm
    self.config = skill_group.config
    self.entity = skill_group.entity

    self.experience_calculator = skill_group.experience_calculator
    self.skill_group = skill_group
    self.exp = 0

    skill_group.skills.add(self)

  def packet(self):
    data = {}

    data['exp']   = self.exp
    data['level'] = self.level.val

    return data

  def add_xp(self, xp):
    self.exp += xp * self.config.PROGRESSION_BASE_XP_SCALE
    new_level = int(self.experience_calculator.level_at_exp(self.exp))

    if new_level > self.level.val:
      self.level.update(new_level)
      self.realm.event_log.record(EventCode.LEVEL_UP, self.entity,
                                  skill=self, level=new_level)

      self.realm.log_milestone(f'Level_{self.__class__.__name__}', new_level,
        f"PROGRESSION: Reached level {new_level} {self.__class__.__name__}",
        tags={"player_id": self.entity.ent_id})

  def set_experience_by_level(self, level):
    self.exp = self.experience_calculator.level_at_exp(level)
    self.level.update(int(level))

  @property
  def level(self):
    raise NotImplementedError(f"Skill {self.__class__.__name__} "\
      "does not implement 'level' property")

### Skill Bases ###
class CombatSkill(Skill):
  def update(self):
    pass

class NonCombatSkill(Skill):
  def __init__(self, skill_group: SkillGroup):
    super().__init__(skill_group)
    self._level = DummyLevel()

  @property
  def level(self):
    return self._level

class HarvestSkill(NonCombatSkill):
  def process_drops(self, matl, drop_table):
    if not self.config.ITEM_SYSTEM_ENABLED:
      return

    entity = self.entity

    # harvest without tool will only yield level-1 item even with high skill level
    # for example, fishing level=5 without rod will only yield level-1 ration
    level = 1
    tool  = entity.equipment.held
    if matl.tool is not None and isinstance(tool, matl.tool):
      level = tool.level.val

    #TODO: double-check drop table quantity
    for drop in drop_table.roll(self.realm, level):
      assert drop.level.val == level, 'Drop level does not match roll specification'

      self.realm.log_milestone(f'Gather_{drop.__class__.__name__}',
        level, f"PROFESSION: Gathered level {level} {drop.__class__.__name__} "
        f"(level {self.level.val} {self.__class__.__name__})",
        tags={"player_id": entity.ent_id})

      if entity.inventory.space:
        entity.inventory.receive(drop)
        self.realm.event_log.record(EventCode.HARVEST_ITEM, entity, item=drop)

  def harvest(self, matl, deplete=True):
    entity = self.entity
    realm  = self.realm

    r, c = entity.pos
    if realm.map.tiles[r, c].state != matl:
      return False

    drop_table = realm.map.harvest(r, c, deplete)
    if drop_table:
      self.process_drops(matl, drop_table)

    return drop_table

  def harvest_adjacent(self, matl, deplete=True):
    entity = self.entity
    realm  = self.realm

    r, c      = entity.pos
    drop_table = None

    if realm.map.tiles[r-1, c].state == matl:
      drop_table = realm.map.harvest(r-1, c, deplete)
    if realm.map.tiles[r+1, c].state == matl:
      drop_table = realm.map.harvest(r+1, c, deplete)
    if realm.map.tiles[r, c-1].state == matl:
      drop_table = realm.map.harvest(r, c-1, deplete)
    if realm.map.tiles[r, c+1].state == matl:
      drop_table = realm.map.harvest(r, c+1, deplete)

    if drop_table:
      self.process_drops(matl, drop_table)

    return drop_table

class AmmunitionSkill(HarvestSkill):
  def process_drops(self, matl, drop_table):
    super().process_drops(matl, drop_table)
    if self.config.PROGRESSION_SYSTEM_ENABLED:
      self.add_xp(self.config.PROGRESSION_AMMUNITION_XP_SCALE)


class ConsumableSkill(HarvestSkill):
  def process_drops(self, matl, drop_table):
    super().process_drops(matl, drop_table)
    if self.config.PROGRESSION_SYSTEM_ENABLED:
      self.add_xp(self.config.PROGRESSION_CONSUMABLE_XP_SCALE)


### Skill groups ###
class Basic(SkillGroup):
  def __init__(self, realm, entity):
    super().__init__(realm, entity)

    self.water = Water(self)
    self.food  = Food(self)

  @property
  def basic_level(self):
    return 0.5 * (self.water.level
            + self.food.level)

class Harvest(SkillGroup):
  def __init__(self, realm, entity):
    super().__init__(realm, entity)

    self.fishing      = Fishing(self)
    self.herbalism    = Herbalism(self)
    self.prospecting  = Prospecting(self)
    self.carving      = Carving(self)
    self.alchemy      = Alchemy(self)

  @property
  def harvest_level(self):
    return max(self.fishing.level,
                self.herbalism.level,
                self.prospecting.level,
                self.carving.level,
                self.alchemy.level)

class Combat(SkillGroup):
  def __init__(self, realm, entity):
    super().__init__(realm, entity)

    self.melee = Melee(self)
    self.range = Range(self)
    self.mage  = Mage(self)

  def packet(self):
    data          = super().packet()
    data['level'] = combat.level(self)

    return data

  @property
  def combat_level(self):
    return max(self.melee.level,
                self.range.level,
                self.mage.level)

  def apply_damage(self, style):
    if self.config.PROGRESSION_SYSTEM_ENABLED:
      skill  = self.__dict__[style]
      skill.add_xp(self.config.PROGRESSION_COMBAT_XP_SCALE)

  def receive_damage(self, dmg):
    pass

class Skills(Basic, Harvest, Combat):
  pass

### Skills ###
class Melee(CombatSkill):
  SKILL_ID = 1

  @property
  def level(self):
    return self.entity.melee_level

class Range(CombatSkill):
  SKILL_ID = 2

  @property
  def level(self):
    return self.entity.range_level

class Mage(CombatSkill):
  SKILL_ID = 3

  @property
  def level(self):
    return self.entity.mage_level

Melee.weakness = Mage
Range.weakness = Melee
Mage.weakness  = Range

### Individual Skills ###

class DummyLevel:
  def __init__(self, val=0):
    self.val = val

  def update(self, val):
    self.val = val

class Water(HarvestSkill):
  def update(self):
    config = self.config
    if not config.RESOURCE_SYSTEM_ENABLED:
      return

    if config.IMMORTAL:
      return

    depletion = config.RESOURCE_DEPLETION_RATE
    water = self.entity.resources.water
    water.decrement(depletion)

    if not self.harvest_adjacent(material.Water, deplete=False):
      return

    restore = np.floor(config.RESOURCE_BASE
                      * config.RESOURCE_HARVEST_RESTORE_FRACTION)
    water.increment(restore)

    self.realm.event_log.record(EventCode.DRINK_WATER, self.entity)


class Food(HarvestSkill):
  def update(self):
    config = self.config
    if not config.RESOURCE_SYSTEM_ENABLED:
      return

    if config.IMMORTAL:
      return

    depletion = config.RESOURCE_DEPLETION_RATE
    food = self.entity.resources.food
    food.decrement(depletion)

    if not self.harvest(material.Foilage):
      return

    restore = np.floor(config.RESOURCE_BASE
                      * config.RESOURCE_HARVEST_RESTORE_FRACTION)
    food.increment(restore)

    self.realm.event_log.record(EventCode.EAT_FOOD, self.entity)


class Fishing(ConsumableSkill):
  SKILL_ID = 4

  @property
  def level(self):
    return self.entity.fishing_level

  def update(self):
    self.harvest_adjacent(material.Fish)

class Herbalism(ConsumableSkill):
  SKILL_ID = 5

  @property
  def level(self):
    return self.entity.herbalism_level

  def update(self):
    self.harvest(material.Herb)

class Prospecting(AmmunitionSkill):
  SKILL_ID = 6

  @property
  def level(self):
    return self.entity.prospecting_level

  def update(self):
    self.harvest(material.Ore)

class Carving(AmmunitionSkill):
  SKILL_ID = 7

  @property
  def level(self):
    return self.entity.carving_level

  def update(self,):
    self.harvest(material.Tree)

class Alchemy(AmmunitionSkill):
  SKILL_ID = 8

  @property
  def level(self):
    return self.entity.alchemy_level

  def update(self):
    self.harvest(material.Crystal)
