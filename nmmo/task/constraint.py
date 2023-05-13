from __future__ import annotations

import random
from numbers import Number
from typing import Union, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod
from nmmo.systems import skill, item
from nmmo.lib import material

from nmmo.task.utils import Scenario
if TYPE_CHECKING:
  from nmmo.task.utils import Scenario


class InvalidConstraint(Exception):
  pass

class Constraint(ABC):
  """ To check the validity of predicates
  and assist generating new predicates. Similar to gym spaces.
  """
  def __init__(self, systems=None):
    if systems is None:
      systems = []
    self._systems = systems

  def check(self, scenario: Scenario, value):
    """ Checks value is in bounds given scenario
    """
    for system in self._systems:
      try:
        if not scenario.config.__getattribute__(system):
          return False
      except:
        return False
    return True

  @abstractmethod
  def sample(self, scenario: Scenario):
    """ Generator to sample valid values given scenario
    """
    raise NotImplementedError
   
class GroupConstraint(Constraint):
  """ Ensures that all agents of a group exist in a scenario
  """
  def __init__(self,
               sample_fn = lambda s: s.team_helper.all_teams,
               systems = None):
    """
    Params
      sample_fn: given a Scenario, return groups to select from
      systems: systems required to operate
    """
    super().__init__(systems)
    self._sample_fn = sample_fn

  def check(self, scenario, value):
    if not super().check(scenario,value):
      return False
    for agent in value.agents:
      if agent > scenario.config.PLAYER_N:
        return False
    return True

  def sample(self, scenario):
    return random.choice(self._sample_fn(scenario))

class ScalarConstraint(Constraint):
  def __init__(self,
               low:  Union[Callable, Number] = 0,
               high: Union[Callable, Number] = 1024,
               dtype = int,
               systems = None):
    super().__init__(systems)
    self._low = low
    self._high = high
    if isinstance(low, Number):
      self._low  = lambda _ : low
    if isinstance(high, Number):
      self._high = lambda _ : high
    self._dtype = dtype
    if self._dtype == int:
      self._dtype = round

  def check(self, scenario, value):
    if not super().check(scenario,value):
      return False
    if self._low(scenario) <= value < self._high(scenario):
      return True
    return False

  def sample(self, scenario):
    l, h = self._low(scenario), self._high(scenario)
    return self._dtype(random.random()*(h-l)+l)

class DiscreteConstraint(Constraint):
  def __init__(self, space, systems=None):
    super().__init__(systems)
    self._space = space

  def check(self, scenario: Scenario, value):
    if not super().check(scenario,value):
      return False
    return value in self._space

  def sample(self, scenario: Scenario):
    return random.choice(self._space)

# Group Constraints
TEAM_GROUPS = GroupConstraint()
INDIVIDUAL_GROUPS = GroupConstraint(sample_fn=lambda th: th.all_agents)

# System Constraints
MATERIAL_CONSTRAINT = DiscreteConstraint(space=list(material.All.materials),
                                         systems=['TERRAIN_SYSTEM_ENABLED',
                                                  'RESOURCE_SYSTEM_ENABLED'])
HABITABLE_CONSTRAINT = DiscreteConstraint(space=list(material.Habitable.materials),
                                         systems=['TERRAIN_SYSTEM_ENABLED'])
combat_skills = [skill.Melee, skill.Mage, skill.Range]
basic_skills = [skill.Water, skill.Food]
harvest_skills = [skill.Fishing, skill.Herbalism, skill.Prospecting, skill.Alchemy, skill.Carving]
SKILL_CONSTRAINT = DiscreteConstraint(space=combat_skills+basic_skills+harvest_skills,
                                      systems=['PROFESSION_SYSTEM_ENABLED'])
COMBAT_SKILL_CONSTRAINT = DiscreteConstraint(space=combat_skills,
                                      systems=['PROFESSION_SYSTEM_ENABLED'])
EVENTCODE_CONSTRAINT = DiscreteConstraint(space=['EAT_FOOD',
                                                 'DRINK_WATER',
                                                 'SCORE_HIT',
                                                 'PLAYER_KILL',
                                                 'CONSUME_ITEM',
                                                 'GIVE_ITEM',
                                                 'DESTROY_ITEM',
                                                 'HARVEST_ITEM',
                                                 'GIVE_GOLD',
                                                 'LIST_ITEM',
                                                 'EARN_GOLD',
                                                 'BUY_ITEM'])
armour = [item.Hat, item.Top, item.Bottom]
weapons = [item.Sword, item.Bow, item.Wand]
tools = [item.Chisel, item.Gloves, item.Rod, item.Pickaxe, item.Arcane]
ammunition = [item.Shard, item.Shaving, item.Scrap]
consumables = [item.Poultice, item.Ration]
ITEM_CONSTRAINT = DiscreteConstraint(space=armour+weapons+tools+ammunition+consumables,
                                     systems=['ITEM_SYSTEM_ENABLED'])
CONSUMABLE_CONSTRAINT = DiscreteConstraint(space=consumables,
                                           systems=['ITEM_SYSTEM_ENABLED'])
# Config Constraints
COORDINATE_CONSTRAINT = ScalarConstraint(high = lambda s: s.config.MAP_CENTER)
PROGRESSION_CONSTRAINT = ScalarConstraint(high = lambda s: s.config.PROGRESSION_LEVEL_MAX+1)
INVENTORY_CONSTRAINT = ScalarConstraint(high=lambda s: s.config.ITEM_INVENTORY_CAPACITY+1)
AGENT_NUMBER_CONSTRAINT = ScalarConstraint(low = 1, high = lambda s: s.config.PLAYER_N+1)