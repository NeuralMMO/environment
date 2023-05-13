#pylint: disable=invalid-name, unused-argument, no-value-for-parameter
from __future__ import annotations
import numpy as np
from numpy import count_nonzero as count

from nmmo.task.predicate import OR, predicate
from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.task import constraint
from nmmo.systems import skill as nmmo_skill
from nmmo.systems.skill import Skill
from nmmo.systems.item import Item
from nmmo.lib.material import Material
from nmmo.lib import utils

@predicate
def Success(gs: GameState):
  ''' Returns True. For debugging.
  '''
  return True

@predicate
def TickGE(gs: GameState,
           num_tick: int                = constraint.ScalarConstraint()):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.
  """
  return gs.current_tick / num_tick

@predicate
def CanSeeTile(gs: GameState,
               subject: Group           = constraint.TEAM_GROUPS,
               tile_type: type[Material]= constraint.MATERIAL_CONSTRAINT):
  """ True if any agent in subject can see a tile of tile_type
  """
  return any(tile_type.index in t for t in subject.obs.tile.material_id)

@predicate
def StayAlive(gs: GameState,
              subject: Group            = constraint.TEAM_GROUPS):
  """True if all subjects are alive.
  """
  return count(subject.health > 0) == len(subject)

@predicate
def AllDead(gs: GameState,
            subject: Group              = constraint.TEAM_GROUPS):
  """True if all subjects are dead.
  """
  return 1.0 - count(subject.health) / len(subject)

@predicate
def OccupyTile(gs: GameState,
               subject: Group,
               row: int                  = constraint.COORDINATE_CONSTRAINT,
               col: int                  = constraint.COORDINATE_CONSTRAINT):
  """True if any subject agent is on the desginated tile.
  """
  return np.any((subject.row == row) & (subject.col == col))

@predicate
def AllMembersWithinRange(gs: GameState,
                          subject: Group = constraint.TEAM_GROUPS,
                          dist: int      = constraint.COORDINATE_CONSTRAINT):
  """True if the max l-inf distance of teammates is
         less than or equal to dist
  """
  current_dist = max(subject.row.max()-subject.row.min(),
      subject.col.max()-subject.col.min())
  if current_dist <= 0:
    return 1.0
  return dist / current_dist

@predicate
def CanSeeAgent(gs: GameState,
                subject: Group             = constraint.TEAM_GROUPS,
                target: int                = constraint.AGENT_NUMBER_CONSTRAINT):
  """True if obj_agent is present in the subjects' entities obs.
  """
  return any(target in e.ids for e in subject.obs.entities)

@predicate
def CanSeeGroup(gs: GameState,
                subject: Group              = constraint.TEAM_GROUPS,
                target: Group               = constraint.TEAM_GROUPS):
  """ Returns True if subject can see any of target
  """
  return OR(*(CanSeeAgent(subject, agent) for agent in target.agents))

@predicate
def DistanceTraveled(gs: GameState,
                     subject: Group         = constraint.TEAM_GROUPS,
                     dist: int              = constraint.ScalarConstraint()):
  """True if the summed l-inf distance between each agent's current pos and spawn pos
        is greater than or equal to the specified _dist.
  """
  if not any(subject.health > 0):
    return False
  r = subject.row
  c = subject.col
  dists = utils.linf(list(zip(r,c)),[gs.spawn_pos[id_] for id_ in subject.entity.id])
  return dists.sum() / dist

@predicate
def AttainSkill(gs: GameState,
                subject: Group              = constraint.TEAM_GROUPS,
                skill: Skill                = constraint.SKILL_CONSTRAINT,
                level: int                  = constraint.PROGRESSION_CONSTRAINT,
                num_agent: int              = constraint.AGENT_NUMBER_CONSTRAINT):
  """True if the number of agents having skill level GE level
        is greather than or equal to num_agent
  """
  skill_level = getattr(subject,skill.__name__.lower() + '_level')
  return sum(skill_level >= level) / num_agent

@predicate
def CountEvent(gs: GameState,
               subject: Group               = constraint.TEAM_GROUPS,
               event: str                   = constraint.EVENTCODE_CONSTRAINT,
               N: int                       = constraint.ScalarConstraint()):
  """True if the number of events occured in subject corresponding
      to event >= N
  """
  return len(getattr(subject.event, event)) / N

@predicate
def ScoreHit(gs: GameState,
             subject: Group                 = constraint.TEAM_GROUPS,
             combat_style: type[Skill]      = constraint.COMBAT_SKILL_CONSTRAINT,
             N: int                         = constraint.ScalarConstraint()):
  """True if the number of hits scored in style
  combat_style >= count
  """
  hits = subject.event.SCORE_HIT.combat_style == combat_style.SKILL_ID
  return count(hits) / N

@predicate
def HoardGold(gs: GameState,
              subject: Group                = constraint.TEAM_GROUPS,
              amount: int                   = constraint.ScalarConstraint()):
  """True iff the summed gold of all teammate is greater than or equal to amount.
  """
  return subject.gold.sum() / amount

@predicate
def EarnGold(gs: GameState,
             subject: Group                 = constraint.TEAM_GROUPS,
             amount: int                    = constraint.ScalarConstraint()):
  """ True if the total amount of gold earned is greater than or equal to amount.
  """
  return subject.event.EARN_GOLD.gold.sum() / amount

@predicate
def SpendGold(gs: GameState,
              subject: Group                = constraint.TEAM_GROUPS,
              amount: int                   = constraint.ScalarConstraint()):
  """ True if the total amount of gold spent is greater than or equal to amount.
  """
  return subject.event.BUY_ITEM.gold.sum() / amount

@predicate
def MakeProfit(gs: GameState,
              subject: Group                = constraint.TEAM_GROUPS,
              amount: int                   = constraint.ScalarConstraint()):
  """ True if the total amount of gold earned-spent is greater than or equal to amount.
  """
  profits = subject.event.EARN_GOLD.gold.sum()
  costs = subject.event.BUY_ITEM.gold.sum()
  return  (profits-costs) / amount

@predicate
def InventorySpaceGE(gs: GameState,
                     subject: Group         = constraint.TEAM_GROUPS,
                     space: int             = constraint.ScalarConstraint()):
  """True if the inventory space of every subjects is greater than or equal to
       the space. Otherwise false.
  """
  max_space = gs.config.ITEM_INVENTORY_CAPACITY
  return all(max_space - inv.len >= space for inv in subject.obs.inventory)

@predicate
def OwnItem(gs: GameState,
            subject: Group                  = constraint.TEAM_GROUPS,
            item: type[Item]                = constraint.ITEM_CONSTRAINT,
            level: int                      = constraint.PROGRESSION_CONSTRAINT,
            quantity: int                   = constraint.INVENTORY_CONSTRAINT):
  """True if the number of items owned (_item_type, >= level)
     is greater than or equal to quantity.
  """
  owned = (subject.item.type_id == item.ITEM_TYPE_ID) & \
          (subject.item.level >= level)
  return sum(subject.item.quantity[owned]) / quantity

@predicate
def EquipItem(gs: GameState,
              subject: Group                = constraint.TEAM_GROUPS,
              item: type[Item]              = constraint.ITEM_CONSTRAINT,
              level: int                    = constraint.PROGRESSION_CONSTRAINT,
              num_agent: int                = constraint.AGENT_NUMBER_CONSTRAINT):
  """True if the number of agents that equip the item (_item_type, >=_level)
     is greater than or equal to _num_agent.
  """
  equipped = (subject.item.type_id == item.ITEM_TYPE_ID) & \
             (subject.item.level >= level) & \
             (subject.item.equipped > 0)
  if num_agent > 0:
    return count(equipped) / num_agent
  return 1.0

@predicate
def FullyArmed(gs: GameState,
               subject: Group               = constraint.TEAM_GROUPS,
               combat_style: type[Skill]    = constraint.COMBAT_SKILL_CONSTRAINT,
               level: int                   = constraint.PROGRESSION_CONSTRAINT,
               num_agent: int               = constraint.AGENT_NUMBER_CONSTRAINT):
  """True if the number of fully equipped agents is greater than or equal to _num_agent
       Otherwise false.
       To determine fully equipped, we look at hat, top, bottom, weapon, ammo, respectively,
       and see whether these are equipped and has level greater than or equal to _level.
  """
  WEAPON_IDS = {
    nmmo_skill.Melee: {'weapon':5, 'ammo':13}, # Sword, Scrap
    nmmo_skill.Range: {'weapon':6, 'ammo':14}, # Bow, Shaving
    nmmo_skill.Mage: {'weapon':7, 'ammo':15} # Wand, Shard
  }
  item_ids = { 'hat':2, 'top':3, 'bottom':4 }
  item_ids.update(WEAPON_IDS[combat_style])

  lvl_flt = (subject.item.level >= level) & \
            (subject.item.equipped > 0)
  type_flt = np.isin(subject.item.type_id,list(item_ids.values()))
  _, equipment_numbers = np.unique(subject.item.owner_id[lvl_flt & type_flt],
                                   return_counts=True)
  if num_agent > 0:
    return (equipment_numbers >= len(item_ids.items())).sum() / num_agent
  return 1.0

@predicate
def ConsumeItem(gs: GameState,
                subject: Group              = constraint.TEAM_GROUPS,
                item: type[Item]            = constraint.CONSUMABLE_CONSTRAINT,
                level: int                  = constraint.PROGRESSION_CONSTRAINT,
                quantity: int               = constraint.ScalarConstraint()):
  """True if total quantity consumed of item type above level is >= quantity
  """
  type_flt = subject.event.CONSUME_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.CONSUME_ITEM.level >= level
  return subject.event.CONSUME_ITEM.number[type_flt & lvl_flt].sum() / quantity

@predicate
def HarvestItem(gs: GameState,
                subject: Group              = constraint.TEAM_GROUPS,
                item: type[Item]            = constraint.ITEM_CONSTRAINT,
                level: int                  = constraint.PROGRESSION_CONSTRAINT,
                quantity: int               = constraint.ScalarConstraint()):
  """True if total quantity harvested of item type above level is >= quantity
  """
  type_flt = subject.event.HARVEST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.HARVEST_ITEM.level >= level
  return subject.event.HARVEST_ITEM.number[type_flt & lvl_flt].sum() / quantity

@predicate
def ListItem(gs: GameState,
             subject: Group                 = constraint.TEAM_GROUPS,
             item: type[Item]               = constraint.ITEM_CONSTRAINT,
             level: int                     = constraint.PROGRESSION_CONSTRAINT,
             quantity: int                  = constraint.ScalarConstraint()):
  """True if total quantity listed of item type above level is >= quantity
  """
  type_flt = subject.event.LIST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.LIST_ITEM.level >= level
  return subject.event.LIST_ITEM.number[type_flt & lvl_flt].sum() / quantity

@predicate
def BuyItem(gs: GameState,
            subject: Group                  = constraint.TEAM_GROUPS,
            item: type[Item]                = constraint.ITEM_CONSTRAINT,
            level: int                      = constraint.PROGRESSION_CONSTRAINT,
            quantity: int                   = constraint.ScalarConstraint()):
  """True if total quantity purchased of item type above level is >= quantity
  """
  type_flt = subject.event.BUY_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.BUY_ITEM.level >= level
  return subject.event.BUY_ITEM.number[type_flt & lvl_flt].sum() / quantity
