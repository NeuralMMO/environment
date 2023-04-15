#pylint: disable=invalid-name, unused-argument
import numpy as np
from numpy import count_nonzero as count
from nmmo.task.predicate.core import predicate
from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.systems.item import Item
from nmmo.systems import skill as Skill

@predicate
def InventorySpaceGE(gs: GameState,
                     subject: Group,
                     space: int):
  """True if the inventory space of every subjects is greater than or equal to
       the space. Otherwise false.
  """
  max_space = gs.config.ITEM_INVENTORY_CAPACITY
  return all(max_space - inv.len >= space for inv in subject.obs.inventory)

@predicate
def OwnItem(gs: GameState,
            subject: Group,
            item: Item,
            level: int,
            quantity: int):
  """True if the number of items owned (_item_type, >= level)
     is greater than or equal to quantity.
  """
  owned = (subject.item.type_id == item.ITEM_TYPE_ID) & \
          (subject.item.level >= level)
  return sum(subject.item.quantity[owned]) / quantity

@predicate
def EquipItem(gs: GameState,
              subject: Group,
              item: Item,
              level: int,
              num_agent: int):
  """True if the number of agents that equip the item (_item_type, >=_level)
     is greater than or equal to _num_agent.
  """
  equipped = (subject.item.type_id == item.ITEM_TYPE_ID) & \
             (subject.item.level >= level) & \
             (subject.item.equipped > 0)
  return count(equipped) >= num_agent

@predicate
def FullyArmed(gs: GameState,
               subject: Group,
               combat_style: Skill.CombatSkill,
               level: int,
               num_agent: int):
  """True if the number of fully equipped agents is greater than or equal to _num_agent
       Otherwise false.
       To determine fully equipped, we look at hat, top, bottom, weapon, ammo, respectively,
       and see whether these are equipped and has level greater than or equal to _level.
  """
  WEAPON_IDS = {
    Skill.Melee: {'weapon':5, 'ammo':13}, # Sword, Scrap
    Skill.Range: {'weapon':6, 'ammo':14}, # Bow, Shaving
    Skill.Mage: {'weapon':7, 'ammo':15} # Wand, Shard
  }
  item_ids = { 'hat':2, 'top':3, 'bottom':4 }
  item_ids.update(WEAPON_IDS[combat_style])

  lvl_flt = (subject.item.level >= level) & \
            (subject.item.equipped > 0)
  type_flt = np.isin(subject.item.type_id,list(item_ids.values()))
  _, equipment_numbers = np.unique(subject.item.owner_id[lvl_flt & type_flt],
                                   return_counts=True)

  return (equipment_numbers >= len(item_ids.items())).sum() >= num_agent

#######################################
# Event-log based predicates
#######################################

@predicate
def ConsumeItem(gs: GameState,
                subject: Group,
                item: Item,
                level: int,
                quantity: int):
  """True if total quantity consumed of item type above level is >= quantity
  """
  type_flt = subject.event.CONSUME_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.CONSUME_ITEM.level >= level
  return subject.event.CONSUME_ITEM.number[type_flt & lvl_flt].sum() / quantity

@predicate
def HarvestItem(gs: GameState,
                subject: Group,
                item: Item,
                level: int,
                quantity: int):
  """True if total quantity harvested of item type above level is >= quantity
  """
  type_flt = subject.event.HARVEST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.HARVEST_ITEM.level >= level
  return subject.event.HARVEST_ITEM.number[type_flt & lvl_flt].sum() / quantity

@predicate
def ListItem(gs: GameState,
             subject: Group,
             item: Item,
             level: int,
             quantity: int):
  """True if total quantity listed of item type above level is >= quantity
  """
  type_flt = subject.event.LIST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.LIST_ITEM.level >= level
  return subject.event.LIST_ITEM.number[type_flt & lvl_flt].sum() / quantity

@predicate
def BuyItem(gs: GameState,
            subject: Group,
            item: Item,
            level: int,
            quantity: int):
  """True if total quantity purchased of item type above level is >= quantity
  """
  type_flt = subject.event.BUY_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.BUY_ITEM.level >= level
  return subject.event.BUY_ITEM.number[type_flt & lvl_flt].sum() / quantity
