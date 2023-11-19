#pylint: disable=invalid-name, unused-argument, no-value-for-parameter
from __future__ import annotations
from typing import Iterable
import numpy as np
from numpy import count_nonzero as count

from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.systems import skill as nmmo_skill
from nmmo.systems.skill import Skill
from nmmo.systems.item import Item
from nmmo.lib.material import Material
from nmmo.lib import utils

def norm(progress):
  return max(min(progress, 1.0), 0.0)

def Success(gs: GameState, subject: Group):
  ''' Returns True. For debugging.
  '''
  return True

def TickGE(gs: GameState, subject: Group, num_tick: int = None):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.
  """
  if num_tick is None:
    num_tick = gs.config.HORIZON
  return norm(gs.current_tick / num_tick)

def CanSeeTile(gs: GameState, subject: Group, tile_type: type[Material]):
  """ True if any agent in subject can see a tile of tile_type
  """
  return any(tile_type.index in t for t in subject.obs.tile.material_id)

def StayAlive(gs: GameState, subject: Group):
  """True if all subjects are alive.
  """
  return count(subject.health > 0) == len(subject)

def AllDead(gs: GameState, subject: Group):
  """True if all subjects are dead.
  """
  return norm(1.0 - count(subject.health) / len(subject))

def CheckAgentStatus(gs: GameState, subject: Group, target: Iterable[int], status: str):
  """Check if target agents are alive or dead using the game status"""
  if isinstance(target, int):
    target = [target]
  num_agents = len(target)
  num_alive = sum(1 for agent in target if agent in gs.alive_agents)
  if status == 'alive':
    return num_alive / num_agents
  if status == 'dead':
    return (num_agents - num_alive) / num_agents
  # invalid status
  return 0.0

def OccupyTile(gs: GameState, subject: Group, row: int, col: int):
  """True if any subject agent is on the desginated tile.
  """
  return np.any((subject.row == row) & (subject.col == col))

def SeizeTile(gs: GameState, subject: Group, row: int, col: int, num_ticks: int,
              progress_bonus = 0.3):
  if not any(subject.health > 0):  # subject should be alive
    return 0.0
  target_tile = (row, col)

  # if subject seized the center tile, start counting ticks
  if target_tile in gs.seize_status and gs.seize_status[target_tile][0] in subject.agents:
    seize_duration = gs.current_tick - gs.seize_status[target_tile][1]
    return norm(progress_bonus + (1.0-progress_bonus)*seize_duration/num_ticks)

  # motivate agents to seize the target tile
  max_dist = utils.linf_single(target_tile, gs.spawn_pos[subject.agents[0]])
  r = subject.row
  c = subject.col
  # NOTE: subject can be multiple agents (e.g., team), so taking the minimum
  dists = min(utils.linf(list(zip(r,c)), target_tile))
  return norm(progress_bonus*(1.0-dists/max_dist)) if max_dist > 0 else 0.0

def SeizeCenter(gs: GameState, subject: Group, num_ticks: int,
                progress_bonus = 0.3):
  row = col = gs.config.MAP_SIZE//2  # center tile
  return SeizeTile(gs, subject, row, col, num_ticks, progress_bonus)

def SeizeQuadCenter(gs: GameState, subject: Group, num_ticks: int, quadrant: str,
                    progress_bonus = 0.3):
  center = gs.config.MAP_SIZE//2
  half_dist = gs.config.MAP_CENTER//4
  if quadrant == "first":
    row = col = center + half_dist
  elif quadrant == "second":
    row, col = center - half_dist, center + half_dist
  elif quadrant == "third":
    row = col = center - half_dist
  elif quadrant == "fourth":
    row, col = center + half_dist, center - half_dist
  else:
    raise ValueError(f"Invalid quadrant {quadrant}")
  return SeizeTile(gs, subject, row, col, num_ticks, progress_bonus)

def AllMembersWithinRange(gs: GameState, subject: Group, dist: int):
  """True if the max l-inf distance of teammates is
         less than or equal to dist
  """
  max_dist = gs.config.MAP_CENTER
  r = subject.row
  if len(r) == 0 or dist < 0:
    return 0.0
  current_dist = max(r.max()-r.min(), subject.col.max()-subject.col.min())
  if current_dist <= dist:
    return 1.0
  return (max_dist - current_dist) / (max_dist - dist)

def CanSeeAgent(gs: GameState, subject: Group, target: int):
  """True if obj_agent is present in the subjects' entities obs.
  """
  return any(target in e.ids for e in subject.obs.entities)

def CanSeeGroup(gs: GameState, subject: Group, target: Iterable[int]):
  """ Returns True if subject can see any of target
  """
  if target is None:
    return False
  return any(CanSeeAgent(gs, subject, agent) for agent in target)

def DistanceTraveled(gs: GameState, subject: Group, dist: int):
  """True if the summed l-inf distance between each agent's current pos and spawn pos
        is greater than or equal to the specified _dist.
  """
  if not any(subject.health > 0):
    return False
  r = subject.row
  c = subject.col
  dists = utils.linf(list(zip(r,c)),[gs.spawn_pos[id_] for id_ in subject.entity.id])
  return norm(dists.sum() / dist)

def AttainSkill(gs: GameState, subject: Group,
                skill: type[Skill], level: int, num_agent: int):
  """True if the number of agents having skill level GE level
        is greather than or equal to num_agent
  """
  if level <= 1:
    return 1.0
  skill_level = getattr(subject,skill.__name__.lower() + '_level') - 1  # base level is 1
  return norm(sum(skill_level) / (num_agent * (level-1)))

def GainExperience(gs: GameState, subject: Group,
                   skill: type[Skill], experience: int, num_agent: int):
  """True if the experience gained for the skill is greater than or equal to experience."""
  skill_exp = getattr(subject,skill.__name__.lower() + '_exp')
  return norm(sum(skill_exp) / (experience*num_agent))

def CountEvent(gs: GameState, subject: Group, event: str, N: int):
  """True if the number of events occured in subject corresponding
      to event >= N
  """
  return norm(len(getattr(subject.event, event)) / N)

def ScoreHit(gs: GameState, subject: Group, combat_style: type[Skill], N: int):
  """True if the number of hits scored in style
  combat_style >= count
  """
  hits = subject.event.SCORE_HIT.combat_style == combat_style.SKILL_ID
  return norm(count(hits) / N)

def DefeatEntity(gs: GameState, subject: Group, agent_type: str, level: int, num_agent: int):
  """True if the number of agents (agent_type, >= level) defeated
        is greater than or equal to num_agent
  """
  # NOTE: there is no way to tell if an agent is a teammate or an enemy
  #   so agents can get rewarded for killing their own teammates
  defeated_type = subject.event.PLAYER_KILL.target_ent > 0 if agent_type == 'player' \
                    else subject.event.PLAYER_KILL.target_ent < 0
  defeated = defeated_type & (subject.event.PLAYER_KILL.level >= level)
  if num_agent > 0:
    return norm(count(defeated) / num_agent)
  return 1.0

def HoardGold(gs: GameState, subject: Group, amount: int):
  """True iff the summed gold of all teammate is greater than or equal to amount.
  """
  return norm(subject.gold.sum() / amount)

def EarnGold(gs: GameState, subject: Group, amount: int):
  """ True if the total amount of gold earned is greater than or equal to amount.
  """
  gold = subject.event.EARN_GOLD.gold.sum() + subject.event.LOOT_GOLD.gold.sum()
  return norm(gold / amount)

def SpendGold(gs: GameState, subject: Group, amount: int):
  """ True if the total amount of gold spent is greater than or equal to amount.
  """
  return norm(subject.event.BUY_ITEM.gold.sum() / amount)

def MakeProfit(gs: GameState, subject: Group, amount: int):
  """ True if the total amount of gold earned-spent is greater than or equal to amount.
  """
  profits = subject.event.EARN_GOLD.gold.sum() + subject.event.LOOT_GOLD.gold.sum()
  costs = subject.event.BUY_ITEM.gold.sum()
  return  norm((profits-costs) / amount)

def InventorySpaceGE(gs: GameState, subject: Group, space: int):
  """True if the inventory space of every subjects is greater than or equal to
       the space. Otherwise false.
  """
  max_space = gs.config.ITEM_INVENTORY_CAPACITY
  return all(max_space - inv.len >= space for inv in subject.obs.inventory)

def OwnItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if the number of items owned (_item_type, >= level)
     is greater than or equal to quantity.
  """
  owned = (subject.item.type_id == item.ITEM_TYPE_ID) & \
          (subject.item.level >= level)
  return norm(sum(subject.item.quantity[owned]) / quantity)

def EquipItem(gs: GameState, subject: Group, item: type[Item], level: int, num_agent: int):
  """True if the number of agents that equip the item (_item_type, >=_level)
     is greater than or equal to _num_agent.
  """
  equipped = (subject.item.type_id == item.ITEM_TYPE_ID) & \
             (subject.item.level >= level) & \
             (subject.item.equipped > 0)
  if num_agent > 0:
    return norm(count(equipped) / num_agent)
  return 1.0

def FullyArmed(gs: GameState, subject: Group,
               combat_style: type[Skill], level: int, num_agent: int):
  """True if the number of fully equipped agents is greater than or equal to _num_agent
       Otherwise false.
       To determine fully equipped, we look at hat, top, bottom, weapon, ammo, respectively,
       and see whether these are equipped and has level greater than or equal to _level.
  """
  WEAPON_IDS = {
    nmmo_skill.Melee: {'weapon':5, 'ammo':13}, # Spear, Whetstone
    nmmo_skill.Range: {'weapon':6, 'ammo':14}, # Bow, Arrow
    nmmo_skill.Mage: {'weapon':7, 'ammo':15} # Wand, Runes
  }
  item_ids = { 'hat':2, 'top':3, 'bottom':4 }
  item_ids.update(WEAPON_IDS[combat_style])

  lvl_flt = (subject.item.level >= level) & \
            (subject.item.equipped > 0)
  type_flt = np.isin(subject.item.type_id,list(item_ids.values()))
  _, equipment_numbers = np.unique(subject.item.owner_id[lvl_flt & type_flt],
                                   return_counts=True)
  if num_agent > 0:
    return norm((equipment_numbers >= len(item_ids.items())).sum() / num_agent)
  return 1.0

def ConsumeItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity consumed of item type above level is >= quantity
  """
  type_flt = subject.event.CONSUME_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.CONSUME_ITEM.level >= level
  return norm(subject.event.CONSUME_ITEM.number[type_flt & lvl_flt].sum() / quantity)

def HarvestItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity harvested of item type above level is >= quantity
  """
  type_flt = subject.event.HARVEST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.HARVEST_ITEM.level >= level
  return norm(subject.event.HARVEST_ITEM.number[type_flt & lvl_flt].sum() / quantity)

def FireAmmo(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity consumed of item type above level is >= quantity
  """
  type_flt = subject.event.FIRE_AMMO.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.FIRE_AMMO.level >= level
  return norm(subject.event.FIRE_AMMO.number[type_flt & lvl_flt].sum() / quantity)

def ListItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity listed of item type above level is >= quantity
  """
  type_flt = subject.event.LIST_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.LIST_ITEM.level >= level
  return norm(subject.event.LIST_ITEM.number[type_flt & lvl_flt].sum() / quantity)

def BuyItem(gs: GameState, subject: Group, item: type[Item], level: int, quantity: int):
  """True if total quantity purchased of item type above level is >= quantity
  """
  type_flt = subject.event.BUY_ITEM.type == item.ITEM_TYPE_ID
  lvl_flt = subject.event.BUY_ITEM.level >= level
  return norm(subject.event.BUY_ITEM.number[type_flt & lvl_flt].sum() / quantity)
