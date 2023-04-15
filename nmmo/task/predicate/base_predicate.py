#pylint: disable=invalid-name, unused-argument, no-value-for-parameter
import numpy as np
from numpy import count_nonzero as count

from nmmo.task.predicate.core import predicate, OR
from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.systems import skill as nmmo_skill
from nmmo.systems.skill import Skill
from nmmo.lib.material import Material
from nmmo.lib import utils

@predicate
def TickGE(gs: GameState,
           num_tick: int):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.
  """
  return gs.current_tick / num_tick

@predicate
def CanSeeTile(gs: GameState,
               subject: Group,
               tile_type: Material):
  """ True if any agent in subject can see a tile of tile_type
  """
  return any(tile_type.index in t for t in subject.obs.tile.material_id)

@predicate
def StayAlive(gs: GameState,
              subject: Group):
  """True if all subjects are alive.
  """
  return count(subject.health > 0) == len(subject)

@predicate
def AllDead(gs: GameState,
            subject: Group):
  """True if all subjects are dead.
  """
  return 1.0 - count(subject.health) / len(subject)

@predicate
def OccupyTile(gs: GameState,
               subject: Group,
               row: int,
               col: int):
  """True if any subject agent is on the desginated tile.
  """
  return np.any((subject.row == row) & (subject.col == col))

@predicate
def AllMembersWithinRange(gs: GameState,
                          subject: Group,
                          dist: int):
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
                subject: Group,
                target: int):
  """True if obj_agent is present in the subjects' entities obs.
  """
  return any(target in e.ids for e in subject.obs.entities)

@predicate
def CanSeeGroup(gs: GameState,
                subject: Group,
                target: Group):
  """ Returns True if subject can see any of target
  """
  return OR(*(CanSeeAgent(subject, agent) for agent in target.agents))

@predicate
def DistanceTraveled(gs: GameState,
                     subject: Group,
                     dist: int):
  """True if the summed l-inf distance between each agent's current pos and spawn pos
        is greater than or equal to the specified _dist.
  """
  r = subject.row
  c = subject.col
  dists = utils.linf(list(zip(r,c)),[gs.spawn_pos[id_] for id_ in subject.agents])
  return dists.sum() / dist

@predicate
def AttainSkill(gs: GameState,
                subject: Group,
                skill: Skill,
                level: int,
                num_agent: int):
  """True if the number of agents having skill level GE level
        is greather than or equal to num_agent
  """
  skill_level = getattr(subject,skill.__name__.lower() + '_level')
  return sum(skill_level >= level) / num_agent

#######################################
# Event-log based predicates
#######################################

@predicate
def CountEvent(gs: GameState,
               subject: Group,
               event: str,
               N: int):
  """True if the number of events occured in subject corresponding
      to event >= N
  """
  return len(getattr(subject.event, event)) / N

@predicate
def ScoreHit(gs: GameState,
             subject: Group,
             combat_style: nmmo_skill.CombatSkill,
             N: int):
  """True if the number of hits scored in style
  combat_style >= count
  """
  hits = subject.event.SCORE_HIT.combat_style == combat_style.SKILL_ID
  return count(hits) / N
