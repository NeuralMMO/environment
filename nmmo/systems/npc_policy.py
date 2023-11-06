import numpy as np

from nmmo.core import action
from nmmo.lib import utils, astar
from nmmo.entity.entity import Entity
from nmmo.entity.entity import EntityState

EntityAttr = EntityState.State.attr_name_to_col

def is_valid_target(ent, targ, rng):
  if targ is None or not targ.alive or utils.linf_single(ent.pos, targ.pos) > rng:
    return False
  config = ent.config
  if config.NPC_SYSTEM_ENABLED and not config.NPC_ALLOW_ATTACK_OTHER_NPCS and targ.is_npc:
    return False
  return True


##########################################################
# Move-related functions

DIRECTIONS = [ # row delta, col delta, action
      (-1, 0, action.North),
      (1, 0, action.South),
      (0, -1, action.West),
      (0, 1, action.East)] * 2

def dir_to_action(direction):
  return {action.Move: {action.Direction: direction}}

def get_habitable_dir(realm_map, ent, np_random):
  r, c = ent.pos
  is_habitable = realm_map.habitable_tiles
  start = np_random.get_direction()
  for i in range(4):
    delta_r, delta_c, direction = DIRECTIONS[start + i]
    if is_habitable[r + delta_r, c + delta_c]:
      return direction
  return action.North

def delta_to_dir(delta, np_random):
  if delta == (-1, 0):
    return action.North
  if delta == (1, 0):
    return action.South
  if delta == (0, -1):
    return action.West
  if delta == (0, 1):
    return action.East
  return np_random.choice(action.Direction.edges)

# pylint: disable=protected-access
def get_dir_toward(realm, entity, goal):
  delta = astar.aStar(realm.map, entity.pos, goal)
  # TODO: do not access realm._np_random directly.
  return delta_to_dir(delta, realm._np_random)


##########################################################
# Used by NPC policies

def meander(realm, entity, guard_pos=None, guard_range=0):
  if guard_pos:
    distance = utils.linf_single(entity.pos, guard_pos)
    if distance > guard_range:
      return dir_to_action(get_dir_toward(realm, entity, guard_pos))
  return dir_to_action(get_habitable_dir(realm.map, entity, realm._np_random))

def move_toward(realm, entity, goal_pos):
  # NOTE: what if the npc gets attacked while moving?
  return dir_to_action(get_dir_toward(realm, entity, goal_pos))

def charge_toward_target(realm, entity, target):
  distance = utils.linf_single(entity.pos, target.pos)
  actions = dir_to_action(get_dir_toward(realm, entity, target.pos)) if distance > 0 else {}
  add_attack_action(realm, actions, entity, target)
  return actions

def add_attack_action(realm, actions, entity, target):
  config = realm.config
  if target is None:
    return
  if config.NPC_SYSTEM_ENABLED and not config.NPC_ALLOW_ATTACK_OTHER_NPCS and target.is_npc:
    return
  distance = utils.linf_single(entity.pos, target.pos)
  if distance > entity.skills.style.attack_range(realm.config):
    return
  actions[action.Attack] = {
    action.Style: entity.skills.style,
    action.Target: target}

def identify_closest_target(realm, entity):
  radius = realm.config.PLAYER_VISION_RADIUS
  visible_entities = Entity.Query.window(
    realm.datastore, entity.pos[0], entity.pos[1], radius)
  dist = utils.linf(visible_entities[:,EntityAttr["row"]:EntityAttr["col"]+1], entity.pos)
  entity_ids = visible_entities[:,EntityAttr["id"]]

  if realm.config.NPC_SYSTEM_ENABLED and not realm.config.NPC_ALLOW_ATTACK_OTHER_NPCS:
    dist = dist[entity_ids > 0]
    entity_ids = entity_ids[entity_ids > 0]

  if len(dist) > 1:
    closest_idx = np.argmin(dist)
    return realm.entity(entity_ids[closest_idx])
  if len(dist) == 1:
    return realm.entity(entity_ids[0])
  return None
