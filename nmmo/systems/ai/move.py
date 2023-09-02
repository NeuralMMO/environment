# pylint: disable=cyclic-import
from nmmo.core import action
from nmmo.systems.ai import utils

DIRECTIONS = [ # row delta, col delta, action
      (-1, 0, action.North),
      (1, 0, action.South),
      (0, -1, action.West),
      (0, 1, action.East)] * 2

def habitable(realm_map, ent, np_random):
  r, c = ent.pos
  is_habitable = realm_map.habitable_tiles
  start = np_random.get_direction()
  for i in range(4):
    dr, dc, act = DIRECTIONS[start + i]
    if is_habitable[r + dr, c + dc]:
      return act

  return action.North

def towards(direction, np_random):
  if direction == (-1, 0):
    return action.North
  if direction == (1, 0):
    return action.South
  if direction == (0, -1):
    return action.West
  if direction == (0, 1):
    return action.East

  return np_random.choice(action.Direction.edges)

def bullrush(ent, targ, np_random):
  direction = utils.directionTowards(ent, targ)
  return towards(direction, np_random)

def pathfind(realm_map, ent, targ, np_random):
  direction = utils.aStar(realm_map, ent.pos, targ.pos)
  return towards(direction, np_random)

def antipathfind(realm_map, ent, targ, np_random):
  er, ec = ent.pos
  tr, tc = targ.pos
  goal   = (2*er - tr , 2*ec-tc)
  direction = utils.aStar(realm_map, ent.pos, goal)
  return towards(direction, np_random)
