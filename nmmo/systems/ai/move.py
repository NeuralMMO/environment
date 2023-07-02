# pylint: disable=cyclic-import
from nmmo.core import action
from nmmo.systems.ai import utils

# pylint: disable=unsubscriptable-object
def random_direction(np_random):
  return action.Direction.edges[np_random.integers(0,len(action.Direction.edges))]

def random_safe(realm_map, ent, np_random):
  r, c  = ent.pos
  tiles = realm_map.tiles
  cands = []
  if not tiles[r-1, c].void:
    cands.append(action.North)
  if not tiles[r+1, c].void:
    cands.append(action.South)
  if not tiles[r, c-1].void:
    cands.append(action.West)
  if not tiles[r, c+1].void:
    cands.append(action.East)

  return np_random.choice(cands)

def habitable(realm_map, ent, np_random):
  r, c  = ent.pos
  tiles = realm_map.habitable_tiles
  direction = np_random.integers(0,4)
  if direction == 0:
    if tiles[r-1, c]:
      return action.North
    if tiles[r+1, c]:
      return action.South
    if tiles[r, c-1]:
      return action.West
    if tiles[r, c+1]:
      return action.East
  elif direction == 1:
    if tiles[r+1, c]:
      return action.South
    if tiles[r, c-1]:
      return action.West
    if tiles[r, c+1]:
      return action.East
    if tiles[r-1, c]:
      return action.North
  elif direction == 2:
    if tiles[r, c-1]:
      return action.West
    if tiles[r, c+1]:
      return action.East
    if tiles[r-1, c]:
      return action.North
    if tiles[r+1, c]:
      return action.South
  else:
    if tiles[r, c+1]:
      return action.East
    if tiles[r-1, c]:
      return action.North
    if tiles[r+1, c]:
      return action.South
    if tiles[r, c-1]:
      return action.West

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
