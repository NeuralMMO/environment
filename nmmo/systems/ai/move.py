# pylint: disable=R0401
from nmmo.core import action
from nmmo.systems.ai import utils


def random_direction(np_random):
  return np_random.choice(action.Direction.edges)

def random_safe(map, ent, np_random):
  r, c  = ent.pos
  tiles = map.tiles
  cands = []
  if not tiles[r-1, c].void:
    cands.append(action.North)
  if not tiles[r+1, c].void:
    cands.append(action.North)
  if not tiles[r, c-1].void:
    cands.append(action.North)
  if not tiles[r, c+1].void:
    cands.append(action.North)

  return np_random.choice(cands)

def habitable(map, ent, np_random):
  r, c  = ent.pos
  tiles = map.habitable_tiles
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

def pathfind(map, ent, targ, np_random):
  direction = utils.aStar(map, ent.pos, targ.pos)
  return towards(direction, np_random)

def antipathfind(map, ent, targ, np_random):
  er, ec = ent.pos
  tr, tc = targ.pos
  goal   = (2*er - tr , 2*ec-tc)
  direction = utils.aStar(map, ent.pos, goal)
  return towards(direction, np_random)
