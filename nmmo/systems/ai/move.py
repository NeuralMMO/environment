# pylint: disable=R0401

import random

from nmmo.io import action
from nmmo.systems.ai import utils


def random_direction():
  return random.choice(action.Direction.edges)

def random_safe(tiles, ent):
  r, c  = ent.pos
  cands = []
  if not tiles[r-1, c].void:
    cands.append(action.North)
  if not tiles[r+1, c].void:
    cands.append(action.South)
  if not tiles[r, c-1].void:
    cands.append(action.West)
  if not tiles[r, c+1].void:
    cands.append(action.East)

  return random.choice(cands)

def habitable(tiles, ent):
  r, c  = ent.pos
  cands = []
  if tiles[r-1, c].habitable:
    cands.append(action.North)
  if tiles[r+1, c].habitable:
    cands.append(action.South)
  if tiles[r, c-1].habitable:
    cands.append(action.West)
  if tiles[r, c+1].habitable:
    cands.append(action.East)

  if len(cands) == 0:
    return action.North

  return random.choice(cands)

def towards(direction):
  if direction == (-1, 0):
    return action.North
  if direction == (1, 0):
    return action.South
  if direction == (0, -1):
    return action.West
  if direction == (0, 1):
    return action.East

  return random.choice(action.Direction.edges)

def bullrush(ent, targ):
  direction = utils.directionTowards(ent, targ)
  return towards(direction)

def pathfind(tiles, ent, targ):
  direction = utils.aStar(tiles, ent.pos, targ.pos)
  return towards(direction)

def antipathfind(tiles, ent, targ):
  er, ec = ent.pos
  tr, tc = targ.pos
  goal   = (2*er - tr , 2*ec-tc)
  direction = utils.aStar(tiles, ent.pos, goal)
  return towards(direction)
