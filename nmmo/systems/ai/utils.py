# pylint: disable=all


import heapq
from typing import Tuple

import numpy as np

from nmmo.lib.utils import in_bounds

def validTarget(ent, targ, rng):
   if targ is None or not targ.alive or lInfty(ent.pos, targ.pos) > rng:
      return False
   return True


def validResource(ent, tile, rng):
   return tile is not None and tile.state.tex in (
      'foilage', 'water') and lInfty(ent.pos, tile.pos) <= rng


def directionTowards(ent, targ):
   sr, sc = ent.pos
   tr, tc = targ.pos

   if abs(sc - tc) > abs(sr - tr):
      direction = (0, np.sign(tc - sc))
   else:
      direction = (np.sign(tr - sr), 0)

   return direction


def closestTarget(ent, tiles, rng=1):
   sr, sc = ent.pos
   for d in range(rng+1):
      for r in range(-d, d+1):
         for e in tiles[sr+r, sc-d].entities.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr + r, sc + d].entities.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr - d, sc + r].entities.values():
            if e is not ent and validTarget(ent, e, rng): return e

         for e in tiles[sr + d, sc + r].entities.values():
            if e is not ent and validTarget(ent, e, rng): return e

def lInf(ent, targ):
   sr, sc = ent.pos
   gr, gc = targ.pos
   return abs(gr - sr) + abs(gc - sc)


def adjacentPos(pos):
   r, c = pos
   return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]


def cropTilesAround(position: Tuple[int, int], horizon: int, tiles):
   line, column = position

   return tiles[max(line - horizon, 0): min(line + horizon + 1, len(tiles)),
          max(column - horizon, 0): min(column + horizon + 1, len(tiles[0]))]

# A* Search
def l1(start, goal):
   sr, sc = start
   gr, gc = goal
   return abs(gr - sr) + abs(gc - sc)

def l2(start, goal):
   sr, sc = start
   gr, gc = goal
   return 0.5*((gr - sr)**2 + (gc - sc)**2)**0.5

#TODO: unify lInfty and lInf
def lInfty(start, goal):
   sr, sc = start
   gr, gc = goal
   return max(abs(gr - sr), abs(gc - sc))

CUTOFF = 100
def aStar(realm_map, start, goal):
   cutoff = CUTOFF
   tiles = realm_map.tiles
   if start == goal:
      return (0, 0)
   if (start,goal) in realm_map.pathfinding_cache:
      return realm_map.pathfinding_cache[(start,goal)]
   initial_goal = goal
   pq = [(0, start)]

   backtrace = {}
   cost = {start: 0}

   closestPos = start
   closestHeuristic = l1(start, goal)
   closestCost = closestHeuristic

   while pq:
      # Use approximate solution if budget exhausted
      cutoff -= 1
      if cutoff <= 0:
         if goal not in backtrace:
            goal = closestPos
         break

      priority, cur = heapq.heappop(pq)

      if cur == goal:
         break

      for nxt in adjacentPos(cur):
         if not in_bounds(*nxt, tiles.shape):
            continue

         newCost = cost[cur] + 1
         if nxt not in cost or newCost < cost[nxt]:
            cost[nxt] = newCost
            heuristic = lInfty(goal, nxt)
            priority = newCost + heuristic

            # Compute approximate solution
            if heuristic < closestHeuristic or (
                    heuristic == closestHeuristic and priority < closestCost):
               closestPos = nxt
               closestHeuristic = heuristic
               closestCost = priority

            heapq.heappush(pq, (priority, nxt))
            backtrace[nxt] = cur

   while goal in backtrace and backtrace[goal] != start:
      gr, gc = goal
      goal = backtrace[goal]
      sr, sc = goal
      realm_map.pathfinding_cache[(goal,initial_goal)] = (gr - sr, gc - sc)

   sr, sc = start
   gr, gc = goal
   realm_map.pathfinding_cache[(start,initial_goal)] = (gr - sr, gc - sc)
   return (gr - sr, gc - sc)
# End A*

# Adjacency functions
def adjacentTiles(tiles, ent):
   r, c = ent.pos


def adjacentDeltas():
   return [(-1, 0), (1, 0), (0, 1), (0, -1)]


def l1Deltas(s):
   rets = []
   for r in range(-s, s + 1):
      for c in range(-s, s + 1):
         rets.append((r, c))
   return rets


def posSum(pos1, pos2):
   return pos1[0] + pos2[0], pos1[1] + pos2[1]


def adjacentEmptyPos(env, pos):
   return [p for p in adjacentPos(pos)
           if in_bounds(*p, env.size)]


def adjacentTiles(env, pos):
   return [env.tiles[p] for p in adjacentPos(pos)
           if in_bounds(*p, env.size)]


def adjacentMats(tiles, pos):
   return [type(tiles[p].state) for p in adjacentPos(pos)
           if in_bounds(*p, tiles.shape)]


def adjacencyDelMatPairs(env, pos):
   return zip(adjacentDeltas(), adjacentMats(env.tiles, pos))
###End###
