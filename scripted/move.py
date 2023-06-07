# pylint: disable=all

import numpy as np
import random

import heapq

from nmmo.core import action
from nmmo.core.observation import Observation
from nmmo.lib import material

from scripted import utils

def adjacentPos(pos):
   r, c = pos
   return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]

def inSight(dr, dc, vision):
    return (
          dr >= -vision and
          dc >= -vision and
          dr <= vision and
          dc <= vision)

def rand(config, ob, actions):
   direction                 = random.choice(action.Direction.edges)
   actions[action.Move] = {action.Direction: direction}

def towards(direction):
   if direction == (-1, 0):
      return action.North
   elif direction == (1, 0):
      return action.South
   elif direction == (0, -1):
      return action.West
   elif direction == (0, 1):
      return action.East
   else:
      return random.choice(action.Direction.edges)

def pathfind(config, ob, actions, rr, cc):
   direction = aStar(config, ob, actions, rr, cc)
   direction = towards(direction)
   actions[action.Move] = {action.Direction: direction}

def meander(config, ob, actions):
   cands = []
   if ob.tile(-1, 0).material_id in material.Habitable.indices:
      cands.append((-1, 0))
   if ob.tile(1, 0).material_id in material.Habitable.indices:
      cands.append((1, 0))
   if ob.tile(0, -1).material_id in material.Habitable.indices:
      cands.append((0, -1))
   if ob.tile(0, 1).material_id in material.Habitable.indices:
      cands.append((0, 1))
   if not cands:
      return (-1, 0)

   direction = random.choices(cands)[0]
   direction = towards(direction)
   actions[action.Move] = {action.Direction: direction}

def explore(config, ob, actions, r, c):
   vision = config.PLAYER_VISION_RADIUS
   sz     = config.MAP_SIZE

   centR, centC = sz//2, sz//2

   vR, vC = centR-r, centC-c

   mmag = max(1, abs(vR), abs(vC))
   rr   = int(np.round(vision*vR/mmag))
   cc   = int(np.round(vision*vC/mmag))
   pathfind(config, ob, actions, rr, cc)

def evade(config, ob: Observation, actions, attacker):
   agent = ob.agent()

   rr, cc = (2*agent.row - attacker.row, 2*agent.col - attacker.col)

   pathfind(config, ob, actions, rr, cc)

def forageDijkstra(config, ob: Observation, actions, food_max, water_max, cutoff=100):
   vision = config.PLAYER_VISION_RADIUS

   agent  = ob.agent()
   food = agent.food
   water = agent.water

   best      = -1000
   start     = (0, 0)
   goal      = (0, 0)

   reward    = {start: (food, water)}
   backtrace = {start: None}

   queue = [start]

   while queue:
      cutoff -= 1
      if cutoff <= 0:
         break

      cur = queue.pop(0)
      for nxt in adjacentPos(cur):
         if nxt in backtrace:
            continue

         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = tile.material_id

         if not matl in material.Habitable.indices:
            continue

         food, water = reward[cur]
         food  = max(0, food - 1)
         water = max(0, water - 1)

         if matl == material.Foilage.index:
            food = min(food+food_max//2, food_max)
         for pos in adjacentPos(nxt):
            if not inSight(*pos, vision):
               continue

            tile = ob.tile(*pos)
            matl = tile.material_id

            if matl == material.Water.index:
               water = min(water+water_max//2, water_max)
               break

         reward[nxt] = (food, water)

         total = min(food, water)
         if total > best or (
                 total == best and max(food, water) > max(reward[goal])):
            best = total
            goal = nxt

         queue.append(nxt)
         backtrace[nxt] = cur

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]
   direction = towards(goal)
   actions[action.Move] = {action.Direction: direction}

def findResource(config, ob: Observation, resource):
    vision = config.PLAYER_VISION_RADIUS

    resource_index = resource.index

    for r in range(-vision, vision+1):
        for c in range(-vision, vision+1):
            tile = ob.tile(r, c)
            material_id = tile.material_id

        if material_id == resource_index:
            return (r, c)

    return False

def gatherAStar(config, ob, actions, resource, cutoff=100):
    resource_pos = findResource(config, ob, resource)
    if not resource_pos:
        return

    rr, cc = resource_pos
    next_pos = aStar(config, ob, actions, rr, cc, cutoff=cutoff)
    if not next_pos or next_pos == (0, 0):
        return

    direction = towards(next_pos)
    actions[action.Move] = {action.Direction: direction}
    return True

def gatherBFS(config, ob: Observation, actions, resource, cutoff=100):
    vision = config.PLAYER_VISION_RADIUS

    start  = (0, 0)

    backtrace = {start: None}

    queue = [start]

    found = False
    while queue:
        cutoff -= 1
        if cutoff <= 0:
            return False

        cur = queue.pop(0)
        for nxt in adjacentPos(cur):
            if found:
                break

            if nxt in backtrace:
                continue

            if not inSight(*nxt, vision):
                continue

            tile     = ob.tile(*nxt)
            matl     = tile.material_id

            if material.Fish in resource and material.Fish.index == matl:
                found = nxt
                backtrace[nxt] = cur
                break

            if not tile.material_id in material.Habitable.indices:
                continue

            if matl in (e.index for e in resource):
                found = nxt
                backtrace[nxt] = cur
                break

            for pos in adjacentPos(nxt):
                if not inSight(*pos, vision):
                    continue

                tile = ob.tile(*pos)
                matl = tile.material_id

                if matl == material.Fish.index:
                    backtrace[nxt] = cur
                    break

            queue.append(nxt)
            backtrace[nxt] = cur

    #Ran out of tiles
    if not found:
        return False

    found_orig = found
    while found in backtrace and backtrace[found] != start:
        found = backtrace[found]

    direction = towards(found)
    actions[action.Move] = {action.Direction: direction}

    return True


def aStar(config, ob: Observation, actions, rr, cc, cutoff=100):
   vision = config.PLAYER_VISION_RADIUS

   start = (0, 0)
   goal  = (rr, cc)

   if start == goal:
      return (0, 0)

   pq = [(0, start)]

   backtrace = {}
   cost = {start: 0}

   closestPos = start
   closestHeuristic = utils.l1(start, goal)
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
         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = tile.material_id

         if not matl in material.Habitable.indices:
           continue

         #Omitted water from the original implementation. Seems key
         if matl in material.Impassible.indices:
            continue

         newCost = cost[cur] + 1
         if nxt not in cost or newCost < cost[nxt]:
            cost[nxt] = newCost
            heuristic = utils.lInfty(goal, nxt)
            priority = newCost + heuristic

            # Compute approximate solution
            if heuristic < closestHeuristic or (
                    heuristic == closestHeuristic and priority < closestCost):
               closestPos = nxt
               closestHeuristic = heuristic
               closestCost = priority

            heapq.heappush(pq, (priority, nxt))
            backtrace[nxt] = cur

   #Not needed with scuffed material list above
   #if goal not in backtrace:
   #   goal = closestPos

   goal = closestPos
   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   return goal

