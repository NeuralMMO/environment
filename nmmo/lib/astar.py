#pylint: disable=invalid-name
import heapq
from nmmo.lib.utils import in_bounds

CUTOFF = 100

def l1(start, goal):
  sr, sc = start
  gr, gc = goal
  return abs(gr - sr) + abs(gc - sc)

def adjacentPos(pos):
  r, c = pos
  return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]

def aStar(realm_map, start, goal, cutoff = CUTOFF):
  tiles = realm_map.tiles
  if start == goal:
    return (0, 0)
  if (start, goal) in realm_map.pathfinding_cache:
    return realm_map.pathfinding_cache[(start, goal)]
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
      if not in_bounds(*nxt, tiles.shape) or realm_map.habitable_tiles[nxt] == 0:
        continue

      newCost = cost[cur] + 1
      if nxt not in cost or newCost < cost[nxt]:
        cost[nxt] = newCost
        heuristic = l1(goal, nxt)
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
    realm_map.pathfinding_cache[(goal, initial_goal)] = (gr - sr, gc - sc)

  sr, sc = start
  gr, gc = goal
  realm_map.pathfinding_cache[(start, initial_goal)] = (gr - sr, gc - sc)
  return (gr - sr, gc - sc)
# End A*
