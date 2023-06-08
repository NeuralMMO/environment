import numpy as np

class SequentialLoader:
  '''config.PLAYER_LOADER that spreads out agent populations'''
  def __init__(self, config):
    items = config.PLAYERS

    self.items = items
    self.idx   = -1

    self.candidate_spawn_pos = spawn_concurrent(config)

  def __iter__(self):
    return self

  def __next__(self):
    self.idx = (self.idx + 1) % len(self.items)
    return self.items[self.idx]

  # pylint: disable=unused-argument
  def get_spawn_position(self, agent_id):
    # the basic SequentialLoader just provides a random spawn position
    return self.candidate_spawn_pos.pop()

def spawn_continuous(config):
  '''Generates spawn positions for new agents

  Randomly selects spawn positions around
  the borders of the square game map

  Returns:
      tuple(int, int):

  position:
      The position (row, col) to spawn the given agent
  '''
  #Spawn at edges
  mmax = config.MAP_CENTER + config.MAP_BORDER
  mmin = config.MAP_BORDER

  var  = np.random.randint(mmin, mmax)
  fixed = np.random.choice([mmin, mmax])
  r, c = int(var), int(fixed)
  if np.random.rand() > 0.5:
    r, c = c, r
  return (r, c)

def get_edge_tiles(config):
  '''Returns a list of all edge tiles'''
  # Accounts for void borders in coord calcs
  left = config.MAP_BORDER
  right = config.MAP_CENTER + config.MAP_BORDER
  lows = config.MAP_CENTER * [left]
  highs = config.MAP_CENTER * [right]
  inc = list(range(config.MAP_BORDER, config.MAP_CENTER+config.MAP_BORDER))

  # All edge tiles in order
  sides = []
  sides.append(list(zip(lows, inc)))
  sides.append(list(zip(inc, highs)))
  sides.append(list(zip(highs, inc[::-1])))
  sides.append(list(zip(inc[::-1], lows)))

  return sides

def spawn_concurrent(config):
  '''Generates spawn positions for new agents

  Evenly spaces agents around the borders
  of the square game map, assuming the edge tiles are all habitable

  Returns:
      list of tuple(int, int):

  position:
      The position (row, col) to spawn the given agent
  '''
  team_size = config.PLAYER_TEAM_SIZE
  team_n = len(config.PLAYERS)
  teammate_sep = config.PLAYER_SPAWN_TEAMMATE_DISTANCE

  # Number of total border tiles
  total_tiles = 4 * config.MAP_CENTER

  # Number of tiles, including within-team sep, occupied by each team
  tiles_per_team = teammate_sep*(team_size-1) + team_size

  # Number of total tiles dedicated to separating teams
  buffer_tiles = 0
  if team_n > 1:
    buffer_tiles = total_tiles - tiles_per_team*team_n

  # Number of tiles between teams
  team_sep = buffer_tiles // team_n

  sides = []
  for side in get_edge_tiles(config):
    sides += side

  if team_n > 1:
    # Space across and within teams
    spawn_positions = []
    for idx in range(team_sep//2, len(sides), tiles_per_team+team_sep):
      for offset in list(range(0,  tiles_per_team, teammate_sep+1)):
        if len(spawn_positions) >= config.PLAYER_N:
          continue
        pos = sides[idx + offset]
        spawn_positions.append(pos)
  else:
    # team_n = 1: to fit 128 agents in a small map, ignore spacing and spawn randomly
    np.random.shuffle(sides)
    spawn_positions = sides[:config.PLAYER_N]

  return spawn_positions

def get_team_spawn_positions(config, num_teams):
  '''Generates spawn positions for new teams
  Agents in the same team spawn together in the same tile
  Evenly spaces teams around the square map borders

  Returns:
      list of tuple(int, int):

  position:
      The position (row, col) to spawn the given teams
  '''
  teams_per_sides = (num_teams + 3) // 4 # 1-4 -> 1, 5-8 -> 2, etc.

  sides = get_edge_tiles(config)
  assert len(sides[0]) >= 4*teams_per_sides, 'Map too small for teams'

  team_spawn_positions = []
  for side in sides:
    for i in range(teams_per_sides):
      idx = int(len(side)*(i+1)/(teams_per_sides + 1))
      team_spawn_positions.append(side[idx])

  return team_spawn_positions
