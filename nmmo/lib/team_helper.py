from typing import Any, Dict, List
import numpy.random
from nmmo.lib import spawn


class TeamHelper:
  def __init__(self, teams: Dict[Any, List[int]], np_random=None):
    self.teams = teams
    self.num_teams = len(teams)
    self.team_list = list(teams.keys())
    self.team_size = {}
    self.team_and_position_for_agent = {}
    self.agent_for_team_and_position = {}

    for team_id, team in teams.items():
      self.team_size[team_id] = len(team)
      for position, agent_id in enumerate(team):
        self.team_and_position_for_agent[agent_id] = (team_id, position)
        self.agent_for_team_and_position[team_id, position] = agent_id

    # Left/right team order is determined by team_list, so shuffling it
    # TODO: check if this is correct
    np_random = np_random or numpy.random
    # np_random.shuffle(self.team_list)

  def agent_position(self, agent_id: int) -> int:
    return self.team_and_position_for_agent[agent_id][1]

  def agent_id(self, team_id: Any, position: int) -> int:
    return self.agent_for_team_and_position[team_id, position]

  def is_agent_in_team(self, agent_id:int , team_id: Any) -> bool:
    return agent_id in self.teams[team_id]

  def get_team_idx(self, agent_id:int) -> int:
    team_id, _ = self.team_and_position_for_agent[agent_id]
    return self.team_list.index(team_id)

  def get_target_agent(self, team_id: Any, target: str):
    idx = self.team_list.index(team_id)
    if target == "left_team":
      target_id = self.team_list[(idx+1) % self.num_teams]
      return self.teams[target_id]
    if target == "left_team_leader":
      target_id = self.team_list[(idx+1) % self.num_teams]
      return self.teams[target_id][0]
    if target == "right_team":
      target_id = self.team_list[(idx-1) % self.num_teams]
      return self.teams[target_id]
    if target == "right_team_leader":
      target_id = self.team_list[(idx-1) % self.num_teams]
      return self.teams[target_id][0]
    if target == "my_team_leader":
      return self.teams[team_id][0]
    if target == "all_foes":
      all_foes = []
      for foe_team_id in self.team_list:
        if foe_team_id != team_id:
          all_foes += self.teams[foe_team_id]
      return all_foes
    return None

class TeamLoader(spawn.SequentialLoader):
  def __init__(self, config, np_random,
               candidate_spawn_pos=None):
    assert config.TEAMS is not None, "config.TEAMS must be specified"
    self.team_helper = TeamHelper(config.TEAMS, np_random)
    # Check if the team specification is valid for spawning
    assert len(self.team_helper.team_and_position_for_agent.keys()) == config.PLAYER_N,\
      "Number of agents in config.TEAMS must be equal to config.PLAYER_N"
    for agent_id in range(1, config.PLAYER_N + 1):
      assert agent_id in self.team_helper.team_and_position_for_agent,\
        f"Agent id {agent_id} is not specified in config.TEAMS"

    super().__init__(config, np_random)
    if candidate_spawn_pos:
      self.candidate_spawn_pos = candidate_spawn_pos
    else:
      self.candidate_spawn_pos = \
        spawn_team_together(config, self.team_helper.num_teams)

  def get_spawn_position(self, agent_id):
    idx = self.team_helper.get_team_idx(agent_id)
    return self.candidate_spawn_pos[idx]

def spawn_team_together(config, num_teams):
  '''Generates spawn positions for new teams
  Agents in the same team spawn together in the same tile
  Evenly spaces teams around the square map borders
  Returns:
      list of tuple(int, int):
  position:
      The position (row, col) to spawn the given teams
  '''
  teams_per_sides = (num_teams + 3) // 4 # 1-4 -> 1, 5-8 -> 2, etc.

  sides = spawn.get_edge_tiles(config)
  assert len(sides[0]) > 4*teams_per_sides, 'Map too small for teams'

  team_spawn_positions = []
  for side in sides:
    for i in range(teams_per_sides):
      idx = int(len(side)*(i+1)/(teams_per_sides + 1))
      team_spawn_positions.append(side[idx])

  return team_spawn_positions
