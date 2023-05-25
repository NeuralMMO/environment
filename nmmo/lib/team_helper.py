from typing import Dict, List


class TeamHelper():
  def __init__(self, teams: Dict[int, List[int]]):
    self.teams = teams
    self.num_teams = len(teams)
    self.team_size = {}
    self.team_and_position_for_agent = {}
    self.agent_for_team_and_position = {}

    for team_id, team in teams.items():
      self.team_size[team_id] = len(team)
      for position, agent_id in enumerate(team):
        self.team_and_position_for_agent[agent_id] = (team_id, position)
        self.agent_for_team_and_position[team_id, position] = agent_id

  def agent_position(self, agent_id: int) -> int:
    return self.team_and_position_for_agent[agent_id][1]

  def agent_id(self, team_id: int, position: int) -> int:
    return self.agent_for_team_and_position[team_id, position]

  def is_agent_in_team(self, agent_id:int , team_id: int) -> bool:
    return agent_id in self.teams[team_id]
