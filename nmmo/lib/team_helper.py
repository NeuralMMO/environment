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

  def get_target_agent(self, team_id: int, target: str):
    team_ids = list(self.teams.keys())
    idx = team_ids.index(team_id)
    if target == "left_team":
      target_id = team_ids[(idx+1) % self.num_teams]
      return self.teams[target_id]
    if target == "left_team_leader":
      target_id = team_ids[(idx+1) % self.num_teams]
      return self.teams[target_id][0]
    if target == "right_team":
      target_id = team_ids[(idx-1) % self.num_teams]
      return self.teams[target_id]
    if target == "right_team_leader":
      target_id = team_ids[(idx-1) % self.num_teams]
      return self.teams[target_id][0]
    if target == "my_team_leader":
      return self.teams[team_id][0]
    if target == "all_foes":
      all_foes = []
      for foe_team_id in team_ids:
        if foe_team_id != team_id:
          all_foes += self.teams[foe_team_id]
      return all_foes
    return None
