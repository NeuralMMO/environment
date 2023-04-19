from typing import List
from nmmo.task.group import Group

class TeamHelper:
  ''' Provides a mapping from ent_id to group as equivalent to the grouping
  expected by the policy
  '''

  def __init__(self, agents: List[int], num_teams: int):
    assert len(agents) % num_teams == 0
    self.team_size = len(agents) // num_teams
    self._team_to_ent, self._ent_to_team = self._map_ent_team(agents, num_teams)

  def _map_ent_team(self, agents, num_teams):
    _team_to_ent = {}
    _ent_to_team = {}
    for ent_id in agents:
      # to assigne agent 1 to team 0, and so forth
      pop_id = (ent_id - 1) % num_teams
      _ent_to_team[ent_id] = pop_id
      if pop_id in _team_to_ent:
        _team_to_ent[pop_id].append(ent_id)
      else:
        _team_to_ent[pop_id] = [ent_id]

    return _team_to_ent, _ent_to_team

  def team(self, pop_id: int) -> Group:
    assert pop_id in self._team_to_ent, "Wrong pop_id"
    return Group(self._team_to_ent[pop_id], f"Team.{pop_id}")

  def own_team(self, ent_id: int) -> Group:
    assert ent_id in self._ent_to_team, "Wrong ent_id"
    pop_id = self._ent_to_team[ent_id]
    return Group(self._team_to_ent[pop_id], f"Team.{pop_id}")

  def left_team(self, ent_id: int) -> Group:
    assert ent_id in self._ent_to_team, "Wrong ent_id"
    pop_id = (self._ent_to_team[ent_id] - 1) % len(self._team_to_ent)
    return Group(self._team_to_ent[pop_id], f"Team.{pop_id}")

  def right_team(self, ent_id: int) -> Group:
    assert ent_id in self._ent_to_team, "Wrong ent_id"
    pop_id = (self._ent_to_team[ent_id] + 1) % len(self._team_to_ent)
    return Group(self._team_to_ent[pop_id], f"Team.{pop_id}")

  def all_agents(self) -> Group:
    return Group(list(self._ent_to_team.keys()), "All")

  def all_teams(self) -> List[Group]:
    return list((Group(v,str(k)) for k,v in self._team_to_ent.items()))

  @staticmethod
  def generate_from_config(config):
    return TeamHelper(list(range(1, config.PLAYER_N+1)), len(config.PLAYERS))
