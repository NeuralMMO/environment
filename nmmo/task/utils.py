from __future__ import annotations

import copy
from typing import Callable, Union, Iterable, \
  Optional, List, TYPE_CHECKING
from nmmo.core.config import Config
from nmmo.task.group import Group
if TYPE_CHECKING:
  from nmmo.task.task_api import Task, PredicateTask

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

  @property
  def all_agents(self) -> Group:
    return Group(list(self._ent_to_team.keys()), "All")

  @property
  def all_teams(self) -> List[Group]:
    return list((Group(v,str(k)) for k,v in self._team_to_ent.items()))

  @staticmethod
  def generate_from_config(config):
    return TeamHelper(list(range(1, config.PLAYER_N+1)), len(config.PLAYERS))

class Scenario:
  ''' Utility class to aid in defining common tasks
  '''
  def __init__(self, config: Config):
    config = copy.deepcopy(config)
    self.team_helper = TeamHelper.generate_from_config(config)
    self.config = config
    self._tasks: List[Task] = []

  def add_task(self, task: Union[Task, List[Task]]):
    if isinstance(task, List):
      for t in task:
        self.add_task(t)
    else:
      self._tasks.append(task)

  def add_tasks_foreach(self, 
                        fn: Callable,
                        groups: Union[str,Iterable[Group]] = 'teams',
                        reward: Optional[PredicateTask] = None):
    """ Utility function to define tasks across teams/agents

    Params:
      fn: Takes in a group and returns a predicate / list of predicates
        / task / list of tasks
      groups: The groups to iterate over. 
      reward: If not, package the output of fn into reward.
    """
    if isinstance(groups, str):
      assert(groups in ['agents','teams'])

    if groups == 'agents':
      groups = self.team_helper.all_agents
    elif groups == 'teams':
      groups = self.team_helper.all_teams

    for group in groups:
      if reward is None:
        self.add_task(fn(group))
      else:
        self.add_task(reward(assignee=group, predicate=fn(group)))

  @property
  def tasks(self) -> List[Task]:
    return self._tasks
