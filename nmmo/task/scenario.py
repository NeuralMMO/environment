from __future__ import annotations

import copy
from typing import Callable, Union, Iterable, \
  Optional, Dict, List
from nmmo.core.config import Config
from nmmo.task.group import Group
from nmmo.task.team_helper import TeamHelper
from nmmo.task.task_api import Task, PredicateTask, Once
from nmmo.task.predicate import Predicate

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

  def add_tasks(self,
                tasks: Union[Predicate,
                             Iterable[Task],
                             Callable[[Group], Task]],
                groups: Optional[Union[str,Iterable[Group]]] = 'teams',
                reward: Optional[type[PredicateTask]] = Once,
                reward_options: Optional[Dict] = None) -> None:
    # pylint: disable=unnecessary-lambda-assignment
    """ Utility function to define symmetric tasks

    Params:

      tasks:
        Iterable[Task]:
          For each Task in the iterable, add to scenario.
        Callable[[Group], Task]:
          A function taking in a group and return a task.
          The result from applying this function to "groups" is added to 
          the scenario.
        Predicate: 
          Mapped to Callable by return an instance of "reward".
          If exists, overrides the "subject" argument of the predicate subjects in "groups".

      groups: 
        Foreach group in groups, add a task.

      reward:
        The class used to automatically generate tasks from predicates.
      reward_options:
        extra arguments passed to reward on construction.
    """
    # Tasks
    if isinstance(tasks, Iterable):
      for task in tasks:
        self.add_task(task)
      return

    # Functional Syntax
      # Tasks
    if isinstance(tasks, Predicate):
      if reward_options is None:
        reward_options = {}
      task_generator = lambda group: reward(assignee=group,
                                            predicate=tasks.sample(cfg=self, subject=group),
                                            **reward_options)
    else:
      task_generator = tasks
      # Groups
    if isinstance(groups, str):
      assert(groups in ['agents','teams'])
    if groups == 'agents':
      groups = self.team_helper.all_agents
    elif groups == 'teams':
      groups = self.team_helper.all_teams
      # Create
    self.add_tasks([task_generator(group) for group in groups])

  @property
  def tasks(self) -> List[Task]:
    return self._tasks
