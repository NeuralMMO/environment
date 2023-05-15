from __future__ import annotations

import copy
from typing import Callable, Union, Iterable, \
  Optional, List, Tuple
from nmmo.core.config import Config
from nmmo.task.group import Group
from nmmo.task.team_helper import TeamHelper
from nmmo.task.task_api import Task, Repeat
from nmmo.task.base_predicates import StayAlive

class Scenario:
  ''' Utility class to aid in defining common tasks
  '''
  def __init__(self, config: Config):
    config = copy.deepcopy(config)
    self.team_helper = TeamHelper.generate_from_config(config)
    self.config = config
    self._tasks: List[Task] = []

  def add_task(self, task: Task):
    self._tasks.append(task)

  def add_tasks(self,
                tasks: Union[Task,
                             Iterable[Task],
                             Callable[[Group], Task]],
                groups: Optional[Union[str,Iterable[Group]]] = 'teams') -> None:
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
        Task: 
          Mapped to Callable by overriding subject

      groups: 
        Foreach group in groups, add a task.
    """
    # Tasks
    if isinstance(tasks, Iterable):
      for task in tasks:
        self.add_task(task)
      return

    # Functional Syntax
      # Tasks
    if isinstance(tasks, Task):
      task_generator = lambda group: tasks.sample(config=self.config, subject=group)
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

def default_task(agents) -> List[Tuple[Task, float]]:
  '''Generates the default reward on env.init
  '''
  return [Repeat(StayAlive(Group([agent]),1)) for agent in agents]
