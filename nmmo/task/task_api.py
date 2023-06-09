# pylint: disable=unused-import
from typing import Callable, Iterable, Dict, List, Union, Tuple
from types import FunctionType
from abc import ABC
import inspect

from nmmo.task.group import Group
from nmmo.task.predicate_api import Predicate, make_predicate, arg_to_string
from nmmo.task import base_predicates as bp
from nmmo.lib.team_helper import TeamHelper

class Task(ABC):
  """ A task is used to calculate rewards for agents in assignee
      based on the predicate and game state
  """
  def __init__(self,
               eval_fn: Callable,
               assignee: Union[Iterable[int], int],
               reward_multiplier = 1.0):
    if isinstance(assignee, int):
      self._assignee = (assignee,)
    else:
      assert len(assignee) > 0, "Assignee cannot be empty"
      self._assignee = tuple(set(assignee)) # dedup
    self._eval_fn = eval_fn
    self._progress = 0.0
    self._completed = False
    self._reward_multiplier = reward_multiplier
    self.name = self._make_name(self.__class__.__name__,
                                eval_fn=eval_fn, assignee=self._assignee)

  def reset(self):
    self._progress = 0.0
    self._completed = False

  @property
  def assignee(self) -> Tuple[int]:
    return self._assignee

  @property
  def completed(self) -> bool:
    return self._completed

  @property
  def reward_multiplier(self) -> float:
    return self._reward_multiplier

  def _map_progress_to_reward(self, gs) -> float:
    """ The default reward is the diff between the old and new progress.
        Once the task is completed, no more reward is provided.

        Override this function to create a custom reward function
    """
    if self._completed:
      return 0.0

    new_progress = max(min(self._eval_fn(gs)*1.0,1.0),0.0)
    diff = new_progress - self._progress
    self._progress = new_progress
    if self._progress >= 1:
      self._completed = True

    return diff

  def compute_rewards(self, gs) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    """ Environment facing API

    Returns rewards and infos for all agents in subject
    """
    reward = self._map_progress_to_reward(gs) * self._reward_multiplier
    rewards = {int(ent_id): reward for ent_id in self._assignee}
    infos = {int(ent_id): {'reward': reward,
                           'progress': self._progress,
                           'completed': self._completed}
             for ent_id in self._assignee}

    # NOTE: tasks do not know whether assignee agents are alive or dead
    #   so the Env must check it before filling in rewards and infos
    return rewards, infos

  def _make_name(self, class_name, **kwargs) -> str:
    name = [class_name] + \
      [f"{arg_to_string(key)}:{arg_to_string(arg)}" for key, arg in kwargs.items()]
    name = "("+'_'.join(name).replace(' ', '')+")"
    return name

  def __str__(self):
    return self.name

  @property
  def subject(self):
    if isinstance(self._eval_fn, Predicate):
      return self._eval_fn.subject.agents
    return self.assignee

  def get_source_code(self):
    if isinstance(self._eval_fn, Predicate):
      return self._eval_fn.get_source_code()
    return inspect.getsource(self._eval_fn).strip()

  def get_signature(self):
    if isinstance(self._eval_fn, Predicate):
      return self._eval_fn.get_signature()
    signature = inspect.signature(self._eval_fn)
    return list(signature.parameters)

  @property
  def args(self):
    if isinstance(self._eval_fn, Predicate):
      return self._eval_fn.args
    # the function _eval_fn must only take gs
    return []

  @property
  def kwargs(self):
    if isinstance(self._eval_fn, Predicate):
      return self._eval_fn.kwargs
    # the function _eval_fn must only take gs
    return {}

class OngoingTask(Task):
  def _map_progress_to_reward(self, gs) -> float:
    """Keep returning the progress reward after the task is completed.
       However, this task tracks the completion status in the same manner.
    """
    self._progress = max(min(self._eval_fn(gs)*1.0,1.0),0.0)
    if self._progress >= 1:
      self._completed = True
    return self._progress


######################################################################

# The same task is assigned each agent in agent_list individually
#   with the agent as the predicate subject and task assignee
def make_same_task(predicate: Union[Predicate, Callable],
                   agent_list: Iterable[int],
                   task_cls = Task, **kwargs) -> List[Task]:
  # if a function is provided, make it a predicate class
  if isinstance(predicate, FunctionType):
    predicate = make_predicate(predicate)

  return [predicate(Group(agent_id),**kwargs).create_task(task_cls=task_cls)
          for agent_id in agent_list]

def nmmo_default_task(agent_list: Iterable[int], test_mode=None) -> List[Task]:
  # (almost) no overhead in env._compute_rewards()
  if test_mode == 'no_task':
    return []

  # eval function on Predicate class, but does not use Group during eval
  if test_mode == 'dummy_eval_fn':
    # pylint: disable=unused-argument
    return make_same_task(lambda gs, subject: True, agent_list, task_cls=OngoingTask)

  # the default is to use the predicate class
  return make_same_task(bp.StayAlive, agent_list, task_cls=OngoingTask)

######################################################################
# TODO: a lot to improve below

REWARD_TO = ['agent', 'team']
VALID_TARGET = ['left_team', 'left_team_leader',
                'right_team', 'right_team_leader',
                'my_team_leader']

def make_team_tasks(teams, task_spec) -> List[Task]:
  """
    task_spec: a list of tuples (reward_to, eval_fn, **kwargs)
    
    each tuple is assigned to the teams
  """
  tasks = []
  team_list = list(teams.keys())
  team_helper = TeamHelper(teams)
  for idx in range(min(len(team_list), len(task_spec))):
    team_id = team_list[idx]
    reward_to, pred_fn, kwargs = task_spec[team_id]

    assert reward_to in REWARD_TO, 'Wrong reward target'

    if 'task_cls' in kwargs:
      task_cls = kwargs.pop('task_cls')
    else:
      task_cls = Task

    # reserve 'target' for relative agent mapping
    if 'target' in kwargs:
      target = kwargs.pop('target')
      assert target in VALID_TARGET, 'Invalid target'
      # translate target to specific agent ids using team_helper
      target = team_helper.get_target_agent(team_id, target)
      kwargs['target'] = target

    # handle some special cases and instantiate the predicate first
    predicate = None
    if isinstance(pred_fn, FunctionType):
      # if a function is provided as a predicate
      pred_cls = make_predicate(pred_fn)

    # TODO: should create a test for these
    if pred_fn in [bp.AllDead]:
      kwargs.pop('target') # remove target
      predicate = pred_cls(Group(target), **kwargs)
    if pred_fn in [bp.StayAlive] and 'target' in kwargs:
      kwargs.pop('target') # remove target
      predicate = pred_cls(Group(target), **kwargs)

    # create the task
    if reward_to == 'team':
      assignee = team_helper.teams[team_id]
      if predicate is None:
        tasks.append(pred_cls(Group(assignee), **kwargs).create_task(task_cls=task_cls))
      else:
        # this branch is for the cases like AllDead, StayAlive
        tasks.append(predicate.create_task(assignee=assignee, task_cls=task_cls))

    elif reward_to == 'agent':
      agent_list = team_helper.teams[team_id]
      if predicate is None:
        tasks += make_same_task(pred_cls, agent_list, task_cls=task_cls, **kwargs)
      else:
        # this branch is for the cases like AllDead, StayAlive
        tasks += [predicate.create_task(assignee=agent_id, task_cls=task_cls)
                  for agent_id in agent_list]

  return tasks
