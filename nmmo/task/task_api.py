# pylint: disable=unused-import,attribute-defined-outside-init
from typing import Callable, Iterable, Dict, List, Union, Tuple, Type
from types import FunctionType
from abc import ABC
import inspect
import numpy as np

from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.task.predicate_api import Predicate, make_predicate, arg_to_string
from nmmo.task import base_predicates as bp

class Task(ABC):
  """ A task is used to calculate rewards for agents in assignee
      based on the predicate and game state
  """
  def __init__(self,
               eval_fn: Callable,
               assignee: Union[Iterable[int], int],
               reward_multiplier = 1.0,
               embedding = None,
               spec_name: str = None,
               reward_to = None,
               tags: List[str] = None):
    if isinstance(assignee, int):
      self._assignee = (assignee,)
    else:
      assert len(assignee) > 0, "Assignee cannot be empty"
      self._assignee = tuple(set(assignee)) # dedup
    self._eval_fn = eval_fn
    self._reward_multiplier = reward_multiplier
    self._embedding = None if embedding is None else np.array(embedding, dtype=np.float16)
    # These are None if not created using TaskSpec
    self.spec_name, self.reward_to, self.tags = spec_name, reward_to, tags
    self.name = self._make_name(self.__class__.__name__,
                                eval_fn=eval_fn, assignee=self._assignee)
    self.reset()

  def reset(self):
    self._stop_eval = False
    self._last_eval_tick = None
    self._progress = 0.0
    self._completed_tick = None
    self._max_progress = 0.0
    self._positive_reward_count = 0
    self._negative_reward_count = 0

  def close(self):
    if self._stop_eval is False:
      if isinstance(self._eval_fn, Predicate):
        self._eval_fn.close()
      self._stop_eval = True

  @property
  def assignee(self) -> Tuple[int]:
    return self._assignee

  @property
  def completed(self) -> bool:
    return self._completed_tick is not None

  @property
  def progress(self) -> float:
    return self._progress

  @property
  def reward_multiplier(self) -> float:
    return self._reward_multiplier

  @property
  def reward_signal_count(self) -> int:
    return self._positive_reward_count + self._negative_reward_count

  @property
  def embedding(self):
    return self._embedding

  def set_embedding(self, embedding):
    self._embedding = embedding

  def _map_progress_to_reward(self, gs: GameState) -> float:
    """ The default reward is the diff between the old and new progress.
        Once the task is completed, no more reward is provided.

        Override this function to create a custom reward function
    """
    if self.completed:
      return 0.0

    new_progress = max(min(float(self._eval_fn(gs)),1.0),0.0)
    diff = new_progress - self._progress
    self._progress = new_progress
    if self._progress >= 1:
      self._completed_tick = gs.current_tick

    return diff

  def compute_rewards(self, gs: GameState) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    """ Environment facing API

    Returns rewards and infos for all agents in subject
    """
    reward = self._map_progress_to_reward(gs) * self._reward_multiplier
    self._last_eval_tick = gs.current_tick
    self._max_progress = max(self._max_progress, self._progress)
    self._positive_reward_count += int(reward > 0)
    self._negative_reward_count += int(reward < 0)
    rewards = {int(ent_id): reward for ent_id in self._assignee}
    infos = {int(ent_id): {"task_spec": self.spec_name,
                           "reward": reward,
                           "progress": self._progress,
                           "completed": self.completed}
             for ent_id in self._assignee}

    # NOTE: tasks do not know whether assignee agents are alive or dead
    #   so the Env must check it before filling in rewards and infos
    return rewards, infos

  def _make_name(self, class_name, **kwargs) -> str:
    name = [class_name] + \
      [f"{arg_to_string(key)}:{arg_to_string(arg)}" for key, arg in kwargs.items()]
    name = "("+"_".join(name).replace(" ", "")+")"
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

  @property
  def progress_info(self):
    return {
      "task_spec_name": self.spec_name,
      "last_eval_tick": self._last_eval_tick,
      "completed": self.completed,
      "completed_tick": self._completed_tick,
      "max_progress": self._max_progress,
      "positive_reward_count": self._positive_reward_count,
      "negative_reward_count": self._negative_reward_count,
      "reward_signal_count": self.reward_signal_count,
    }

class OngoingTask(Task):
  def _map_progress_to_reward(self, gs: GameState) -> float:
    """Keep returning the progress reward after the task is completed.
       However, this task tracks the completion status in the same manner.
    """
    self._progress = max(min(self._eval_fn(gs)*1.0,1.0),0.0)
    if self._progress >= 1 and self._completed_tick is None:
      self._completed_tick = gs.current_tick
    return self._progress

class HoldDurationTask(Task):
  def __init__(self,
               eval_fn: Callable,
               assignee: Union[Iterable[int], int],
               hold_duration: int,
               **kwargs):
    super().__init__(eval_fn, assignee, **kwargs)
    self._hold_duration = hold_duration
    self._reset_timer()

  def _reset_timer(self):
    self._timer = 0
    self._last_success_tick = 0

  def reset(self):
    super().reset()
    self._reset_timer()

  def _map_progress_to_reward(self, gs: GameState) -> float:
    # pylint: disable=attribute-defined-outside-init
    if self.completed:
      return 0.0

    curr_eval = max(min(self._eval_fn(gs)*1.0,1.0),0.0)
    if curr_eval < 1:
      self._reset_timer()
    else:
      self._timer += 1
      self._last_success_tick = gs.current_tick

    new_progress = self._timer / self._hold_duration
    diff = new_progress - self._progress
    self._progress = new_progress
    if self._progress >= 1 and self._completed_tick is None:
      self._completed_tick = gs.current_tick
      diff = 1.0  # give out the max reward when task is completed

    return diff

######################################################################

# The same task is assigned each agent in agent_list individually
#   with the agent as the predicate subject and task assignee
def make_same_task(pred_cls: Union[Type[Predicate], Callable],
                   agent_list: Iterable[int],
                   pred_kwargs=None,
                   task_cls: Type[Task]=Task,
                   task_kwargs=None) -> List[Task]:
  # if a function is provided, make it a predicate class
  if isinstance(pred_cls, FunctionType):
    pred_cls = make_predicate(pred_cls)
  if pred_kwargs is None:
    pred_kwargs = {}
  if task_kwargs is None:
    task_kwargs = {}

  task_list = []
  for agent_id in agent_list:
    predicate = pred_cls(Group(agent_id), **pred_kwargs)
    task_list.append(predicate.create_task(task_cls=task_cls, **task_kwargs))
  return task_list

def nmmo_default_task(agent_list: Iterable[int], test_mode=None) -> List[Task]:
  # (almost) no overhead in env._compute_rewards()
  if test_mode == "no_task":
    return []

  # eval function on Predicate class, but does not use Group during eval
  if test_mode == "dummy_eval_fn":
    # pylint: disable=unused-argument
    return make_same_task(lambda gs, subject: True, agent_list, task_cls=OngoingTask)

  return make_same_task(bp.TickGE, agent_list)
