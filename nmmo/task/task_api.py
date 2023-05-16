from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import inspect
from numbers import Real
import math

from nmmo.core.config import Config
from nmmo.task.group import Group, union
from nmmo.task.game_state import GameState
from nmmo.task.constraint import Constraint, InvalidConstraint, GroupConstraint

class InvalidTaskDefinition(Exception):
  pass

class Task(ABC):
  """ A task is used to calculate rewards for agents in "assignee"
  """
  def __init__(self,
               subject: Group,
               *args,
               constraints: Optional[List[Tuple[str,Optional[Constraint]]]] = None,
               **kwargs):
    self.name = self._make_name(self.__class__.__name__, args, kwargs)

    def is_group(x):
      return isinstance(x, Group)
    self._groups: List[Group] = list(filter(is_group, args))
    self._groups = self._groups + list(filter(is_group, kwargs.values()))
    self._groups.append(subject)

    self._args = args
    self._kwargs = kwargs
    self._constraints = constraints
    self._config = None
    self._score = 0.0
    self._subject = subject

  def compute_rewards(self, gs) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    """ Environment facing API

    Returns rewards and infos for all agents in subject
    """
    reward = self(gs) - self._score
    self._score += reward
    rewards = {int(ent_id): reward for ent_id in self._subject}
    infos = {int(ent_id): {self.name: self._score}
             for ent_id in self._subject}
    return rewards, infos

  def __call__(self, gs: GameState) -> float:
    """ Calculates score

    Params:
      gs: GameState

    Returns:
      score
    """
    if not self._config == gs.config:
      # TODO(mark) should we make this explicitly called by environment
      self._reset(gs.config)
    # Update views
    for group in self._groups:
      group.update(gs)
    # Calculate score
    cache = gs.cache_result
    if self.name in cache:
      score = cache[self.name]
    else:
      score = self._evaluate(gs)
      cache[self.name] = score
    # Calculate score
    return score

  def _reset(self, config: Config):
    self._score = 0.0
    self._config = config
    if not self.check(self._config):
      raise InvalidConstraint()

  def check(self, config: Config):
    """ Checks whether the task is valid

    A satisfiable task "makes sense" given a config
    ie. Not trying to reach target off the map
    """
    if not GroupConstraint().check(config, self._subject):
      return False
    for i, (name, constraint) in enumerate(self._constraints):
      if constraint is None:
        continue
      if i < len(self._args):
        if not constraint.check(config, self._args[i]):
          return False
      elif not constraint.check(config, self._kwargs[name]):
        return False
    return True

  def sample(self, config: Config, **overload):
    """ Samples a concrete instance of a given task.
    
    Allows overloading of previous parameters.
    """
    # Sample Constraint
    nargs = [arg.sample(config) if isinstance(arg, Constraint) else arg
              for arg in self._args]
    nkwargs = {k : v.sample(config) if isinstance(v, Constraint) else v
                for k,v in self._kwargs.items()}
    # Overload
    for i, (name, _) in enumerate(self._constraints):
      if i < len(nargs):
        nkwargs[name] = overload[name]
     # Result
    return self.__class__(**nkwargs)

  @abstractmethod
  def _evaluate(self, gs: GameState) -> float:
    """ A mapping from a game state to the desirability of that state.
    """
    raise NotImplementedError

  def _make_name(self, class_name, args, kwargs) -> str:
    def arg_to_string(arg):
      if isinstance(arg, type): # class
        return arg.__name__
      if arg is None:
        return 'Any'
      return str(arg)

    name = [class_name] + \
      list(map(arg_to_string, args)) + \
      [f"{arg_to_string(key)}:{arg_to_string(arg)}" for key, arg in kwargs.items()]
    name = "("+'_'.join(name).replace(' ', '')+")"
    return name

  def __str__(self):
    return self.name

  @property
  def subject(self):
    return self._subject

  def __add__(self, other):
    return ADD(self, other)
  def __radd__(self, other):
    return ADD(self, other)
  def __mul__(self, other):
    return MUL(self, other)
  def __rmul__(self, other):
    return MUL(self, other)
  def __and__(self, other):
    return AND(self, other)
  def __or__(self, other):
    return OR(self, other)
  def __invert__(self):
    return NOT(self)

class Predicate(Task):
  """ A task with evaluate restricted to boolean values.

  True = 1.0
  False = 0.0
  """
  def __call__(self, gs: GameState) -> float:
    if not self._config == gs.config:
      self._reset(gs.config)
    # Update views
    for group in self._groups:
      group.update(gs)
    # Calculate score
    cache = gs.cache_result
    if self.name in cache:
      score = cache[self.name]
    else:
      score = max(min(self._evaluate(gs)*1,1.0),0.0)
      cache[self.name] = score
    # Calculate score
    return score

  def __and__(self, other):
    return PAND(self, other)
  def __or__(self, other):
    return POR(self, other)
  def __invert__(self):
    return PNOT(self)
  def __rshift__(self, other):
    return IMPLY(self, other)

################################################

def define_task(fn: Callable) -> type[Task]:
  """ Syntactic sugar API for defining tasks

  See examples at base_predicates.py
  """
  signature = inspect.signature(fn)
  for i, param in enumerate(signature.parameters.values()):
    if i == 0 and param.name != 'gs':
      raise InvalidTaskDefinition('First parameter must be gs: GameState')
    if i == 1 and (param.name != 'subject'):
      raise InvalidTaskDefinition("Second parameter must be subject: Group")

  class FunctionTask(Task):
    def __init__(self, *args, **kwargs) -> None:
      constraints = []
      self._signature = signature
      args = list(args)
      for i, param in enumerate(self._signature.parameters.values()):
        if i == 0:
          continue
        # Calculate list of constraints
        if isinstance(param.default, Constraint):
          constraints.append((param.name,param.default))
        else:
          constraints.append((param.name,None))
        # Insert default values from function definition
        if not param.name in kwargs and i-1 >= len(args):
          if param.default == inspect.Parameter.empty:
            args.append(param.default)
          else:
            kwargs[param.name] = param.default
      super().__init__(*args, **kwargs, constraints=constraints)
      self._args = args
      self._kwargs = kwargs
      self.name = self._make_name(fn.__name__, args, kwargs)
    def _evaluate(self, gs: GameState) -> float:
      # pylint: disable=redefined-builtin, unused-variable
      __doc = fn.__doc__
      result = fn(gs, *self._args, **self._kwargs)
      if isinstance(result, Task):
        return result(gs)
      return result

  return FunctionTask

def define_predicate(fn: Callable) -> type[Predicate]:
  T = define_task(fn)
  class FunctionPredicate(Predicate, T):
    # pylint: disable=super-init-not-called
    def __init__(self, *args, **kwargs) -> None:
      T.__init__(self, *args, **kwargs)
  return FunctionPredicate

################################################
class TaskOperator(Task):
  def __init__(self, n, *tasks: Union[Task, Real] ,subject: Group=None):
    if not n(len(tasks)):
      raise InvalidTaskDefinition(f"Need {n} arguments")
    tasks = list(tasks)
    self._subject_argument = subject
    if subject is None:
      try:
        subject = union(*[t.subject for t in filter(lambda t: isinstance(t, Task), tasks)])
      except AttributeError:
        subject = GroupConstraint()
    super().__init__(subject, *tasks)

    for i, t in enumerate(tasks):
      if isinstance(t, Real):
        tasks[i] = lambda _,v=tasks[i] : v
    self._tasks = tasks

  def check(self, config: Config) -> bool:
    return all((t.check(config) if isinstance(t, Task) else True for t in self._tasks))

  def sample(self, config: Config, cls: type[TaskOperator], **kwargs):
    subject = self._subject_argument if 'subject' not in kwargs else kwargs['subject']
    tasks = [t.sample(config, **kwargs) if isinstance(t, Task) else t(None) for t in self._tasks]
    return cls(*tasks, subject=subject)
class OR(TaskOperator):
  def __init__(self, *tasks: Union[Task, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *tasks, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return max(t(gs) for t in self._tasks)
  def sample(self, config: Config, **overload):
    return super().sample(config, OR **overload)

class AND(TaskOperator):
  def __init__(self, *tasks: Union[Task, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *tasks, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return min(t(gs) for t in self._tasks)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, AND, **kwargs)

class NOT(TaskOperator):
  def __init__(self, *tasks: Union[Task, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *tasks, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return -sum(t(gs) for t in self._tasks)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, NOT, **kwargs)

class ADD(TaskOperator):
  def __init__(self, *tasks: Union[Task, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *tasks, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return sum(t(gs) for t in self._tasks)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, ADD, **kwargs)

class MUL(TaskOperator):
  def __init__(self, *tasks: Union[Task, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *tasks, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    result = 1.0
    for t in self._tasks:
      result = result * t(gs)
    return result
  def sample(self, config: Config, **kwargs):
    return super().sample(config, MUL, **kwargs)

class POR(TaskOperator, Predicate):
  def __init__(self, *tasks: Predicate, subject: Group=None):
    super().__init__(lambda n: n>0, *tasks, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return any(t(gs) for t in self._tasks)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, POR, **kwargs)

class PAND(TaskOperator, Predicate):
  def __init__(self, *tasks: Predicate, subject: Group=None):
    super().__init__(lambda n: n>0, *tasks, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return all(t(gs) for t in self._tasks)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, PAND, **kwargs)

class PNOT(TaskOperator, Predicate):
  def __init__(self, task: Predicate, subject: Group=None):
    super().__init__(lambda n: n==1, task, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return not self._tasks[0](gs)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, PNOT, **kwargs)

class IMPLY(TaskOperator, Predicate):
  def __init__(self, p: Predicate, q: Predicate, subject: Group=None):
    super().__init__(lambda n: n==2, p,q, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    if self._tasks[0](gs):
      return self._tasks[1](gs)
    return True
  def sample(self, config: Config, **kwargs):
    return super().sample(config, IMPLY, **kwargs)

class Once(TaskOperator):
  def __init__(self, task: Task, subject: Group=None):
    super().__init__(lambda n: n==1, task, subject=subject)
    self._maximum_score = -math.inf
  def _evaluate(self, gs: GameState) -> float:
    self._maximum_score = max(self._maximum_score, self._tasks[0](gs))
    return self._maximum_score
  def sample(self, config: Config, **kwargs):
    return super().sample(config, Once, **kwargs)

class Repeat(TaskOperator):
  def __init__(self, task: Task, subject: Group=None):
    super().__init__(lambda n: n==1, task, subject=subject)
    self._current_score = 0
  def _evaluate(self, gs: GameState) -> float:
    self._current_score += self._tasks[0](gs)
    return self._current_score
  def sample(self, config: Config, **kwargs):
    return super().sample(config, Repeat, **kwargs)

# TODO(mark) should we define the remaining available operators
# such as multiply, modulo...
