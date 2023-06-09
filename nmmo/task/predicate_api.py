from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Union, Iterable, Type, TYPE_CHECKING
from types import FunctionType
from abc import ABC, abstractmethod
import inspect
from numbers import Real

from nmmo.core.config import Config
from nmmo.task.group import Group, union
from nmmo.task.game_state import GameState
from nmmo.task.constraint import Constraint, InvalidConstraint, GroupConstraint

if TYPE_CHECKING:
  from nmmo.task.task_api import Task

class InvalidPredicateDefinition(Exception):
  pass

class Predicate(ABC):
  """ A mapping from a game state to bounded [0, 1] float
  """
  def __init__(self,
               subject: Group,
               *args,
               constraints: Optional[List[Tuple[str,Optional[Constraint]]]] = None,
               **kwargs):
    self.name = self._make_name(self.__class__.__name__, args, kwargs)

    self._groups: List[Group] = [x for x in list(args) + list(kwargs.values())
                                 if isinstance(x, Group)]

    self._groups.append(subject)

    self._args = args
    self._kwargs = kwargs
    self._constraints = constraints
    self._config = None
    self._subject = subject

  def __call__(self, gs: GameState) -> float:
    """ Calculates score

    Params:
      gs: GameState

    Returns:
      progress: float bounded between [0, 1], 1 is considered to be true
    """
    if not self._config == gs.config:
      # TODO(mark) should we make this explicitly called by environment
      self._reset(gs.config)
    # Update views
    for group in self._groups:
      group.update(gs)
    # Calculate score
    # cache = gs.cache_result
    if self.name in gs.cache_result:
      progress = gs.cache_result[self.name]
    else:
      progress = max(min(self._evaluate(gs)*1.0,1.0),0.0)
      gs.cache_result[self.name] = progress
    return progress

  def _reset(self, config: Config):
    self._config = config
    if not self.check(self._config):
      raise InvalidConstraint()

  def check(self, config: Config):
    """ Checks whether the predicate is valid

    A satisfiable predicate "makes sense" given a config
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
    for i, (name, _) in enumerate(self._constraints):
      if i < len(nargs):
        if name in nkwargs:
          raise InvalidPredicateDefinition("Constraints should match arguments.")
        nkwargs[name] = nargs[i]
      else:
        break

    for k, v in overload.items():
      nkwargs[k] = v
     # Result
    return self.__class__(**nkwargs)

  @abstractmethod
  def _evaluate(self, gs: GameState) -> float:
    """ A mapping from a game state to the desirability/progress of that state.
        __call__() will cap its value to [0, 1]
    """
    raise NotImplementedError

  def _make_name(self, class_name, args, kwargs) -> str:
    name = [class_name] + \
      list(map(arg_to_string, args)) + \
      [f"{arg_to_string(key)}:{arg_to_string(arg)}" for key, arg in kwargs.items()]
    name = "("+'_'.join(name).replace(' ', '')+")"
    return name

  def __str__(self):
    return self.name

  @abstractmethod
  def get_source_code(self) -> str:
    """ Returns the actual source code how the game state/progress evaluation is done.
    """
    raise NotImplementedError

  @abstractmethod
  def get_signature(self) -> List:
    """ Returns the signature of the game state/progress evaluation function.
    """
    raise NotImplementedError

  @property
  def args(self):
    return self._args

  @property
  def kwargs(self):
    return self._kwargs

  @property
  def subject(self):
    return self._subject

  def create_task(self, task_cls: Type[Task]=None,
                  assignee: Union[Iterable[int], int]=None,
                  reward_multiplier=1.0,
                  task_embedding=None) -> Task:
    """ Creates a task from this predicate"""
    if task_cls is None:
      from nmmo.task.task_api import Task
      task_cls = Task

    if assignee is None:
      # the new task is assigned to this predicate's subject
      assignee = self._subject.agents

    return task_cls(eval_fn=self, assignee=assignee, reward_multiplier=reward_multiplier,
                    embedding=task_embedding)

  def __and__(self, other):
    return AND(self, other)
  def __or__(self, other):
    return OR(self, other)
  def __invert__(self):
    return NOT(self)
  def __add__(self, other):
    return ADD(self, other)
  def __radd__(self, other):
    return ADD(self, other)
  def __sub__(self, other):
    return SUB(self, other)
  def __rsub__(self, other):
    return SUB(self, other)
  def __mul__(self, other):
    return MUL(self, other)
  def __rmul__(self, other):
    return MUL(self, other)

# _make_name helper functions
def arg_to_string(arg):
  if isinstance(arg, (type, FunctionType)): # class or function
    return arg.__name__
  if arg is None:
    return 'Any'
  return str(arg)

################################################

def make_predicate(fn: Callable) -> Type[Predicate]:
  """ Syntactic sugar API for defining predicates from function
  """
  signature = inspect.signature(fn)
  for i, param in enumerate(signature.parameters.values()):
    if i == 0 and param.name != 'gs':
      raise InvalidPredicateDefinition('First parameter must be gs: GameState')
    if i == 1 and (param.name != 'subject'):
      raise InvalidPredicateDefinition("Second parameter must be subject: Group")

  class FunctionPredicate(Predicate):
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
      return fn(gs, *self._args, **self._kwargs)
    def get_source_code(self):
      return inspect.getsource(fn).strip()
    def get_signature(self) -> List:
      return list(self._signature.parameters)

  return FunctionPredicate


################################################
class PredicateOperator(Predicate):
  def __init__(self, n, *predicates: Union[Predicate, Real], subject: Group=None):
    if not n(len(predicates)):
      raise InvalidPredicateDefinition(f"Need {n} arguments")
    predicates = list(predicates)
    self._subject_argument = subject
    if subject is None:
      try:
        subject = union(*[p.subject
                          for p in filter(lambda p: isinstance(p, Predicate), predicates)])
      except AttributeError:
        subject = GroupConstraint()
    super().__init__(subject, *predicates)

    for i, p in enumerate(predicates):
      if isinstance(p, Real):
        predicates[i] = lambda _,v=predicates[i] : v
    self._predicates = predicates

  def check(self, config: Config) -> bool:
    return all((p.check(config) if isinstance(p, Predicate)
                else True for p in self._predicates))

  def sample(self, config: Config, cls: Type[PredicateOperator], **kwargs):
    subject = self._subject_argument if 'subject' not in kwargs else kwargs['subject']
    predicates = [p.sample(config, **kwargs) if isinstance(p, Predicate)
                  else p(None) for p in self._predicates]
    return cls(*predicates, subject=subject)

  def get_source_code(self) -> str:
    # NOTE: get_source_code() of the combined predicates returns the joined str
    #   of each predicate's source code, which may NOT represent what the actual
    #   predicate is doing
    # TODO: try to generate "the source code" that matches
    #   what the actual instantiated predicate returns,
    #   which perhaps should reflect the actual agent ids, etc...
    src_list = []
    for pred in self._predicates:
      if isinstance(pred, Predicate):
        src_list.append(pred.get_source_code())
    return '\n\n'.join(src_list).strip()

  def get_signature(self):
    # TODO: try to generate the correct signature
    return []

  @property
  def args(self):
    # TODO: try to generate the correct args
    return []

  @property
  def kwargs(self):
    # NOTE: This is incorrect implementation. kwargs of the combined predicates returns
    #   all summed kwargs dict, which can OVERWRITE the values of duplicated keys
    # TODO: try to match the eval function and kwargs, which can be correctly used downstream
    # for pred in self._predicates:
    #   if isinstance(pred, Predicate):
    #     kwargs.update(pred.kwargs)
    return {}

class OR(PredicateOperator, Predicate):
  def __init__(self, *predicates: Predicate, subject: Group=None):
    super().__init__(lambda n: n>0, *predicates, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    # using max as OR for the [0,1] float
    return max(p(gs) for p in self._predicates)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, OR, **kwargs)

class AND(PredicateOperator, Predicate):
  def __init__(self, *predicates: Predicate, subject: Group=None):
    super().__init__(lambda n: n>0, *predicates, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    # using min as AND for the [0,1] float
    return min(p(gs) for p in self._predicates)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, AND, **kwargs)

class NOT(PredicateOperator, Predicate):
  def __init__(self, predicate: Predicate, subject: Group=None):
    super().__init__(lambda n: n==1, predicate, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return 1.0 - self._predicates[0](gs)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, NOT, **kwargs)

class ADD(PredicateOperator, Predicate):
  def __init__(self, *predicate: Union[Predicate, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *predicate, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return max(min(sum(p(gs) for p in self._predicates),1.0),0.0)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, ADD, **kwargs)

class SUB(PredicateOperator, Predicate):
  def __init__(self, p: Predicate, q: Union[Predicate, Real], subject: Group=None):
    super().__init__(lambda n: n==2, p,q, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return max(min(self._predicates[0](gs)-self._predicates[1](gs),1.0),0.0)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, SUB, **kwargs)

class MUL(PredicateOperator, Predicate):
  def __init__(self, *predicate: Union[Predicate, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *predicate, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    result = 1.0
    for p in self._predicates:
      result = result * p(gs)
    return max(min(result,1.0),0.0)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, MUL, **kwargs)
