from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import inspect
from numbers import Real

from nmmo.core.config import Config
from nmmo.task.group import Group, union
from nmmo.task.game_state import GameState
from nmmo.task.constraint import Constraint, InvalidConstraint, GroupConstraint

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

    def is_group(x):
      return isinstance(x, Group)
    self._groups: List[Group] = list(filter(is_group, args))
    self._groups = self._groups + list(filter(is_group, kwargs.values()))
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
      score: float bounded between [0, 1], 1 is considered to be true
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
      score = max(min(self._evaluate(gs)*1.0,1.0),0.0)
      cache[self.name] = score
    # Calculate score
    return score

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
    """ A mapping from a game state to the desirability of that state.
        __call__() will cap its value to [0, 1]
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

  def __and__(self, other):
    return PAND(self, other)
  def __or__(self, other):
    return POR(self, other)
  def __invert__(self):
    return PNOT(self)
  def __rshift__(self, other):
    return IMPLY(self, other)
  def __add__(self, other):
    return PADD(self, other)
  def __radd__(self, other):
    return PADD(self, other)
  def __sub__(self, other):
    return PSUB(self, other)
  def __rsub__(self, other):
    return PSUB(self, other)
  def __mul__(self, other):
    return PMUL(self, other)
  def __rmul__(self, other):
    return PMUL(self, other)

################################################

def define_predicate(fn: Callable) -> type[Predicate]:
  """ Syntactic sugar API for defining predicates

  See examples at base_predicates.py
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
      # pylint: disable=redefined-builtin, unused-variable
      __doc = fn.__doc__
      result = fn(gs, *self._args, **self._kwargs)
      if isinstance(result, Predicate):
        return result(gs)
      return result

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

  def sample(self, config: Config, cls: type[PredicateOperator], **kwargs):
    subject = self._subject_argument if 'subject' not in kwargs else kwargs['subject']
    predicates = [p.sample(config, **kwargs) if isinstance(p, Predicate)
                  else p(None) for p in self._predicates]
    return cls(*predicates, subject=subject)

class POR(PredicateOperator, Predicate):
  def __init__(self, *predicates: Predicate, subject: Group=None):
    super().__init__(lambda n: n>0, *predicates, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return any(p(gs) == 1 for p in self._predicates)*1.0
  def sample(self, config: Config, **kwargs):
    return super().sample(config, POR, **kwargs)

class PAND(PredicateOperator, Predicate):
  def __init__(self, *predicates: Predicate, subject: Group=None):
    super().__init__(lambda n: n>0, *predicates, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return all(p(gs) == 1 for p in self._predicates)*1.0
  def sample(self, config: Config, **kwargs):
    return super().sample(config, PAND, **kwargs)

class PNOT(PredicateOperator, Predicate):
  def __init__(self, predicate: Predicate, subject: Group=None):
    super().__init__(lambda n: n==1, predicate, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return 1.0 - self._predicates[0](gs)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, PNOT, **kwargs)

class IMPLY(PredicateOperator, Predicate):
  def __init__(self, p: Predicate, q: Predicate, subject: Group=None):
    super().__init__(lambda n: n==2, p,q, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    if self._predicates[0](gs) == 1:
      return self._predicates[1](gs)*1.0
    return True
  def sample(self, config: Config, **kwargs):
    return super().sample(config, IMPLY, **kwargs)

class PADD(PredicateOperator, Predicate):
  def __init__(self, *predicate: Union[Predicate, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *predicate, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return max(min(sum(p(gs) for p in self._predicates),1.0),0.0)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, PADD, **kwargs)

class PSUB(PredicateOperator, Predicate):
  def __init__(self, p: Predicate, q: Union[Predicate, Real], subject: Group=None):
    super().__init__(lambda n: n==2, p,q, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    return max(min(self._predicates[0](gs)-self._predicates[1](gs),1.0),0.0)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, PSUB, **kwargs)

class PMUL(PredicateOperator, Predicate):
  def __init__(self, *predicate: Union[Predicate, Real], subject: Group=None):
    super().__init__(lambda n: n>0, *predicate, subject=subject)
  def _evaluate(self, gs: GameState) -> float:
    result = 1.0
    for p in self._predicates:
      result = result * p(gs)
    return max(min(result,1.0),0.0)
  def sample(self, config: Config, **kwargs):
    return super().sample(config, PMUL, **kwargs)
