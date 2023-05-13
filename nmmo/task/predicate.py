from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import copy
import inspect

from nmmo.task.game_state import GameState
from nmmo.task.group import Group
from nmmo.task.utils import Scenario
from nmmo.task.constraint import Constraint, InvalidConstraint

class Predicate(ABC):
  ''' A mapping from the state of an episode to a float in the range [0,1]
  where 1 stands for True and 0 for False
  '''

  def __init__(self, *args, name: str=None, **kwargs) -> None:
    if name is None:
      self._name = Predicate._make_name(self.__class__.__name__, args, kwargs)
    else:
      self._name = Predicate._make_name(name, args, kwargs)
    def is_group(x):
      return isinstance(x, Group)

    self._groups: List[Group] = list(filter(is_group, args))
    self._groups = self._groups + list(filter(is_group, kwargs.values()))
    self._config = None

  def __call__(self, gs: GameState) -> float:
    # Check validity
    if not self._config == gs.config:
      self._config = gs.config
      scenario = Scenario(gs.config)
      if not self._check(scenario):
        raise InvalidConstraint()
    # Update views
    for group in self._groups:
      group.update(gs)
    # Evaluate
    return max(min(self._evaluate(gs)*1,1.0),0)

  @abstractmethod
  def _evaluate(self, gs: GameState) -> Union[float, bool]:
    """One should describe the code how evaluation is done.
       LLM might use it to produce goal embedding, which will be
       used by the RL agent to produce action.
    """
    raise NotImplementedError

  @staticmethod
  def _make_name(class_name, args, kwargs):
    def arg_to_string(arg):
      if isinstance(arg, type): # class
        return arg.__name__
      if arg is None:
        return 'Any'
      return str(arg)

    name = [class_name] + \
      list(map(arg_to_string, args)) + \
      [f"{arg_to_string(key)}:{arg_to_string(arg)}" for key, arg in kwargs.items()]
    name = '_'.join(name).replace(' ', '')

    return name

  @property
  def name(self) -> str:
    return self._name

  def _desc(self, class_type):
    return {
      "type": class_type,
      "name": self.name,
      "evaluate": self._evaluate.__doc__
    }

  @property
  def description(self) -> Dict:
    return self._desc("Predicate")

  def sample(self, scenario: Scenario) -> Predicate:
    """ Returns a concrete instance of this predicate
    confined within the limits of scenario. 

    Allows each predicate instance to act as a generator
    for more predicates. Supports partial application.
    
    See constraint.py
    """
    return copy.deepcopy(self)

  def _check(self, scenario: Scenario) -> bool:
    """ Checks whether the predicate is valid

    A valid predicate is a predicate that "makes sense" given a scenario
    ie. Not trying to reach target off the map

    Not the same as satisfiability or a tautology.
    """
    return True

  def __and__(self, other):
    return AND(self, other)
  def __or__(self, other):
    return OR(self, other)
  def __invert__(self):
    return NOT(self)
  def __rshift__(self, other):
    return IMPLY(self, other)

################################################
# Sweet syntactic sugar
def predicate(fn) -> Predicate:
  class FunctionPredicate(Predicate):
    def __init__(self, *args, **kwargs) -> None:
      super().__init__(*args, name=fn.__name__, **kwargs)
      self._signature = inspect.signature(fn)
      self._args = args
      self._kwargs = kwargs

    def _evaluate(self, gs: GameState):
      # pylint: disable=redefined-builtin, unused-variable
      __doc__ = fn.__doc__
      result = fn(gs, *self._args, **self._kwargs)
      if isinstance(result, Predicate):
        return result(gs)
      return result

    def sample(self, scenario: Scenario):
      nargs = [arg.sample(scenario) if isinstance(arg, Constraint) else arg
              for arg in self._args]
      nkwargs = {k : v.sample(scenario) if isinstance(v, Constraint) else v
                 for k,v in self._kwargs}
      return FunctionPredicate(nargs, nkwargs)

    def _check(self, scenario: Scenario):
      for i, param in enumerate(self._signature.parameters.values()):
        if isinstance(param.default, Constraint):
          if i == 0:
            continue
          if i-1 < len(self._args):
            if not param.default.check(scenario,self._args[i-1]):
              return False
          elif not param.default.check(scenario, self._kwargs[param.name]):
            return False
      return True

  return FunctionPredicate

################################################
# Connectives

class AND(Predicate):
  def __init__(self, *predicates: Predicate):
    super().__init__()
    assert len(predicates) > 0
    self._predicates = predicates

    # the name is AND(task1,task2,task3)
    self._name = 'AND(' + ','.join([p.name for p in self._predicates]) + ')'

  def _evaluate(self, gs: GameState):
    """True if all _predicates are evaluated to be True.
    """
    return min(p(gs) for p in self._predicates)

  def _check(self, scenario: Scenario) -> bool:
    return all([p._check(scenario) for p in self._predicates])

  def sample(self, scenario: Scenario):
    return AND((p.sample(scenario) for p in self._predicates))

  @property
  def description(self) -> Dict:
    desc = self._desc("Conjunction")
    desc.update({ 'desc_child': ["AND"] + [t.description for t in self._predicates] })
    return desc

class OR(Predicate):
  def __init__(self, *predicates: Predicate):
    super().__init__()
    assert len(predicates) > 0
    self._predicates = predicates

    # the name is OR(task1,task2,task3,...)
    self._name = 'OR(' + ','.join([p.name for p in self._predicates]) + ')'

  def _evaluate(self, gs: GameState):
    """True if any of _predicates is evaluated to be True.
    """
    return max(p(gs) for p in self._predicates)

  def _check(self, scenario: Scenario) -> bool:
    return all([p._check(scenario) for p in self._predicates])
  
  def sample(self, scenario: Scenario):
    return OR((p.sample(scenario) for p in self._predicates))

  @property
  def description(self) -> Dict:
    desc = self._desc("Disjunction")
    desc.update({ 'desc_child': ["OR"] + [t.description for t in self._predicates] })
    return desc

class NOT(Predicate):
  def __init__(self, p: Predicate):
    super().__init__()
    # pylint: disable=super-init-not-called
    self._p = p

    # the name is NOT(task)
    self._name = f'NOT({self._p.name})'
  
  def _check(self, scenario: Scenario) -> bool:
    return self._p._check(scenario)
  
  def sample(self, scenario: Scenario):
    return NOT(self._p.sample(scenario))

  def _evaluate(self, gs: GameState):
    """True if _task is evaluated to be False.
    """
    return 1- self._p(gs)

  @property
  def description(self) -> Dict:
    desc = self._desc("Negation")
    desc.update({ 'desc_child': ["NOT", self._p.description] })
    return desc

class IMPLY(Predicate):
  def __init__(self, p: Predicate, q: Predicate):
    super().__init__()
    self._p = p
    self._q = q

    # the name is IMPLY(p->q)
    self._name = f'IMPLY({self._p.name}->{self._q.name})'

  def _check(self, scenario: Scenario) -> bool:
    return self._p._check(scenario) and self._q._check(scenario)

  def sample(self, scenario: Scenario):
    return IMPLY(self._p.sample(scenario), self._q.sample(scenario))

  def _evaluate(self, gs: GameState):
    """False if _p is true and _q is false.
       Otherwise true."""
    if self._p(gs) == 1.0:
      return self._q(gs)
    return 1.0

  @property
  def description(self) -> Dict:
    desc = self._desc("Conditional")
    desc.update({ 'desc_child': ["IMPLY"] + [t.description for t in [self._p, self._q]] })
    return desc
