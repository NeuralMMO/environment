from abc import ABC, abstractmethod
from typing import Dict, List, Union

from nmmo.task.game_state import GameState
from nmmo.task.group import Group

class Predicate(ABC):
  ''' A mapping from the state of an episode to a float in the range [0,1]
  where 1 stands for True and 0 for False
  '''

  def __init__(self, *args,name: str=None, **kwargs) -> None:
    if name is None:
      self._name = Predicate._make_name(self.__class__.__name__, args, kwargs)
    else:
      self._name = Predicate._make_name(name, args, kwargs)
    def is_group(x):
      return isinstance(x, Group)

    self._groups: List[Group] = list(filter(is_group, args))
    self._groups = self._groups + list(filter(is_group, kwargs.values()))

  def __call__(self, gs: GameState) -> float:
    for group in self._groups:
      group.update(gs)
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
      self._args = args
      self._kwargs = kwargs

    def _evaluate(self, gs: GameState):
      # pylint: disable=redefined-builtin, unused-variable
      __doc__ = fn.__doc__
      result = fn(gs, *self._args, **self._kwargs)
      if isinstance(result, Predicate):
        return result(gs)
      return result

  return FunctionPredicate

################################################
# Connectives

class AND(Predicate):
  def __init__(self, *predicates: Predicate):
    super().__init__()
    assert len(predicates) > 0
    self._predicates = predicates

    # the name is AND(task1,task2,task3)
    self._name = 'AND(' + ','.join([t.name for t in self._predicates]) + ')'

  def _evaluate(self, gs: GameState):
    """True if all _predicates are evaluated to be True.
    """
    return min(t(gs) for t in self._predicates)

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
    self._name = 'OR(' + ','.join([t.name for t in self._predicates]) + ')'

  def _evaluate(self, gs: GameState):
    """True if any of _predicates is evaluated to be True.
    """
    return max(t(gs) for t in self._predicates)

  @property
  def description(self) -> Dict:
    desc = self._desc("Disjunction")
    desc.update({ 'desc_child': ["OR"] + [t.description for t in self._predicates] })
    return desc

class NOT(Predicate):
  def __init__(self, task: Predicate):
    super().__init__()
    # pylint: disable=super-init-not-called
    self._task = task

    # the name is NOT(task)
    self._name = f'NOT({self._task.name})'

  def _evaluate(self, gs: GameState):
    """True if _task is evaluated to be False.
    """
    return 1- self._task(gs)

  @property
  def description(self) -> Dict:
    desc = self._desc("Negation")
    desc.update({ 'desc_child': ["NOT", self._task.description] })
    return desc

class IMPLY(Predicate):
  def __init__(self, p: Predicate, q: Predicate):
    super().__init__()
    self._p = p
    self._q = q

    # the name is IMPLY(p->q)
    self._name = f'IMPLY({self._p.name}->{self._q.name})'

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
