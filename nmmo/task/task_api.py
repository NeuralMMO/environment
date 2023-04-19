from __future__ import annotations
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

from nmmo.task.predicate import Predicate
from nmmo.task.predicate.base_predicate import StayAlive
from nmmo.task.group import Group
from nmmo.task.game_state import GameState

class Task(ABC):
  def __init__(self, assignee: Group):
    self._assignee = assignee

  def __call__(self, gs: GameState):
    return self._rewards(gs)

  @property
  def assignee(self):
    return self._assignee

  @abstractmethod
  def _rewards(self, gs: GameState) -> Tuple[Dict[int, float], Dict[int, Dict]]:
    """ Returns a mapping from ent_id to rewards and infos for all 
    entities in assignee
    """
    raise NotImplementedError

class PredicateTask(Task, ABC):
  def __init__(self,
               assignee: Group,
               predicate: Predicate):
    super().__init__(assignee)
    self._predicate = predicate

  def evaluate(self, gs: GameState) -> float:
    name = self._predicate.name
    cache = gs.cache_result
    if name not in cache:
      cache[name] = self._predicate(gs)
    return cache[name]

class Once(PredicateTask):
  def __init__(self,
               assignee: Group,
               predicate: Predicate,
               reward: float = 1):
    super().__init__(assignee, predicate)
    self._reward = reward
    self._completed = False

  def _rewards(self, gs: GameState):
    '''Each agent in assignee is rewarded self._reward the 
    first time self._predicate evaluates to true.
    '''
    rewards = {int(ent_id): 0 for ent_id in self._assignee}
    infos = {int(ent_id): {self._predicate.name: self.evaluate(gs)}
             for ent_id in self._assignee}
    if not self._completed and self.evaluate(gs) == 1.0:
      self._completed = True
      rewards = {int(ent_id): self._reward for ent_id in self._assignee}
    return rewards, infos

class Repeat(PredicateTask):
  def __init__(self,
               assignee: Group,
               predicate: Predicate,
               reward = 1):
    super().__init__(assignee, predicate)
    self._reward = reward

  def _rewards(self, gs: GameState):
    '''Each agent in assignee is rewarded self._reward the 
    whenever self._predicate evaluates to true.
    '''
    rewards = {int(ent_id): 0 for ent_id in self._assignee}
    infos = {int(ent_id): {self._predicate.name: self.evaluate(gs)}
             for ent_id in self._assignee}
    if self.evaluate(gs) == 1.0:
      rewards = {int(ent_id): self._reward for ent_id in self._assignee}
    return rewards, infos

#pylint: disable=no-value-for-parameter
def default_task(agents) -> List[Tuple[Task, float]]:
  '''Generates the default reward on env.init
  '''
  return [Repeat(Group([agent]), StayAlive(Group([agent])),1)
          for agent in agents]
