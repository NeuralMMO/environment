from __future__ import annotations
from typing import Dict, Iterable, Tuple, MutableMapping, Set, List
from dataclasses import dataclass, field
from copy import deepcopy
from collections import defaultdict
import weakref

from abc import ABC, abstractmethod
import functools
import numpy as np

from nmmo.core.config import Config
from nmmo.core.realm import Realm
from nmmo.core.observation import Observation
from nmmo.task.group import Group
from nmmo.entity.entity import EntityState
from nmmo.lib.event_log import EventState, ATTACK_COL_MAP, ITEM_COL_MAP, LEVEL_COL_MAP
from nmmo.lib.event_code import EventCode
from nmmo.systems.item import ItemState
from nmmo.core.tile import TileState

EntityAttr = EntityState.State.attr_name_to_col
EntityAttrKeys = EntityAttr.keys()
EventAttr = EventState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col
EventAttr.update(ITEM_COL_MAP)
EventAttr.update(ATTACK_COL_MAP)
EventAttr.update(LEVEL_COL_MAP)

@dataclass(frozen=True) # make gs read-only, except cache_result
class GameState:
  current_tick: int
  config: Config
  spawn_pos: Dict[int, Tuple[int, int]] # ent_id: (row, col) of all spawned agents

  alive_agents: Set[int] # of alive agents' ent_id (for convenience)
  env_obs: Dict[int, Observation] # env passes the obs of only alive agents

  entity_data: np.ndarray # a copied, whole Entity ds table
  entity_index: Dict[int, Iterable] # precomputed index for where_in_1d
  item_data: np.ndarray # a copied, whole Item ds table
  item_index: Dict[int, Iterable]
  event_data: np.ndarray # a copied, whole Event log table
  event_index: Dict[int, Iterable]

  # status of the seize target tiles (row, col) -> (ent_id, tick)
  seize_status: Dict[Tuple[int, int], Tuple[int, int]]

  cache_result: MutableMapping # cache for general memoization
  _group_view: List[GroupView] = field(default_factory=list) # cache for GroupView

  # add helper functions below
  @functools.lru_cache
  def entity_or_none(self, ent_id):
    if ent_id not in self.entity_index:
      return None
    return EntityState.parse_array(self.entity_data[self.entity_index[ent_id]][0])

  def where_in_id(self, data_type, subject: Iterable[int]):
    k = (data_type, subject)
    if k in self.cache_result:
      return self.cache_result[k]

    if data_type == 'entity':
      flt_idx = [row for sbj in subject for row in self.entity_index.get(sbj,[])]
      self.cache_result[k] = self.entity_data[flt_idx]
    if data_type == 'item':
      flt_idx = [row for sbj in subject for row in self.item_index.get(sbj,[])]
      self.cache_result[k] = self.item_data[flt_idx]
    if data_type == 'event':
      flt_idx = [row for sbj in subject for row in self.event_index.get(sbj,[])]
      self.cache_result[k] = self.event_data[flt_idx]
    if data_type in ['entity', 'item', 'event']:
      return self.cache_result[k]

    raise ValueError("data_type must be in entity, item, event")

  def get_subject_view(self, subject: Group):
    new_group_view = GroupView(self, subject)
    self._group_view.append(new_group_view)
    return new_group_view

  def clear_cache(self):
    # clear the cache, so that this object can be garbage collected
    self.entity_or_none.cache_clear()  # pylint: disable=no-member
    self.cache_result.clear()
    self.alive_agents.clear()
    while self._group_view:
      weakref.ref(self._group_view.pop())  # clear the cache

# Wrapper around an iterable datastore
class CachedProperty:
  def __init__(self, func):
    self.func = func
    # Allows the instance keys to be garbage collected
    #   when they are no longer referenced elsewhere
    self.cache = weakref.WeakKeyDictionary()

  def __get__(self, instance, owner):
    if instance is None:
      return self
    if instance not in self.cache:
      self.cache[instance] = self.func(instance)
    return self.cache[instance]

class ArrayView(ABC):
  def __init__(self,
               mapping,
               name: str,
               gs: GameState,
               subject: Group,
               arr: np.ndarray):
    self._mapping = mapping
    self._name = name
    self._gs = gs
    self._subject = subject
    self._hash = hash(subject) ^ hash(name)
    self._arr = arr
    self._cache = self._gs.cache_result

  def __len__(self):
    return len(self._arr)

  @abstractmethod
  def get_attribute(self, attr) -> np.ndarray:
    raise NotImplementedError

  def __getattr__(self, attr) -> np.ndarray:
    k = (self._hash, attr)
    if k in self._cache:
      return self._cache[k]
    v = object.__getattribute__(self, 'get_attribute')(attr)
    self._cache[k] = v
    return v

class ItemView(ArrayView):
  def __init__(self, gs: GameState, subject: Group, arr: np.ndarray):
    super().__init__(ItemAttr, 'item', gs, subject, arr)
    self._mapping = ItemAttr

  def get_attribute(self, attr) -> np.ndarray:
    return self._arr[:, self._mapping[attr]]

class EntityView(ArrayView):
  def __init__(self, gs: GameState, subject: Group, arr: np.ndarray):
    super().__init__(EntityAttr, 'entity', gs, subject, arr)

  def get_attribute(self, attr) -> np.ndarray:
    return self._arr[:, self._mapping[attr]]

class EventView(ArrayView):
  def __init__(self, gs: GameState, subject: Group, arr: np.ndarray):
    super().__init__(EventAttr, 'event', gs, subject, arr)

  def get_attribute(self, attr) -> np.ndarray:
    assert hasattr(EventCode, attr), 'Invalid event code'
    arr = self._arr[np.in1d(self._arr[:, EventAttr['event']],
                                          getattr(EventCode, attr))]
    return EventCodeView(attr, self._gs, self._subject, arr)

class TileView(ArrayView):
  def __init__(self, gs: GameState, subject: Group, arr: np.ndarray):
    super().__init__(TileAttr, 'tile', gs, subject, arr)

  def get_attribute(self, attr) -> np.ndarray:
    return [o[:, self._mapping[attr]] for o in self._arr]

class EventCodeView(ArrayView):
  def __init__(self,
               name: str,
               gs: GameState,
               subject: Group,
               arr: np.ndarray):
    super().__init__(EventAttr, name, gs, subject, arr)

  def get_attribute(self, attr) -> np.ndarray:
    return self._arr[:, self._mapping[attr]]

# Group
class GroupObsView:
  def __init__(self, gs: GameState, subject: Group):
    self._gs = gs

    valid_agents = filter(lambda eid: eid in gs.env_obs,subject.agents)
    self._obs = [gs.env_obs[ent_id] for ent_id in valid_agents]
    self._subject = subject

  @CachedProperty
  def tile(self):
    return TileView(self._gs, self._subject, [o.tiles for o in self._obs])

  def __getattr__(self, attr):
    return [getattr(o, attr) for o in self._obs]

class GroupView:
  def __init__(self, gs: GameState, subject: Group):
    self._gs = gs
    self._subject = subject
    self._subject_hash = hash(subject)

  @CachedProperty
  def obs(self):
    return GroupObsView(self._gs, self._subject)

  @CachedProperty
  def _sbj_ent(self):
    return self._gs.where_in_id('entity', self._subject.agents)

  @CachedProperty
  def entity(self):
    return EntityView(self._gs, self._subject, self._sbj_ent)

  @CachedProperty
  def _sbj_item(self):
    return self._gs.where_in_id('item', self._subject.agents)

  @CachedProperty
  def item(self):
    return ItemView(self._gs, self._subject, self._sbj_item)

  @CachedProperty
  def _sbj_event(self):
    return self._gs.where_in_id('event', self._subject.agents)

  @CachedProperty
  def event(self):
    return EventView(self._gs, self._subject, self._sbj_event)

  def __getattribute__(self, attr):
    if attr in {'_gs','_subject','_sbj_ent','_sbj_item',
    'entity','item','event','obs', '_subject_hash'}:
      return object.__getattribute__(self, attr)

    # Cached optimization
    k = (self._subject_hash, attr)
    cache = self._gs.cache_result
    if k in cache:
      return cache[k]

    try:
      # Get property
      if attr in EntityAttrKeys:
        v = getattr(self.entity, attr)
      else:
        v = object.__getattribute__(self, attr)
      cache[k] = v
      return v
    except AttributeError:
      # View behavior
      return object.__getattribute__(self._gs, attr)

class GameStateGenerator:
  def __init__(self, realm: Realm, config: Config):
    self.config = deepcopy(config)
    self.spawn_pos: Dict[int, Tuple[int, int]] = {}

    for ent_id, ent in realm.players.items():
      self.spawn_pos.update( {ent_id: ent.pos} )

  def generate(self, realm: Realm, env_obs: Dict[int, Observation]) -> GameState:
    # copy the datastore, by running astype
    entity_all = EntityState.Query.table(realm.datastore).copy()
    alive_agents = entity_all[:, EntityAttr["id"]]
    alive_agents = set(alive_agents[alive_agents > 0])
    item_data = ItemState.Query.table(realm.datastore).copy()
    event_data = EventState.Query.table(realm.datastore).copy()
    return GameState(
      current_tick = realm.tick,
      config = self.config,
      spawn_pos = self.spawn_pos,
      alive_agents = alive_agents,
      env_obs = env_obs,
      entity_data = entity_all,
      entity_index = precompute_index(entity_all, EntityAttr["id"]),
      item_data = item_data,
      item_index = precompute_index(item_data, ItemAttr["owner_id"]),
      event_data = event_data,
      event_index = precompute_index(event_data, EventAttr['ent_id']),
      seize_status = realm.seize_status,
      cache_result = {}
    )

def precompute_index(table, id_col):
  index = defaultdict()
  for row, id_ in enumerate(table[:,id_col]):
    if id_ in index:
      index[id_].append(row)
    else:
      index[id_] = [row]
  return index
