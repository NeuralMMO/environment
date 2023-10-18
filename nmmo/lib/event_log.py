from types import SimpleNamespace
from typing import List
from copy import deepcopy

import numpy as np

from nmmo.datastore.serialized import SerializedState
from nmmo.entity import Entity
from nmmo.systems.item import Item
from nmmo.lib.event_code import EventCode

# pylint: disable=no-member
EventState = SerializedState.subclass("Event", [
  "recorded", # event_log is write-only, no update or delete, so no need for row id
  "ent_id",
  "tick",

  "event",

  "type",
  "level",
  "number",
  "gold",
  "target_ent",
])

EventAttr = EventState.State.attr_name_to_col

EventState.Query = SimpleNamespace(
  table=lambda ds: ds.table("Event").where_eq(EventAttr["recorded"], 1),
  by_event=lambda ds, event_code: ds.table("Event").where_eq(
    EventAttr["event"], event_code),
  by_tick=lambda ds, tick: ds.table("Event").where_eq(
    EventAttr["tick"], tick),
)

# defining col synoyms for different event types
ATTACK_COL_MAP = {
  'combat_style': EventAttr['type'],
  'damage': EventAttr['number'] }

ITEM_COL_MAP = {
  'item_type': EventAttr['type'],
  'quantity': EventAttr['number'],
  'price': EventAttr['gold'],
  'item_id': EventAttr['target_ent'] }

LEVEL_COL_MAP = { 'skill': EventAttr['type'] }

EXPLORE_COL_MAP = { 'distance': EventAttr['number'] }


class EventLogger(EventCode):
  def __init__(self, realm):
    self.realm = realm
    self.config = realm.config
    self.datastore = realm.datastore

    self.valid_events = { val: evt for evt, val in EventCode.__dict__.items()
                           if isinstance(val, int) }
    self._data_by_tick = {}
    self._last_tick = 0
    self._empty_data = np.empty((0, len(EventAttr)))

    # add synonyms to the attributes
    self.attr_to_col = deepcopy(EventAttr)
    self.attr_to_col.update(ATTACK_COL_MAP)
    self.attr_to_col.update(ITEM_COL_MAP)
    self.attr_to_col.update(LEVEL_COL_MAP)
    self.attr_to_col.update(EXPLORE_COL_MAP)

  def reset(self):
    EventState.State.table(self.datastore).reset()

  # define event logging
  def _create_event(self, entity: Entity, event_code: int):
    log = EventState(self.datastore)
    log.recorded.update(1)
    log.ent_id.update(entity.ent_id)
    # the tick increase by 1 after executing all actions
    log.tick.update(self.realm.tick+1)
    log.event.update(event_code)

    return log

  def record(self, event_code: int, entity: Entity, **kwargs):
    if event_code in [EventCode.EAT_FOOD, EventCode.DRINK_WATER,
                      EventCode.GIVE_ITEM, EventCode.DESTROY_ITEM,
                      EventCode.GIVE_GOLD]:
      # Logs for these events are for counting only
      self._create_event(entity, event_code)
      return

    if event_code == EventCode.GO_FARTHEST: # use EXPLORE_COL_MAP
      if ('distance' in kwargs and kwargs['distance'] > 0):
        log = self._create_event(entity, event_code)
        log.number.update(kwargs['distance'])
        return

    if event_code == EventCode.SCORE_HIT:
      # kwargs['combat_style'] should be Skill.CombatSkill
      if ('combat_style' in kwargs and kwargs['combat_style'].SKILL_ID in [1, 2, 3]) & \
         ('target' in kwargs and isinstance(kwargs['target'], Entity)) & \
         ('damage' in kwargs and kwargs['damage'] >= 0):
        log = self._create_event(entity, event_code)
        log.type.update(kwargs['combat_style'].SKILL_ID)
        log.number.update(kwargs['damage'])
        log.target_ent.update(kwargs['target'].ent_id)
        return

    if event_code == EventCode.PLAYER_KILL:
      if ('target' in kwargs and isinstance(kwargs['target'], Entity)):
        target = kwargs['target']
        log = self._create_event(entity, event_code)
        log.target_ent.update(target.ent_id)
        log.level.update(target.attack_level)
        return

    if event_code == EventCode.LOOT_ITEM:
      if ('item' in kwargs and isinstance(kwargs['item'], Item)) & \
         ('target' in kwargs and isinstance(kwargs['target'], Entity)):
        item = kwargs['item']
        log = self._create_event(entity, event_code)
        log.type.update(item.ITEM_TYPE_ID)
        log.level.update(item.level.val)
        log.number.update(item.quantity.val)
        log.target_ent.update(item.id.val)
        return

    if event_code == EventCode.LOOT_GOLD:
      if ('amount' in kwargs and kwargs['amount'] > 0) & \
         ('target' in kwargs and isinstance(kwargs['target'], Entity)):
        log = self._create_event(entity, event_code)
        log.gold.update(kwargs['amount'])
        log.target_ent.update(kwargs['target'].ent_id)
        return

    if event_code in [EventCode.CONSUME_ITEM, EventCode.HARVEST_ITEM, EventCode.EQUIP_ITEM,
                      EventCode.FIRE_AMMO]:
      if ('item' in kwargs and isinstance(kwargs['item'], Item)):
        item = kwargs['item']
        log = self._create_event(entity, event_code)
        log.type.update(item.ITEM_TYPE_ID)
        log.level.update(item.level.val)
        log.number.update(item.quantity.val)
        log.target_ent.update(item.id.val)
        return

    if event_code in [EventCode.LIST_ITEM, EventCode.BUY_ITEM]:
      if ('item' in kwargs and isinstance(kwargs['item'], Item)) & \
         ('price' in kwargs and kwargs['price'] > 0):
        item = kwargs['item']
        log = self._create_event(entity, event_code)
        log.type.update(item.ITEM_TYPE_ID)
        log.level.update(item.level.val)
        log.number.update(item.quantity.val)
        log.gold.update(kwargs['price'])
        log.target_ent.update(item.id.val)
        return

    if event_code == EventCode.EARN_GOLD:
      if ('amount' in kwargs and kwargs['amount'] > 0):
        log = self._create_event(entity, event_code)
        log.gold.update(kwargs['amount'])
        return

    if event_code == EventCode.LEVEL_UP:
      # kwargs['skill'] should be Skill.Skill
      if ('skill' in kwargs and kwargs['skill'].SKILL_ID in range(1,9)) & \
         ('level' in kwargs and kwargs['level'] >= 0):
        log = self._create_event(entity, event_code)
        log.type.update(kwargs['skill'].SKILL_ID)
        log.level.update(kwargs['level'])
        return

    # If reached here, then something is wrong
    # CHECK ME: The below should be commented out after debugging
    raise ValueError(f"Event code: {event_code}", kwargs)

  def update(self):
    curr_tick = self.realm.tick
    if curr_tick > self._last_tick:
      self._data_by_tick[curr_tick] = EventState.Query.by_tick(self.datastore, curr_tick)
      self._last_tick = curr_tick

  def get_data(self, event_code=None, agents: List[int]=None, tick: int=None) -> np.ndarray:
    if tick is not None:
      if tick == -1:
        tick = self._last_tick
      if tick not in self._data_by_tick:
        return self._empty_data
      event_data = self._data_by_tick[tick]
    else:
      event_data = EventState.Query.table(self.datastore)

    if event_data.shape[0] > 0:
      if event_code is None:
        flt_idx = event_data[:, EventAttr["event"]] > 0
      else:
        flt_idx = event_data[:, EventAttr["event"]] == event_code
      if agents:
        flt_idx &= np.in1d(event_data[:, EventAttr["ent_id"]], agents)
      return event_data[flt_idx]

    return self._empty_data
