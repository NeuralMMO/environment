from types import SimpleNamespace
from typing import List
from copy import deepcopy

import numpy as np

from nmmo.datastore.serialized import SerializedState
from nmmo.entity import Entity
from nmmo.systems.item import Item
from nmmo.lib.log import EventCode

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
)

# defining col synoyms for different event types
ATTACK_COL_MAP = {
  'combat_style': EventAttr['type'],
  'damage': EventAttr['number'] }

ITEM_COL_MAP = {
  'item_type': EventAttr['type'],
  'quantity': EventAttr['number'],
  'price': EventAttr['gold'] }

LEVEL_COL_MAP = { 'skill': EventAttr['type'] }

EXPLORE_COL_MAP = { 'distance': EventAttr['number'] }


class EventLogger(EventCode):
  def __init__(self, realm):
    self.realm = realm
    self.config = realm.config
    self.datastore = realm.datastore

    self.valid_events = { val: evt for evt, val in EventCode.__dict__.items()
                           if isinstance(val, int) }

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
         ('damage' in kwargs and kwargs['damage'] >= 0):
        log = self._create_event(entity, event_code)
        log.type.update(kwargs['combat_style'].SKILL_ID)
        log.number.update(kwargs['damage'])
        return

    if event_code == EventCode.PLAYER_KILL:
      if ('target' in kwargs and isinstance(kwargs['target'], Entity)):
        target = kwargs['target']
        log = self._create_event(entity, event_code)
        log.target_ent.update(target.ent_id)

        # CHECK ME: attack_level or "general" level?? need to clarify
        log.level.update(target.attack_level)
        return

    if event_code in [EventCode.CONSUME_ITEM, EventCode.HARVEST_ITEM, EventCode.EQUIP_ITEM]:
      # CHECK ME: item types should be checked. For example,
      #   Only Ration and Potion can be consumed
      #   Only Ration, Potion, Whetstone, Arrow, Runes can be produced
      #   The quantity should be 1 for all of these events
      if ('item' in kwargs and isinstance(kwargs['item'], Item)):
        item = kwargs['item']
        log = self._create_event(entity, event_code)
        log.type.update(item.ITEM_TYPE_ID)
        log.level.update(item.level.val)
        log.number.update(item.quantity.val)
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

  def get_data(self, event_code=None, agents: List[int]=None):
    if event_code is None:
      event_data = EventState.Query.table(self.datastore)
    elif event_code in self.valid_events:
      event_data = EventState.Query.by_event(self.datastore, event_code)
    else:
      return None

    if agents:
      flt_idx = np.in1d(event_data[:, EventAttr['ent_id']], agents)
      return event_data[flt_idx]

    return event_data
