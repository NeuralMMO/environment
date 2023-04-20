from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import numpy as np

import nmmo
from nmmo.core.log_helper import LogHelper
from nmmo.core.map import Map
from nmmo.core.render_helper import RenderHelper
from nmmo.core.replay_helper import ReplayHelper
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.entity.entity_manager import NPCManager, PlayerManager
from nmmo.io.action import Action, Buy
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.systems.exchange import Exchange
from nmmo.systems.item import Item, ItemState
from nmmo.lib.event_log import EventLogger, EventState

def prioritized(entities: Dict, merged: Dict):
  """Sort actions into merged according to priority"""
  for idx, actions in entities.items():
    for atn, args in actions.items():
      merged[atn.priority].append((idx, (atn, args.values())))
  return merged


class Realm:
  """Top-level world object"""

  def __init__(self, config):
    self.config = config
    assert isinstance(
        config, nmmo.config.Config
    ), f"Config {config} is not a config instance (did you pass the class?)"

    Action.hook(config)

    # Generate maps if they do not exist
    config.MAP_GENERATOR(config).generate_all_maps()

    self.datastore = NumpyDatastore()
    for s in [TileState, EntityState, ItemState, EventState]:
      self.datastore.register_object_type(s._name, s.State.num_attributes)

    self.tick = 0
    self.exchange = None

    # Load the world file
    self.map = Map(config, self)

    self.replay_helper = ReplayHelper.create(self)
    self.render_helper = RenderHelper.create(self)
    self.log_helper = LogHelper.create(self)
    self.event_log = EventLogger(self)

    # Entity handlers
    self.players = PlayerManager(self)
    self.npcs = NPCManager(self)

    # Global item registry
    self.items = {}

    # Initialize actions
    nmmo.Action.init(config)

  def reset(self, map_id: int = None):
    """Reset the environment and load the specified map

    Args:
        idx: Map index to load
    """
    self.log_helper.reset()
    self.event_log.reset()
    self.map.reset(map_id or np.random.randint(self.config.MAP_N) + 1)

    # EntityState and ItemState tables must be empty after players/npcs.reset()
    self.players.reset()
    self.npcs.reset()


    EntityState.State.table(self.datastore).reset()

    assert EntityState.State.table(self.datastore).is_empty(), \
        "EntityState table is not empty"

    # TODO(kywch): ItemState table is not empty after players/npcs.reset()
    #   but should be. Will fix this while debugging the item system.
    # assert ItemState.State.table(self.datastore).is_empty(), \
    #     "ItemState table is not empty"
    ItemState.State.table(self.datastore).reset()

    self.players.spawn()
    self.npcs.spawn()
    self.tick = 0

    # Global item exchange
    self.exchange = Exchange(self)

    # Global item registry
    Item.INSTANCE_ID = 0
    self.items = {}

    self.replay_helper.update()

  def packet(self):
    """Client packet"""
    return {
      "environment": self.map.repr,
      "border": self.config.MAP_BORDER,
      "size": self.config.MAP_SIZE,
      "resource": self.map.packet,
      "player": self.players.packet,
      "npc": self.npcs.packet,
      "market": self.exchange.packet,
  }

  @property
  def num_players(self):
    """Number of player agents"""
    return len(self.players.entities)

  def entity(self, ent_id):
    e = self.entity_or_none(ent_id)
    assert e is not None, f"Entity {ent_id} does not exist"
    return e

  def entity_or_none(self, ent_id):
    if ent_id is None:
      return None

    """Get entity by ID"""
    if ent_id < 0:
      return self.npcs.get(ent_id)

    return self.players.get(ent_id)

  def step(self, actions):
    """Run game logic for one tick

    Args:
        actions: Dict of agent actions

    Returns:
        dead: List of dead agents
    """
    # Prioritize actions
    npc_actions = self.npcs.actions(self)
    merged = defaultdict(list)
    prioritized(actions, merged)
    prioritized(npc_actions, merged)

    # Update entities and perform actions
    self.players.update(actions)
    self.npcs.update(npc_actions)

    # Execute actions -- CHECK ME the below priority
    #  - 10: Use - equip ammo, restore HP, etc.
    #  - 20: Buy - exchange while sellers, items, buyers are all intact
    #  - 30: Give, GiveGold - transfer while both are alive and at the same tile
    #  - 40: Destroy - use with SELL/GIVE, if not gone, destroy and recover space
    #  - 50: Attack
    #  - 60: Move
    #  - 70: Sell - to guarantee the listed items are available to buy
    #  - 99: Comm
    for priority in sorted(merged):
      # TODO: we should be randomizing these, otherwise the lower ID agents
      # will always go first. --> ONLY SHUFFLE BUY
      if priority == Buy.priority:
        np.random.shuffle(merged[priority])

      # CHECK ME: do we need this line?
      # ent_id, (atn, args) = merged[priority][0]
      for ent_id, (atn, args) in merged[priority]:
        ent = self.entity(ent_id)
        if ent.alive:
          atn.call(self, ent, *args)

    dead = self.players.cull()
    self.npcs.cull()

    # Update map
    self.map.step()
    self.exchange.step(self.tick)
    self.log_helper.update(dead)

    self.tick += 1

    self.replay_helper.update()

    return dead

  def log_milestone(self, category: str, value: float, message: str = None, tags: Dict = None):
    self.log_helper.log_milestone(category, value)
    self.log_helper.log_event(category, value)

    if self.config.LOG_VERBOSE:
      # TODO: more general handling of tags, if necessary
      if tags and 'player_id' in tags:
        logging.info("Milestone (Player %d): %s %s %s", tags['player_id'], category, value, message)
      else:
        logging.info("Milestone: %s %s %s", category, value, message)
