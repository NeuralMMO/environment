from __future__ import annotations
from collections import defaultdict
from typing import Dict
import numpy as np

import nmmo
from nmmo.core.map import Map
from nmmo.core.tile import TileState
from nmmo.core.action import Action, Buy, Comm
from nmmo.entity.entity import EntityState
from nmmo.entity.entity_manager import PlayerManager
from nmmo.entity.npc_manager import NPCManager
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.systems.exchange import Exchange
from nmmo.systems.item import ItemState
from nmmo.lib.event_log import EventLogger, EventState
from nmmo.render.replay_helper import ReplayHelper

def prioritized(entities: Dict, merged: Dict):
  """Sort actions into merged according to priority"""
  for idx, actions in entities.items():
    for atn, args in actions.items():
      merged[atn.priority].append((idx, (atn, args.values())))
  return merged


class Realm:
  """Top-level world object"""

  def __init__(self, config, np_random):
    self.config = config
    self._np_random = np_random # rng
    assert isinstance(
        config, nmmo.config.Config
    ), f"Config {config} is not a config instance (did you pass the class?)"

    Action.hook(config)

    self.datastore = NumpyDatastore()
    for s in [TileState, EntityState, ItemState, EventState]:
      self.datastore.register_object_type(s._name, s.State.num_attributes)

    self.tick = None # to use as a "reset" checker

    # Load the world file
    self.map = Map(config, self, self._np_random)
    self.fog_map = np.zeros((config.MAP_SIZE, config.MAP_SIZE), dtype=np.float16)

    # Event logger
    self.event_log = EventLogger(self)

    # Entity handlers
    self.players = PlayerManager(self, self._np_random)
    self.npcs = NPCManager(self, self._np_random)

    # Global item registry
    self.items = {}

    # Global item exchange
    self.exchange = Exchange(self)

    # Replay helper
    self._replay_helper = None

    # Initialize actions
    nmmo.Action.init(config)

  def reset(self, np_random, map_dict,
            custom_spawn=False,
            seize_targets=None,
            delete_dead_player=True):
    """Reset the sub-systems and load the provided map"""
    self._np_random = np_random
    self.tick = 0
    self.update_fog_map(reset=True)
    #self.event_log.reset()
    self.items.clear()
    self.exchange.reset()
    if self._replay_helper is not None:
      self._replay_helper.reset()

    # Load the map np array into the map, tiles and reset
    self.map.reset(map_dict, self._np_random, seize_targets)

    # EntityState and ItemState tables must be empty after players/npcs.reset()
    self.players.reset(self._np_random, delete_dead_player)
    self.npcs.reset(self._np_random)
    # assert EntityState.State.table(self.datastore).is_empty(), \
    #     "EntityState table is not empty"
    # assert ItemState.State.table(self.datastore).is_empty(), \
    #     "ItemState table is not empty"

    # DataStore id allocator must be reset to be deterministic
    EntityState.State.table(self.datastore).reset()
    ItemState.State.table(self.datastore).reset()

    self.event_log.reset()  # reset this last for debugging

    if custom_spawn is False:
      # NOTE: custom spawning npcs and agents can be done outside, after reset()
      self.npcs.default_spawn()
      self.players.spawn()

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
    """Number of alive player agents"""
    return len(self.players.entities)

  @property
  def seize_status(self):
    return self.map.seize_status

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
    npc_actions = self.npcs.actions()
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
        self._np_random.shuffle(merged[priority])

      # CHECK ME: do we need this line?
      # ent_id, (atn, args) = merged[priority][0]
      for ent_id, (atn, args) in merged[priority]:
        ent = self.entity(ent_id)
        if (ent.alive and not ent.status.frozen) or \
           (ent.is_recon and priority == Comm.priority):  # recons can always comm
          atn.call(self, ent, *args)
    dead_players = self.players.cull()
    dead_npcs = self.npcs.cull()

    self.tick += 1

    # These require the updated tick
    self.map.step()
    self.update_fog_map()
    self.exchange.step()
    self.event_log.update()
    if self._replay_helper is not None:
      self._replay_helper.update()

    return dead_players, dead_npcs

  def update_fog_map(self, reset=False):
    fog_start_tick = self.config.DEATH_FOG_ONSET
    if fog_start_tick is None:
      return

    fog_speed = self.config.DEATH_FOG_SPEED
    center = self.config.MAP_SIZE // 2
    safe = self.config.DEATH_FOG_FINAL_SIZE

    if reset:
      dist = -self.config.MAP_BORDER
      for i in range(center):
        l, r = i, self.config.MAP_SIZE - i
        # positive value represents the poison strength
        # negative value represents the shortest distance to poison area
        self.fog_map[l:r, l:r] = -dist
        dist += 1
      # mark the safe area
      self.fog_map[center-safe:center+safe+1, center-safe:center+safe+1] = -self.config.MAP_SIZE
      return

    # consider the map border so that the fog can hit the border at fog_start_tick
    if self.tick >= fog_start_tick:
      self.fog_map += fog_speed
      # mark the safe area
      self.fog_map[center-safe:center+safe+1, center-safe:center+safe+1] = -self.config.MAP_SIZE

  def record_replay(self, replay_helper: ReplayHelper) -> ReplayHelper:
    self._replay_helper = replay_helper
    self._replay_helper.set_realm(self)
    return replay_helper
