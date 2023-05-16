from collections.abc import Mapping
from typing import Dict

import numpy as np
from ordered_set import OrderedSet

from nmmo.entity.entity import Entity
from nmmo.entity.npc import NPC
from nmmo.entity.player import Player
from nmmo.lib import spawn
from nmmo.systems import combat


class EntityGroup(Mapping):
  def __init__(self, realm):
    self.datastore = realm.datastore
    self.realm = realm
    self.config = realm.config

    self.entities: Dict[int, Entity] = {}
    self.dead: Dict[int, Entity] = {}

  def __len__(self):
    return len(self.entities)

  def __contains__(self, e):
    return e in self.entities

  def __getitem__(self, key) -> Entity:
    return self.entities[key]

  def __iter__(self) -> Entity:
    yield from self.entities

  def items(self):
    return self.entities.items()

  @property
  def corporeal(self):
    return {**self.entities, **self.dead}

  @property
  def packet(self):
    return {k: v.packet() for k, v in self.corporeal.items()}

  def reset(self):
    for ent in self.entities.values():
      ent.datastore_record.delete()

    self.entities = {}
    self.dead = {}

  def spawn(self, entity):
    pos, ent_id = entity.pos, entity.id.val
    self.realm.map.tiles[pos].add_entity(entity)
    self.entities[ent_id] = entity

  def cull(self):
    self.dead = {}
    for ent_id in list(self.entities):
      player = self.entities[ent_id]
      if not player.alive:
        r, c  = player.pos
        ent_id = player.ent_id
        self.dead[ent_id] = player

        self.realm.map.tiles[r, c].remove_entity(ent_id)
        self.entities[ent_id].datastore_record.delete()
        del self.entities[ent_id]

    return self.dead

  def update(self, actions):
    for entity in self.entities.values():
      entity.update(self.realm, actions)


class NPCManager(EntityGroup):
  def __init__(self, realm):
    super().__init__(realm)
    self.next_id = -1
    self.spawn_dangers = []

  def reset(self):
    super().reset()
    self.next_id = -1
    self.spawn_dangers = []

  def spawn(self):
    config = self.config

    if not config.NPC_SYSTEM_ENABLED:
      return

    for _ in range(config.NPC_SPAWN_ATTEMPTS):
      if len(self.entities) >= config.NPC_N:
        break

      if self.spawn_dangers:
        danger = self.spawn_dangers[-1]
        r, c   = combat.spawn(config, danger)
      else:
        center = config.MAP_CENTER
        border = self.config.MAP_BORDER
        # pylint: disable=unbalanced-tuple-unpacking
        r, c   = np.random.randint(border, center+border, 2).tolist()

      npc = NPC.spawn(self.realm, (r, c), self.next_id)
      if npc:
        super().spawn(npc)
        self.next_id -= 1

    if self.spawn_dangers:
      self.spawn_dangers.pop()

  def cull(self):
    for entity in super().cull().values():
      self.spawn_dangers.append(entity.spawn_danger)

  def actions(self, realm):
    actions = {}
    for idx, entity in self.entities.items():
      actions[idx] = entity.decide(realm)
    return actions

class PlayerManager(EntityGroup):
  def __init__(self, realm):
    super().__init__(realm)
    self.loader  = self.realm.config.PLAYER_LOADER
    self.agents = None
    self.spawned = None

  def reset(self):
    super().reset()
    self.agents  = self.loader(self.config)
    self.spawned = OrderedSet()

  def spawn_individual(self, r, c, idx):
    agent = next(self.agents)
    agent      = agent(self.config, idx)
    player     = Player(self.realm, (r, c), agent)
    super().spawn(player)

  def spawn(self):
    idx = 0
    for r, c in spawn.spawn_concurrent(self.config):
      idx += 1

      if idx in self.entities:
        continue

      if idx in self.spawned:
        continue

      self.spawned.add(idx)
      self.spawn_individual(r, c, idx)
