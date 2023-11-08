from collections.abc import Mapping
from typing import Dict

from nmmo.entity.entity import Entity
from nmmo.entity.npc import NPC
from nmmo.entity.player import Player
from nmmo.lib import spawn
from nmmo.systems import combat


class EntityGroup(Mapping):
  def __init__(self, realm, np_random):
    self.datastore = realm.datastore
    self.realm = realm
    self.config = realm.config
    self._np_random = np_random

    self.entities: Dict[int, Entity] = {}
    self.dead_this_tick: Dict[int, Entity] = {}

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
    return {**self.entities, **self.dead_this_tick}

  @property
  def packet(self):
    return {k: v.packet() for k, v in self.corporeal.items()}

  def reset(self, np_random):
    self._np_random = np_random # reset the RNG
    for ent in self.entities.values():
      # destroy the items
      if self.config.ITEM_SYSTEM_ENABLED:
        for item in list(ent.inventory.items):
          item.destroy()
      ent.datastore_record.delete()

    self.entities.clear()
    self.dead_this_tick.clear()

  def spawn_entity(self, entity):
    pos, ent_id = entity.pos, entity.id.val
    self.realm.map.tiles[pos].add_entity(entity)
    self.entities[ent_id] = entity

  def cull(self):
    self.dead_this_tick.clear()
    for ent_id in list(self.entities):
      player = self.entities[ent_id]
      if not player.alive:
        r, c  = player.pos
        ent_id = player.ent_id
        self.dead_this_tick[ent_id] = player

        self.realm.map.tiles[r, c].remove_entity(ent_id)

        # destroy the remaining items (of starved/dehydrated players)
        #    of the agents who don't go through receive_damage()
        if self.config.ITEM_SYSTEM_ENABLED:
          for item in list(player.inventory.items):
            item.destroy()

        self.entities[ent_id].datastore_record.delete()
        del self.entities[ent_id]

    return self.dead_this_tick

  def update(self, actions):
    for entity in self.entities.values():
      entity.update(self.realm, actions)


class NPCManager(EntityGroup):
  def __init__(self, realm, np_random):
    super().__init__(realm, np_random)
    self.next_id = -1
    self.spawn_dangers = []

  def reset(self, np_random):
    super().reset(np_random)
    self.next_id = -1
    self.spawn_dangers.clear()

  def default_spawn(self):
    config = self.config

    if not config.NPC_SYSTEM_ENABLED:
      return

    for _ in range(config.NPC_SPAWN_ATTEMPTS):
      if len(self.entities) >= config.NPC_N:
        break

      if self.spawn_dangers:
        danger = self.spawn_dangers.pop()
        r, c   = combat.spawn(config, danger, self._np_random)
      else:
        center = config.MAP_CENTER
        border = self.config.MAP_BORDER
        # pylint: disable=unbalanced-tuple-unpacking
        r, c   = self._np_random.integers(border, center+border, 2).tolist()

      npc = NPC.default_spawn(self.realm, (r, c), self.next_id, self._np_random)
      if npc:
        super().spawn_entity(npc)
        self.next_id -= 1

  def actions(self, realm):
    actions = {}
    for idx, entity in self.entities.items():
      actions[idx] = entity.decide(realm)
    return actions

class PlayerManager(EntityGroup):
  def spawn(self, agent_loader: spawn.SequentialLoader = None):
    if agent_loader is None:
      agent_loader = self.config.PLAYER_LOADER(self.config, self._np_random)

    # Check and assign the reslient flag
    resilient_flag = [False] * self.config.PLAYER_N
    if self.config.RESOURCE_SYSTEM_ENABLED:
      num_resilient = round(self.config.RESOURCE_RESILIENT_POPULATION * self.config.PLAYER_N)
      for idx in range(num_resilient):
        resilient_flag[idx] = self.config.RESOURCE_DAMAGE_REDUCTION > 0
      self._np_random.shuffle(resilient_flag)

    # Spawn the players
    for agent_id in self.config.POSSIBLE_AGENTS:
      r, c = agent_loader.get_spawn_position(agent_id)

      if agent_id in self.entities:
        continue

      # NOTE: put spawn_individual() here. Is a separate function necessary?
      agent = next(agent_loader)  # get agent cls from config.PLAYERS
      agent = agent(self.config, agent_id)
      player = Player(self.realm, (r, c), agent, resilient_flag[agent_id-1])
      super().spawn_entity(player)
