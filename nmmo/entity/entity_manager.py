from collections.abc import Mapping
from typing import Dict

from nmmo.entity.entity import Entity, EntityState
from nmmo.entity.player import Player
from nmmo.lib import spawn, event_code


class EntityGroup(Mapping):
  def __init__(self, realm, np_random):
    self.datastore = realm.datastore
    self.realm = realm
    self.config = realm.config
    self._np_random = np_random
    self._entity_table = EntityState.Query.table(self.datastore)

    self.entities: Dict[int, Entity] = {}
    self.dead_this_tick: Dict[int, Entity] = {}
    self._delete_dead_entity = True  # is default

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

  def reset(self, np_random, delete_dead_entity=True):
    self._np_random = np_random # reset the RNG
    self._delete_dead_entity = delete_dead_entity
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

  def cull_entity(self, entity):
    pos, ent_id = entity.pos, entity.id.val
    self.realm.map.tiles[pos].remove_entity(ent_id)
    self.entities.pop(ent_id)
    # destroy the remaining items (of starved/dehydrated players)
    #    of the agents who don't go through receive_damage()
    if self.config.ITEM_SYSTEM_ENABLED:
      for item in list(entity.inventory.items):
        item.destroy()
    if ent_id > 0:
      self.realm.event_log.record(event_code.EventCode.AGENT_CULLED, entity)

  def cull(self):
    self.dead_this_tick.clear()
    for ent in [ent for ent in self.entities.values() if not ent.alive]:
      self.dead_this_tick[ent.ent_id] = ent
      self.cull_entity(ent)
      if self._delete_dead_entity:
        ent.datastore_record.delete()
    return self.dead_this_tick

  def update(self, actions):
    # # batch updates
    # # time_alive, damage are from entity.py, History.update()
    # ent_idx = self._entity_table[:, EntityState.State.attr_name_to_col["id"]] != 0
    # self._entity_table[ent_idx, EntityState.State.attr_name_to_col["time_alive"]] += 1
    # self._entity_table[ent_idx, EntityState.State.attr_name_to_col["damage"]] = 0
    # # freeze from entity.py, Status.update()
    # freeze_idx = self._entity_table[:, EntityState.State.attr_name_to_col["freeze"]] > 0
    # self._entity_table[freeze_idx, EntityState.State.attr_name_to_col["freeze"]] -= 1

    for entity in self.entities.values():
      entity.update(self.realm, actions)

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
