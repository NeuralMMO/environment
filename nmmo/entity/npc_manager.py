from nmmo.entity.entity_manager import EntityGroup
from nmmo.entity.npc import NPC, Soldier
from nmmo.core import action
from nmmo.systems import combat
from nmmo.lib import spawn


class NPCManager(EntityGroup):
  def __init__(self, realm, np_random):
    super().__init__(realm, np_random)
    self.next_id = -1
    self.spawn_dangers = []

  def reset(self, np_random):
    super().reset(np_random)
    self.next_id = -1
    self.spawn_dangers.clear()

  def actions(self):
    return {idx: entity.decide() for idx, entity in self.entities.items()}

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

  def spawn_soldier(self, r, c, name, order=None):
    npc = Soldier(self.realm, (r, c), self.next_id, name, order) \
          if self.realm.map.tiles[r, c].habitable else None
    if npc:
      super().spawn_entity(npc)
      self.next_id -= 1
      # NOTE: randomly set the combat style. revisit later
      npc.skills.style = self._np_random.choice([action.Melee, action.Range, action.Mage])
    return npc

  def area_spawn(self, r_min, r_max, c_min, c_max, num_spawn, npc_init_fn):
    assert r_min < r_max and c_min < c_max, "Invalid area"
    assert num_spawn > 0, "Invalid number of spawns"
    while num_spawn > 0:
      r = self._np_random.integers(r_min, r_max+1)
      c = self._np_random.integers(c_min, c_max+1)
      if npc_init_fn(r, c):
        num_spawn -= 1

  def edge_spawn(self, num_spawn, npc_init_fn):
    assert num_spawn > 0, "Invalid number of spawns"
    edge_locs = spawn.spawn_concurrent(self.config, self._np_random)
    assert len(edge_locs) >= num_spawn, "Not enough edge locations"
    while num_spawn > 0:
      r, c = edge_locs.pop()
      npc = npc_init_fn(r, c)
      if npc:
        num_spawn -= 1
