from types import SimpleNamespace
import numpy as np

from nmmo.datastore.serialized import SerializedState
from nmmo.lib import material

# pylint: disable=no-member,protected-access
TileState = SerializedState.subclass(
  "Tile", [
    "row",
    "col",
    "material_id",
  ])

TileState.Limits = lambda config: {
  "row": (0, config.MAP_SIZE-1),
  "col": (0, config.MAP_SIZE-1),
  "material_id": (0, config.MAP_N_TILE),
}

TileState.Query = SimpleNamespace(
  window=lambda ds, r, c, radius: ds.table("Tile").window(
    TileState.State.attr_name_to_col["row"],
    TileState.State.attr_name_to_col["col"],
    r, c, radius),
  get_map=lambda ds, map_size:
    ds.table("Tile")._data[1:(map_size*map_size+1)]
                    .reshape((map_size,map_size,len(TileState.State.attr_name_to_col)))
)

class Tile(TileState):
  def __init__(self, realm, r, c):
    super().__init__(realm.datastore, TileState.Limits(realm.config))
    self.realm = realm
    self.config = realm.config

    self.row.update(r)
    self.col.update(c)

    self.state = None
    self.material = None
    self.depleted = False
    self.tex = None

    self.entities = {}

  @property
  def repr(self):
    return ((self.row.val, self.col.val))

  @property
  def pos(self):
    return self.row.val, self.col.val

  @property
  def habitable(self):
    return self.material in material.Habitable

  @property
  def impassible(self):
    return self.material in material.Impassible

  @property
  def void(self):
    return self.material == material.Void

  def reset(self, mat, config):
    self.state = mat(config)
    self.material = mat(config)
    self.material_id.update(self.state.index)

    self.depleted = False
    self.tex = self.material.tex

    self.entities = {}

  def add_entity(self, ent):
    assert ent.ent_id not in self.entities
    self.entities[ent.ent_id] = ent

  def remove_entity(self, ent_id):
    assert ent_id in self.entities
    del self.entities[ent_id]

  def step(self):
    if not self.depleted or np.random.rand() > self.material.respawn:
      return

    self.depleted = False
    self.state = self.material
    self.material_id.update(self.state.index)

  def harvest(self, deplete):
    assert not self.depleted, f'{self.state} is depleted'
    assert self.state in material.Harvestable, f'{self.state} not harvestable'

    if deplete:
      self.depleted = True
      self.state = self.material.deplete(self.config)
      self.material_id.update(self.state.index)

    return self.material.harvest()
