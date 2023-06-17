import os
import logging
import numpy as np
from ordered_set import OrderedSet

from nmmo.core.tile import Tile
from nmmo.lib import material


class Map:
  '''Map object representing a list of tiles

  Also tracks a sparse list of tile updates
  '''
  def __init__(self, config, realm):
    self.config = config
    self._repr  = None
    self.realm  = realm
    self.update_list = None

    sz          = config.MAP_SIZE
    self.tiles  = np.zeros((sz, sz), dtype=object)

    for r in range(sz):
      for c in range(sz):
        self.tiles[r, c] = Tile(realm, r, c)

  @property
  def packet(self):
    '''Packet of degenerate resource states'''
    missing_resources = []
    for e in self.update_list:
      missing_resources.append(e.pos)
    return missing_resources

  @property
  def repr(self):
    '''Flat matrix of tile material indices'''
    if not self._repr:
      self._repr = [[t.material.index for t in row] for row in self.tiles]

    return self._repr

  def reset(self, map_id):
    '''Reuse the current tile objects to load a new map'''
    config = self.config
    self.update_list = OrderedSet() # critical for determinism

    path_map_suffix = config.PATH_MAP_SUFFIX.format(map_id)
    f_path = os.path.join(config.PATH_CWD, config.PATH_MAPS, path_map_suffix)

    try:
      map_file = np.load(f_path)
    except FileNotFoundError:
      logging.error('Maps not found')
      raise

    materials = {mat.index: mat for mat in material.All}
    r, c = 0, 0
    for r, row in enumerate(map_file):
      for c, idx in enumerate(row):
        mat  = materials[idx]
        tile = self.tiles[r, c]
        tile.reset(mat, config)

    assert c == config.MAP_SIZE - 1
    assert r == config.MAP_SIZE - 1

    self._repr = None

  def step(self):
    '''Evaluate updatable tiles'''
    self.realm.log_milestone('Resource_Depleted', len(self.update_list),
        f'RESOURCE: Depleted {len(self.update_list)} resource tiles')

    for e in self.update_list.copy():
      if not e.depleted:
        self.update_list.remove(e)
      e.step()

  def harvest(self, r, c, deplete=True):
    '''Called by actions that harvest a resource tile'''

    if deplete:
      self.update_list.add(self.tiles[r, c])

    return self.tiles[r, c].harvest(deplete)

  def is_valid_pos(self, row, col):
    '''Check if a position is valid'''
    return 0 <= row < self.config.MAP_SIZE and 0 <= col < self.config.MAP_SIZE
