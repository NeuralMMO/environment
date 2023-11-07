from typing import List, Tuple
import numpy as np
from ordered_set import OrderedSet

from nmmo.core.tile import Tile
from nmmo.lib import material, utils
from nmmo.core.terrain import (
  fractal_to_material,
  process_map_border,
  spawn_profession_resources,
  scatter_extra_resources,
)


class Map:
  '''Map object representing a list of tiles

  Also tracks a sparse list of tile updates
  '''
  def __init__(self, config, realm, np_random):
    self.config = config
    self._repr  = None
    self.realm  = realm
    self.update_list = None
    self.pathfinding_cache = {} # Avoid recalculating A*, paths don't move

    sz          = config.MAP_SIZE
    self.tiles  = np.zeros((sz,sz), dtype=object)
    self.habitable_tiles = np.zeros((sz,sz))

    for r in range(sz):
      for c in range(sz):
        self.tiles[r, c] = Tile(realm, r, c, np_random)

    # the map center, and the centers in each quadrant are important targets
    self.dist_border_center = None
    self.center_coord = None
    self.quad_centers = None
    self.seize_targets: List[Tuple] = None  # a list of (r, c) coords

    # used to place border
    self.l1 = utils.l1_map(sz)

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

  def reset(self, map_dict, np_random, seize_targets=None):
    '''Reuse the current tile objects to load a new map'''
    config = self.config
    assert map_dict["map"].shape == (config.MAP_SIZE,config.MAP_SIZE),\
      "Map shape is inconsistent with config.MAP_SIZE"

    # NOTE: MAP_CENTER and MAP_BORDER can change from episode to episode
    self.center_coord = (config.MAP_SIZE//2, config.MAP_SIZE//2)
    self.dist_border_center = config.MAP_CENTER // 2
    half_dist = self.dist_border_center // 2
    self.quad_centers = {
      "first": (self.center_coord[0] + half_dist, self.center_coord[1] + half_dist),
      "second": (self.center_coord[0] - half_dist, self.center_coord[1] + half_dist),
      "third": (self.center_coord[0] - half_dist, self.center_coord[1] - half_dist),
      "fourth": (self.center_coord[0] + half_dist, self.center_coord[1] - half_dist),
    }
    assert config.MAP_BORDER > config.PLAYER_VISION_RADIUS,\
      "MAP_BORDER must be greater than PLAYER_VISION_RADIUS"

    self._repr = None
    self.update_list = OrderedSet() # critical for determinism
    self.seize_targets = []
    if seize_targets:
      assert isinstance(seize_targets, list), "seize_targets must be a list of reserved words"
      for target in seize_targets:
        # pylint: disable=consider-iterating-dictionary
        assert target in list(self.quad_centers.keys()) + ["center"], "Invalid seize target"
        self.seize_targets.append(self.center_coord if target == "center"
                                  else self.quad_centers[target])

    # process map_np_array according to config
    matl_map = self._process_map(map_dict, np_random)
    if "mark_center" in map_dict and map_dict["mark_center"]:
      self._mark_tile(matl_map, *self.center_coord)
    for r, c in self.seize_targets:
      self._mark_tile(matl_map, r, c)

    # reset tiles with new materials
    materials = {mat.index: mat for mat in material.All}
    for r, row in enumerate(matl_map):
      for c, idx in enumerate(row):
        mat = materials[idx]
        tile = self.tiles[r, c]
        tile.reset(mat, config, np_random)
        self.habitable_tiles[r, c] = tile.habitable

  def _process_map(self, map_dict, np_random):
    map_np_array = map_dict["map"]
    if not self.config.TERRAIN_SYSTEM_ENABLED:
      map_np_array[:] = material.Grass.index
    else:
      if self.config.MAP_RESET_FROM_FRACTAL:
        map_tiles = fractal_to_material(self.config, map_dict["fractal"],
                                        self.config.TERRAIN_RESET_TO_GRASS)
        # Place materials here, before converting map_tiles into an int array
        if self.config.PROFESSION_SYSTEM_ENABLED:
          spawn_profession_resources(self.config, map_tiles, np_random)
        if self.config.TERRAIN_SCATTER_EXTRA_RESOURCES:
          scatter_extra_resources(self.config, map_tiles, np_random)
        map_np_array = map_tiles.astype(int)

      # Disable materials here
      if self.config.TERRAIN_DISABLE_STONE:
        map_np_array[map_np_array == material.Stone.index] = material.Grass.index

    # Make the edge tiles habitable, and place the void tiles outside the border
    map_np_array = process_map_border(self.config, map_np_array, self.l1)
    return map_np_array

  @staticmethod
  def _mark_tile(map_np_array, row, col, dist=2):
    map_np_array[row-dist:row+dist+1,col-dist:col+dist+1] = material.Grass.index
    map_np_array[row,col] = material.Herb.index

  def step(self):
    '''Evaluate updatable tiles'''
    for tile in self.update_list.copy():
      if not tile.depleted:
        self.update_list.remove(tile)
      tile.step()
    if self.seize_targets:
      for r, c in self.seize_targets:
        self.tiles[r, c].update_seize()

  def harvest(self, r, c, deplete=True):
    '''Called by actions that harvest a resource tile'''
    if deplete:
      self.update_list.add(self.tiles[r, c])
    return self.tiles[r, c].harvest(deplete)

  def is_valid_pos(self, row, col):
    '''Check if a position is valid'''
    return 0 <= row < self.config.MAP_SIZE and 0 <= col < self.config.MAP_SIZE

  def make_spawnable(self, row, col, radius=2):
    '''Make the area centered around row, col spawnable'''
    assert self._repr is None, "Cannot make spawnable after map is generated"
    assert radius > 0, "Radius must be positive"
    assert self.config.MAP_BORDER < row-radius and self.config.MAP_BORDER < col-radius \
           and row+radius < self.config.MAP_SIZE-self.config.MAP_BORDER \
           and col+radius < self.config.MAP_SIZE-self.config.MAP_BORDER,\
            "Cannot make spawnable near the border"
    for r in range(row-radius, row+radius+1):
      for c in range(col-radius, col+radius+1):
        tile = self.tiles[r, c]
        # pylint: disable=protected-access
        tile.reset(material.Grass, self.config, self.realm._np_random)
        self.habitable_tiles[r, c] = tile.habitable  # must be true

  @property
  def seize_status(self):
    if self.seize_targets is None:
      return {}
    return {
      (r, c): self.tiles[r, c].seize_history[-1]
      for r, c in self.seize_targets
      if self.tiles[r, c].seize_history
    }
