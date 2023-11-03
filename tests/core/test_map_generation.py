# pylint: disable=protected-access
import unittest
import os
import shutil
import numpy as np

import nmmo
from nmmo.lib import material


class TestMapGeneration(unittest.TestCase):
  def test_insufficient_maps(self):
    config = nmmo.config.Small()
    config.set("PATH_MAPS", "maps/test_map_gen")
    config.set("MAP_N", 20)

    # clear the directory
    path_maps = os.path.join(config.PATH_CWD, config.PATH_MAPS)
    shutil.rmtree(path_maps, ignore_errors=True)

    # this generates 20 maps
    nmmo.Env(config)

    # test if MAP_FORCE_GENERATION can be overriden, when the maps are insufficient
    config2 = nmmo.config.Small()
    config2.set("PATH_MAPS", "maps/test_map_gen")  # the same map dir
    config2.set("MAP_N", 30)
    config2.set("MAP_FORCE_GENERATION", False)

    test_env = nmmo.Env(config2)
    test_env.reset(map_id=config.MAP_N)

    # this should finish without error

  def test_map_preview(self):
    class MapConfig(
      nmmo.config.Small, # no fractal, grass only
      nmmo.config.Terrain, # water, grass, foilage, stone
      nmmo.config.Item, # no additional effect on the map
      nmmo.config.Profession, # add ore, tree, crystal, herb, fish
    ):
      PATH_MAPS = 'maps/test_preview'
      MAP_FORCE_GENERATION = True
      MAP_GENERATE_PREVIEWS = True
    config = MapConfig()

    # clear the directory
    path_maps = os.path.join(config.PATH_CWD, config.PATH_MAPS)
    shutil.rmtree(path_maps, ignore_errors=True)

    nmmo.Env(config)

    # this should finish without error

  def test_map_reset_from_fractal(self):
    class MapConfig(
      nmmo.config.Small, # no fractal, grass only
      nmmo.config.Terrain, # water, grass, foilage, stone
      nmmo.config.Item, # no additional effect on the map
      nmmo.config.Profession, # add ore, tree, crystal, herb, fish
    ):
      PATH_MAPS = 'maps/test_fractal'
      MAP_FORCE_GENERATION = True
      MAP_RESET_FROM_FRACTAL = True
    config = MapConfig()
    self.assertEqual(config.MAP_SIZE, 64)
    self.assertEqual(config.MAP_CENTER, 32)

    # clear the directory
    path_maps = os.path.join(config.PATH_CWD, config.PATH_MAPS)
    shutil.rmtree(path_maps, ignore_errors=True)

    test_env = nmmo.Env(config)

    # the fractals should be saved
    fractal_file = os.path.join(path_maps, config.PATH_FRACTAL_SUFFIX.format(1))
    self.assertTrue(os.path.exists(fractal_file))

    config = test_env.config
    map_size = config.MAP_SIZE
    np_random = test_env._np_random

    # Return the Grass map
    config.set_for_episode("TERRAIN_SYSTEM_ENABLED", False)
    map_dict = test_env._load_map_file()
    map_array = test_env.realm.map._process_map(map_dict, np_random)
    self.assertEqual(np.sum(map_array == material.Void.index)+\
                     np.sum(map_array == material.Grass.index), map_size*map_size)
    # NOTE: +1 to make the center tile, really the center
    self.assertEqual((config.MAP_CENTER+1)**2, np.sum(map_array == material.Grass.index))

    # Another way to make the grass map (which can place other tiles, if want to)
    config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    config.set_for_episode("TERRAIN_RESET_TO_GRASS", True)
    config.set_for_episode("PROFESSION_SYSTEM_ENABLED", False)  # harvestalbe tiles
    config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", False)
    map_dict = test_env._load_map_file()
    map_array = test_env.realm.map._process_map(map_dict, np_random)
    self.assertEqual(np.sum(map_array == material.Void.index)+\
                     np.sum(map_array == material.Grass.index), map_size*map_size)
    # NOTE: +1 to make the center tile, really the center
    self.assertEqual((config.MAP_CENTER+1)**2, np.sum(map_array == material.Grass.index))

    # Generate from fractal, but not spawn profession tiles
    config.reset()
    config.set_for_episode("PROFESSION_SYSTEM_ENABLED", False)
    map_dict = test_env._load_map_file()
    map_array = test_env.realm.map._process_map(map_dict, np_random)
    self.assertEqual(np.sum(map_array == material.Void.index)+\
                     np.sum(map_array == material.Grass.index)+\
                     np.sum(map_array == material.Water.index)+\
                     np.sum(map_array == material.Stone.index)+\
                     np.sum(map_array == material.Foilage.index),
                     map_size*map_size - 1)  # -1 for the center tile, marked with Herb

    # Use the saved map, but disable stone
    config.reset()
    config.set_for_episode("MAP_RESET_FROM_FRACTAL", False)
    config.set_for_episode("TERRAIN_DISABLE_STONE", True)
    map_dict = test_env._load_map_file()
    org_map = map_dict["map"].copy()
    self.assertTrue("fractal" not in map_dict)
    map_array = test_env.realm.map._process_map(map_dict, np_random)
    self.assertTrue(np.sum(org_map == material.Stone.index) > 0)
    self.assertTrue(np.sum(map_array == material.Stone.index) == 0)

    # Generate from fractal, test add-on functions
    config.reset()
    config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    config.set_for_episode("PROFESSION_SYSTEM_ENABLED", True)
    config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", True)
    map_dict = test_env._load_map_file()
    map_array = test_env.realm.map._process_map(map_dict, np_random)

    # this should finish without error

if __name__ == '__main__':
  unittest.main()
