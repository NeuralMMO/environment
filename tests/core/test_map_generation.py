import unittest
import os
import shutil

import nmmo

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

    # test if MAP_FORCE_GENERATION can be overriden
    config.MAP_N = 30
    config.MAP_FORCE_GENERATION = False

    test_env = nmmo.Env(config)
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

    test_env = nmmo.Env(config) # pylint: disable=unused-variable

    # this should finish without error

if __name__ == '__main__':
  unittest.main()
