import unittest

import copy
import nmmo
from scripted.baselines import Sleeper

HORIZON = 32


class TestTileProperty(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.config = nmmo.config.Default()
    cls.config.PLAYERS = [Sleeper]
    env = nmmo.Env(cls.config)
    env.reset()
    cls.start = copy.deepcopy(env.realm)
    for _ in range(HORIZON):
      env.step({})
    cls.end = copy.deepcopy(env.realm)

  # Test immutable invariants assumed for certain optimizations
  def test_fixed_habitability_passability(self):
    # Used in optimization with habitability lookup table
    start_habitable = [tile.habitable for tile in self.start.map.tiles.flatten()]
    end_habitable = [tile.habitable for tile in self.end.map.tiles.flatten()]
    self.assertListEqual(start_habitable, end_habitable)

    # Used in optimization that caches the result of A*
    start_passable = [tile.impassible for tile in self.start.map.tiles.flatten()]
    end_passable = [tile.impassible for tile in self.end.map.tiles.flatten()]
    self.assertListEqual(start_passable, end_passable)

if __name__ == '__main__':
  unittest.main()
