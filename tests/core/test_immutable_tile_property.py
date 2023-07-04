# Test immutable invariants assumed for certain optimizations

import unittest

import copy
import nmmo
from scripted.baselines import Random

def rollout():
  config = nmmo.config.Default()
  config.PLAYERS = [Random]
  env = nmmo.Env(config)
  env.reset()
  start = copy.deepcopy(env.realm)
  for _ in range(64):
    env.step({})
  end = copy.deepcopy(env.realm)
  return (start, end)

class TestImmutableTileProperty(unittest.TestCase):

  def test_passability_immutable(self):
    # Used in optimization that caches the result of A*
    start, end = rollout()
    start_passable = [tile.impassible for tile in start.map.tiles.flatten()]
    end_passable = [tile.impassible for tile in end.map.tiles.flatten()]
    self.assertListEqual(start_passable, end_passable)

  def test_habitability_immutable(self):
    # Used in optimization with habitability lookup table
    start, end = rollout()
    start_habitable = [tile.habitable for tile in start.map.tiles.flatten()]
    end_habitable = [tile.habitable for tile in end.map.tiles.flatten()]
    self.assertListEqual(start_habitable, end_habitable)

if __name__ == '__main__':
  unittest.main()
