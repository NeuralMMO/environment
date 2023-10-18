import unittest

import nmmo
import nmmo.core.config as cfg


class Config(cfg.Config, cfg.Terrain, cfg.Combat):
  pass

class TestConfig(unittest.TestCase):
  def test_config_attr_set_episode(self):
    config = nmmo.config.Default()
    self.assertEqual(config.RESOURCE_SYSTEM_ENABLED, True)

    config.set_for_episode("RESOURCE_SYSTEM_ENABLED", False)
    self.assertEqual(config.RESOURCE_SYSTEM_ENABLED, False)

    config.reset()
    self.assertEqual(config.RESOURCE_SYSTEM_ENABLED, True)

  def test_cannot_change_immutable_attr(self):
    config = Config()
    with self.assertRaises(AssertionError):
      config.set_for_episode("PLAYER_N", 100)

  def test_cannot_change_obs_attr(self):
    config = Config()
    with self.assertRaises(AssertionError):
      config.set_for_episode("PLAYER_N_OBS", 50)

  def test_cannot_use_noninit_system(self):
    config = Config()
    with self.assertRaises(AssertionError):
      config.set_for_episode("ITEM_SYSTEM_ENABLED", True)

if __name__ == '__main__':
  unittest.main()
