import unittest

import nmmo

class TestConfig(unittest.TestCase):
  def test_config_attr_set_episode(self):
    config = nmmo.config.Default()
    self.assertEqual(config.RESOURCE_SYSTEM_ENABLED, True)

    config.set_for_episode("RESOURCE_SYSTEM_ENABLED", False)
    self.assertEqual(config.RESOURCE_SYSTEM_ENABLED, False)

    config.reset()
    self.assertEqual(config.RESOURCE_SYSTEM_ENABLED, True)

if __name__ == '__main__':
  unittest.main()
