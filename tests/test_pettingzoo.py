import unittest
from pettingzoo.test import parallel_api_test

import nmmo
from scripted import baselines


class TestPettingZoo(unittest.TestCase):
  def test_pettingzoo_api(self):
    config = nmmo.config.Default()
    config.set("PLAYERS", [baselines.Random])
    config.set("HORIZON", 290)
    env = nmmo.Env(config)
    parallel_api_test(env, num_cycles=300)

if __name__ == "__main__":
  unittest.main()
