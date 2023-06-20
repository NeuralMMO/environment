import unittest

import nmmo

class TestGymSpaces(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = nmmo.config.Default()
    cls.env = nmmo.Env(cls.config)
    cls.env.reset(seed=1)
    for _ in range(3):
      cls.env.step({})

  def test_obs_space(self):
    obs_spec = self.env.observation_space(1)
    obs, _, _, _ = self.env.step({})

    for agent_obs in obs.values():
      for key, val in agent_obs.items():
        if key != 'ActionTargets':
          self.assertTrue(obs_spec[key].contains(val),
                          f"Invalid obs format -- key: {key}, val: {val}")

      if 'ActionTargets' in agent_obs:
        val = agent_obs['ActionTargets']
        for atn in nmmo.Action.edges(self.config):
          if atn.enabled(self.config):
            for arg in atn.edges: # pylint: disable=not-an-iterable
              self.assertTrue(obs_spec['ActionTargets'][atn][arg].contains(val[atn][arg]),
                              f"Invalid obs format -- key: {atn}/{arg}, val: {val[atn][arg]}")


if __name__ == '__main__':
  unittest.main()
