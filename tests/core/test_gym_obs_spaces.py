import unittest

import nmmo

class TestGymObsSpaces(unittest.TestCase):
  def _test_gym_obs_space(self, env):
    obs_spec = env.observation_space(1)
    obs, _, _, _ = env.step({})

    for agent_obs in obs.values():
      for key, val in agent_obs.items():
        if key != 'ActionTargets':
          self.assertTrue(obs_spec[key].contains(val),
                          f"Invalid obs format -- key: {key}, val: {val}")

      if 'ActionTargets' in agent_obs:
        val = agent_obs['ActionTargets']
        for atn in nmmo.Action.edges(env.config):
          if atn.enabled(env.config):
            for arg in atn.edges: # pylint: disable=not-an-iterable
              mask_spec = obs_spec['ActionTargets'][atn.__name__][arg.__name__]
              mask_val = val[atn.__name__][arg.__name__]
              self.assertTrue(mask_spec.contains(mask_val),
                              "Invalid obs format -- " + \
                              f"key: {atn.__name__}/{arg.__name__}, val: {mask_val}")

  def test_env_without_noop(self):
    config = nmmo.config.Default()
    config.PROVIDE_NOOP_ACTION_TARGET = False
    env = nmmo.Env(config)
    env.reset(seed=1)
    for _ in range(3):
      env.step({})

    self._test_gym_obs_space(env)

  def test_env_with_noop(self):
    config = nmmo.config.Default()
    config.PROVIDE_NOOP_ACTION_TARGET = True
    env = nmmo.Env(config)
    env.reset(seed=1)
    for _ in range(3):
      env.step({})

    self._test_gym_obs_space(env)

if __name__ == '__main__':
  unittest.main()
