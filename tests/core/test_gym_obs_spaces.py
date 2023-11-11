import unittest
from copy import deepcopy

import nmmo
from nmmo.core.game_api import DefaultGame


class TestGymObsSpaces(unittest.TestCase):
  def _is_obs_valid(self, obs_spec, obs):
    for agent_obs in obs.values():
      for key, val in agent_obs.items():
        self.assertTrue(obs_spec[key].contains(val),
                        f"Invalid obs format -- key: {key}, val: {val}")

  def _test_gym_obs_space(self, env):
    obs_spec = env.observation_space(1)
    obs, _, _, _ = env.step({})
    self._is_obs_valid(obs_spec, obs)
    for agent_obs in obs.values():
      if "ActionTargets" in agent_obs:
        val = agent_obs["ActionTargets"]
        for atn in nmmo.Action.edges(env.config):
          if atn.enabled(env.config):
            for arg in atn.edges: # pylint: disable=not-an-iterable
              mask_spec = obs_spec["ActionTargets"][atn.__name__][arg.__name__]
              mask_val = val[atn.__name__][arg.__name__]
              self.assertTrue(mask_spec.contains(mask_val),
                              "Invalid obs format -- " + \
                              f"key: {atn.__name__}/{arg.__name__}, val: {mask_val}")
    return obs

  def test_env_without_noop(self):
    config = nmmo.config.Default()
    config.set("PROVIDE_NOOP_ACTION_TARGET", False)
    env = nmmo.Env(config)
    env.reset(seed=1)
    for _ in range(3):
      env.step({})

    self._test_gym_obs_space(env)

  def test_env_with_noop(self):
    config = nmmo.config.Default()
    config.set("PROVIDE_NOOP_ACTION_TARGET", True)
    env = nmmo.Env(config)
    env.reset(seed=1)
    for _ in range(3):
      env.step({})

    self._test_gym_obs_space(env)

  def test_env_with_fogmap(self):
    config = nmmo.config.Default()
    config.set("PROVIDE_DEATH_FOG_OBS", True)
    env = nmmo.Env(config)
    env.reset(seed=1)
    for _ in range(3):
      env.step({})

    self._test_gym_obs_space(env)

  def test_system_disable(self):
    class CustomGame(DefaultGame):
      def _set_config(self, np_random):
        self.config.reset()
        self.config.set_for_episode("COMBAT_SYSTEM_ENABLED", False)
        self.config.set_for_episode("ITEM_SYSTEM_ENABLED", False)
        self.config.set_for_episode("EXCHANGE_SYSTEM_ENABLED", False)
        self.config.set_for_episode("COMMUNICATION_SYSTEM_ENABLED", False)

    config = nmmo.config.Default()
    env = nmmo.Env(config)

    # test the default game
    env.reset()
    for _ in range(3):
      env.step({})
    self._test_gym_obs_space(env)
    org_obs_spec = deepcopy(env.observation_space(1))

    # test the custom game
    game = CustomGame(env)
    env.reset(game=game)
    for _ in range(3):
      env.step({})
    new_obs = self._test_gym_obs_space(env)

    # obs format must match between episodes
    self._is_obs_valid(org_obs_spec, new_obs)

    # check if the combat system is disabled
    for agent_obs in new_obs.values():
      self.assertEqual(sum(agent_obs["ActionTargets"]["Attack"]["Target"]),
                       int(config.PROVIDE_NOOP_ACTION_TARGET))


if __name__ == "__main__":
  unittest.main()
