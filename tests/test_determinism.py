import unittest
import numpy as np
from tqdm import tqdm

import nmmo
from nmmo.lib import seeding
from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv
from tests.testhelpers import observations_are_equal

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = np.random.randint(0, 100000)


class TestDeterminism(unittest.TestCase):
  def test_gym_np_random(self):
    _, _np_seed_1 = seeding.np_random(RANDOM_SEED)
    _, _np_seed_2 = seeding.np_random(RANDOM_SEED)
    self.assertEqual(_np_seed_1, _np_seed_2)

  def test_map_determinism(self):
    config = nmmo.config.Default()
    config.MAP_FORCE_GENERATION = True
    config.TERRAIN_FLIP_SEED = False

    map_generator = config.MAP_GENERATOR(config)
    np_random1, _ = seeding.np_random(RANDOM_SEED)
    np_random1_1, _ = seeding.np_random(RANDOM_SEED)

    terrain1, tiles1 = map_generator.generate_map(0, np_random1)
    terrain1_1, tiles1_1 = map_generator.generate_map(0, np_random1_1)

    self.assertTrue(np.array_equal(terrain1, terrain1_1))
    self.assertTrue(np.array_equal(tiles1, tiles1_1))

    # test flip seed
    config2 = nmmo.config.Default()
    config2.MAP_FORCE_GENERATION = True
    config2.TERRAIN_FLIP_SEED = True

    map_generator2 = config2.MAP_GENERATOR(config2)
    np_random2, _ = seeding.np_random(RANDOM_SEED)
    terrain2, tiles2 = map_generator2.generate_map(0, np_random2)

    self.assertFalse(np.array_equal(terrain1, terrain2))
    self.assertFalse(np.array_equal(tiles1, tiles2))

  def test_env_level_rng(self):
    # two envs running independently should return the same results

    # config to always generate new maps, to test map determinism
    config1 = ScriptedAgentTestConfig()
    setattr(config1, 'MAP_FORCE_GENERATION', True)
    setattr(config1, 'PATH_MAPS', 'maps/det1')
    config2 = ScriptedAgentTestConfig()
    setattr(config2, 'MAP_FORCE_GENERATION', True)
    setattr(config2, 'PATH_MAPS', 'maps/det2')

    # to create the same maps, seed must be provided
    env1 = ScriptedAgentTestEnv(config1, seed=RANDOM_SEED)
    env2 = ScriptedAgentTestEnv(config2, seed=RANDOM_SEED)
    envs = [env1, env2]

    init_obs = [env.reset(seed=RANDOM_SEED+1) for env in envs]

    self.assertTrue(observations_are_equal(init_obs[0], init_obs[0])) # sanity check
    self.assertTrue(observations_are_equal(init_obs[0], init_obs[1]),
                    f"The multi-env determinism failed. Seed: {RANDOM_SEED}.")

    for _ in tqdm(range(TEST_HORIZON)):
      # step returns a tuple of (obs, rewards, dones, infos)
      step_results = [env.step({}) for env in envs]
      self.assertTrue(observations_are_equal(step_results[0][0], step_results[1][0]),
                      f"The multi-env determinism failed. Seed: {RANDOM_SEED}.")

    event_logs = [env.realm.event_log.get_data() for env in envs]
    self.assertTrue(np.array_equal(event_logs[0], event_logs[1]),
                    f"The multi-env determinism failed. Seed: {RANDOM_SEED}.")


if __name__ == '__main__':
  unittest.main()
