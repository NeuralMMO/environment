import unittest
import numpy as np
from tqdm import tqdm

from nmmo.lib import seeding
from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv
from tests.testhelpers import observations_are_equal

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = np.random.randint(0, 100000)


def rollout_with_seed(env, seed):
  init_obs = env.reset(seed=seed)
  for _ in tqdm(range(TEST_HORIZON)):
    obs, _, _, _ = env.step({})
  event_log = env.realm.event_log.get_data()

  return init_obs, obs, event_log

class TestDeterminism(unittest.TestCase):
  def test_gym_np_random(self):
    _, _np_seed_1 = seeding.np_random(RANDOM_SEED)
    _, _np_seed_2 = seeding.np_random(RANDOM_SEED)
    self.assertEqual(_np_seed_1, _np_seed_2)

  def test_env_level_rng(self):
    # two envs running independently should return the same results
    config = ScriptedAgentTestConfig()
    env1 = ScriptedAgentTestEnv(config)
    env2 = ScriptedAgentTestEnv(config)
    envs = [env1, env2]

    init_obs = [env.reset(seed=RANDOM_SEED) for env in envs]

    for _ in tqdm(range(TEST_HORIZON)):
      # step returns a tuple of (obs, rewards, dones, infos)
      step_results = [env.step({}) for env in envs]

    event_logs = [env.realm.event_log.get_data() for env in envs]

    # sanity checks
    self.assertTrue(observations_are_equal(init_obs[0], init_obs[0]))
    self.assertTrue(observations_are_equal(step_results[0][0], step_results[0][0]))

    self.assertTrue(observations_are_equal(init_obs[0], init_obs[1]),
                    f"The multi-env determinism failed. Seed: {RANDOM_SEED}.")
    self.assertTrue(observations_are_equal(step_results[0][0], step_results[1][0]),
                    f"The multi-env determinism failed. Seed: {RANDOM_SEED}.") # after 30 runs
    self.assertTrue(np.array_equal(event_logs[0], event_logs[1]),
                    f"The multi-env determinism failed. Seed: {RANDOM_SEED}.")


if __name__ == '__main__':
  unittest.main()
