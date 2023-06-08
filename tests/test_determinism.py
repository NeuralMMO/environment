import unittest
import random
import numpy as np
from tqdm import tqdm

from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv
from tests.testhelpers import observations_are_equal

# 30 seems to be enough to test variety of agent actions
TEST_HORIZON = 30
RANDOM_SEED = random.randint(0, 100000)


def rollout_with_seed(env, seed):
  init_obs = env.reset(seed=seed)
  for _ in tqdm(range(TEST_HORIZON)):
    obs, _, _, _ = env.step({})
  event_log = env.realm.event_log.get_data()

  return init_obs, obs, event_log

class TestDeterminism(unittest.TestCase):
  def test_single_proc(self):
    config = ScriptedAgentTestConfig()
    env = ScriptedAgentTestEnv(config)

    # the source run
    init_obs_src, final_obs_src, event_log_src = rollout_with_seed(env, RANDOM_SEED)

    # the replication run
    init_obs_rep, final_obs_rep, event_log_rep = rollout_with_seed(env, RANDOM_SEED)

    # sanity checks
    self.assertTrue(observations_are_equal(init_obs_src, init_obs_src))
    self.assertTrue(observations_are_equal(final_obs_src, final_obs_src))

    # pylint: disable=expression-not-assigned
    # compare the source and replication
    self.assertTrue(observations_are_equal(init_obs_src, init_obs_rep)),\
      f"The determinism test failed. Seed: {RANDOM_SEED}."
    self.assertTrue(observations_are_equal(final_obs_src, final_obs_rep)),\
      f"The determinism test failed. Seed: {RANDOM_SEED}." # after 30 runs
    assert np.array_equal(event_log_src, event_log_rep),\
      f"The determinism test failed. Seed: {RANDOM_SEED}."

  def test_realm_level_rng(self):
    # the below test doesn't work now
    # having a realm-level random number generator would fix this
    # for example see https://github.com/openai/gym/pull/135/files
    #   how self.np_random is initialized and used
    pass

    # config = ScriptedAgentTestConfig()
    # env1 = ScriptedAgentTestEnv(config)
    # env2 = ScriptedAgentTestEnv(config)
    # envs = [env1, env2]

    # init_obs = [env.reset(seed=RANDOM_SEED) for env in envs]

    # for _ in tqdm(range(TEST_HORIZON)):
    #   # step returns a tuple of (obs, rewards, dones, infos)
    #   step_results = [env.step({}) for env in envs]

    # event_logs = [env.realm.event_log.get_data() for env in envs]

    # self.assertTrue(observations_are_equal(init_obs[0], init_obs[1])),\
    #   f"The multi-env determinism failed. Seed: {RANDOM_SEED}."
    # self.assertTrue(observations_are_equal(step_results[0][0], step_results[1][0])),\
    #   f"The multi-env determinism failed. Seed: {RANDOM_SEED}." # after 30 runs
    # assert np.array_equal(event_logs[0], event_logs[1]),\
    #   f"The multi-env determinism failed. Seed: {RANDOM_SEED}."


if __name__ == '__main__':
  unittest.main()
