# pylint: disable=protected-access
import unittest
import numpy as np
import nmmo
from nmmo import minigames as mg
from nmmo.lib import team_helper

TEST_HORIZON = 10


class TestMinigames(unittest.TestCase):
  def test_mini_games(self):
    config = nmmo.config.Default()
    config.set("TEAMS", team_helper.make_teams(config, num_teams=16))
    env = nmmo.Env(config)

    for game_cls in mg.AVAILABLE_GAMES:
      game = game_cls(env)
      env.reset(game=game)
      game.test(env, TEST_HORIZON)

      # Check if the gym_obs is correctly set, on alive agents
      for agent_id in env.realm.players:
        gym_obs = env.obs[agent_id].to_gym()
        self.assertEqual(gym_obs["AgentId"], agent_id)
        self.assertEqual(gym_obs["CurrentTick"], env.realm.tick)
        self.assertTrue(
          np.array_equal(gym_obs["Task"], env.agent_task_map[agent_id][0].embedding))

if __name__ == "__main__":
  unittest.main()
