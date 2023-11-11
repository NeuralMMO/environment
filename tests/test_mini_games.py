# pylint: disable=protected-access
import unittest
import nmmo
from nmmo import minigames as mg
from nmmo.lib import team_helper

TEST_HORIZON = 30


class TestMinigames(unittest.TestCase):
  def test_mini_games(self):
    config = nmmo.config.Default()
    config.set("TEAMS", team_helper.make_teams(config, num_teams=16))
    env = nmmo.Env(config)

    for game_cls in [mg.UnfairFight, mg.RacetoCenter, mg.KingoftheHill, mg.Sandwich]:
      game = game_cls(env)
      env.reset(game=game)
      game.test(env, TEST_HORIZON)

if __name__ == "__main__":
  unittest.main()
