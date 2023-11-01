# pylint: disable=protected-access
import unittest
import nmmo
from nmmo import minigames as mg

TEST_HORIZON = 30


class TestMinigames(unittest.TestCase):
  def test_mini_games(self):
    config = nmmo.config.Default()
    env = nmmo.Env(config)

    for game_cls in [mg.UnfairFight, mg.RacetoCenter]:
      game = game_cls(env)
      env.reset(game=game)
      game.test(env, TEST_HORIZON)

if __name__ == "__main__":
  unittest.main()
