# pylint: disable=protected-access
import unittest
import nmmo
from nmmo import minigames as mg

HORIZON = 30


class TestMinigames(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = nmmo.config.Default()
    cls.env = nmmo.Env(cls.config)

  def test_center_race(self):
    game = mg.RacetoCenter(self.env)
    self.env.reset(game=game)

    # Check configs
    config = self.env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is False
    assert config.ITEM_SYSTEM_ENABLED is False
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False
    assert config.PLAYER_DEATH_FOG == 32

    for _ in range(HORIZON):
      self.env.step({})

    # Test if the difficulty increases
    org_map_center = game.map_center
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append({"result": result, "map_center": game.map_center})
      game._determine_difficulty()
    assert game.map_center == (org_map_center + game.step_size)

  def test_unfair_fight(self):
    game = mg.UnfairFight(self.env)
    self.env.reset(game=game)

    # Check configs
    config = self.env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is True
    assert config.TERRAIN_SYSTEM_ENABLED is False
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False
    assert config.PLAYER_DEATH_FOG == 32

    for _ in range(HORIZON):
      self.env.step({})

    # Check the tasks
    for eid in game.teams["defense"]:
      assert "TickGE" in self.env.agent_task_map[eid][0].name,\
        "TickGE must be assigned to the defense team"
    for eid in game.teams["offense"]:
      assert "CheckAgentStatus" in self.env.agent_task_map[eid][0].name,\
        "CheckAgentStatus must be assigned to the offense team"

    # Test if the difficulty increases
    org_def_size = game.defense_size
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append({"result": result, "defense_size": game.defense_size})
      game._determine_difficulty()  # pylint: disable=protected-access
    assert game.defense_size == (org_def_size + game.step_size)


if __name__ == "__main__":
  unittest.main()
