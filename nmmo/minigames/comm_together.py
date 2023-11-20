# pylint: disable=duplicate-code, invalid-name
from nmmo.core.game_api import TeamBattle
from nmmo.task import task_spec
from nmmo.task.base_predicates import AllMembersWithinRange
from nmmo.lib import utils, team_helper


def seek_task(within_dist):
  return task_spec.TaskSpec(
    eval_fn=AllMembersWithinRange,
    eval_fn_kwargs={"dist": within_dist},
    reward_to="team")

class CommTogether(TeamBattle):
  required_systems = ["TERRAIN", "COMBAT", "COMMUNICATION"]

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    # NOTE: all members should fit in 5x5 square.
    self.team_within_dist = 5  # gather all team members within this distance

    self._map_size = 32  # determines the difficulty
    self._spawn_immunity = 480  # so that agents can start to attack each other later
    self.adaptive_difficulty = True
    self.num_game_won = 1  # at the same map size, threshold to increase the difficulty
    self.step_size = 8
    self._grass_map = False
    self.num_player_resurrect = 0

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but heavy
    self.task_embedding = utils.get_hash_embedding(seek_task, self.config.TASK_EMBED_DIM)

  @property
  def map_size(self):
    return self._map_size

  def set_map_size(self, map_size):
    self._map_size = map_size

  def set_spawn_immunity(self, spawn_immunity):
    self._spawn_immunity = spawn_immunity

  def set_grass_map(self, grass_map):
    self._grass_map = grass_map

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    super().reset(np_random, map_dict)
    self.history[-1]["map_size"] = self.map_size
    self._grass_map = False  # reset to default
    self.num_player_resurrect = 0

  def _set_config(self, np_random):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)
    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.1)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.9)
    self.config.set_for_episode("TERRAIN_RESET_TO_GRASS", self._grass_map)
    # NO death fog
    self.config.set_for_episode("DEATH_FOG_ONSET", None)
    # Enable +1 hp per tick
    self.config.set_for_episode("PLAYER_HEALTH_INCREMENT", True)
    # Make the attacks weaker
    self.config.set_for_episode("COMBAT_MELEE_DAMAGE", 5)
    self.config.set_for_episode("COMBAT_RANGE_DAMAGE", 5)
    self.config.set_for_episode("COMBAT_MAGE_DAMAGE", 5)

    self._determine_difficulty()  # sets the map size
    self.config.set_for_episode("MAP_CENTER", self.map_size)
    self.config.set_for_episode("COMBAT_SPAWN_IMMUNITY", self._spawn_immunity)

  def _determine_difficulty(self):
    # Determine the difficulty (the map size) based on the previous results
    if self.adaptive_difficulty and self.history \
       and self.history[-1]["result"]:  # the last game was won
      last_results = [r["result"] for r in self.history if r["map_size"] == self.map_size]
      if sum(last_results) >= self.num_game_won:
        self._map_size = min(self.map_size + self.step_size,
                             self.config.original["MAP_CENTER"])
        # Decrease the spawn immunity, to increase attack window
        if self._spawn_immunity > self.history[-1]["winning_tick"]:
          next_immunity = (self._spawn_immunity + self.history[-1]["winning_tick"]) / 2
          self._spawn_immunity = max(next_immunity, 64)  # 64 is the minimum

  def _set_realm(self, np_random, map_dict):
    # NOTE: this game respawns dead players at the edge, so setting delete_dead_entity=False
    self.realm.reset(np_random, map_dict, delete_dead_player=False)

  def _define_tasks(self, np_random):
    spec_list = [seek_task(self.team_within_dist)] * len(self.teams)
    return task_spec.make_task_from_spec(self.teams, spec_list)

  def _process_dead_players(self, dones, dead_players):
    # Respawn dead players at the edge
    for player in dead_players.values():
      player.resurrect(freeze_duration=30, health_prop=1)
      self.num_player_resurrect += 1

  def _check_winners(self, dones):
    # No winner game is possible
    return self._who_completed_task()

  @property
  def winning_score(self):
    if self._winners:
      time_limit = self.config.HORIZON
      speed_bonus = (time_limit - self.realm.tick) / time_limit
      return speed_bonus
    return 0.0

  @staticmethod
  def test(env, horizon=30):
    game = CommTogether(env)
    env.reset(game=game)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is True
    assert config.DEATH_FOG_ONSET is None
    assert config.ITEM_SYSTEM_ENABLED is False
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False

    for _ in range(horizon):
      env.step({})

    # pylint: disable=protected-access
    # These should run without errors
    game.history.append({"result": False, "map_size": 0, "winning_tick": 512})
    game._determine_difficulty()
    game.history.append({"result": True, "winners": None, "map_size": 0, "winning_tick": 512})
    game._determine_difficulty()

    # Test if the difficulty changes
    org_map_size = game.map_size
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append({"result": result, "map_size": game.map_size, "winning_tick": 128})
      game._determine_difficulty()  # pylint: disable=protected-access
    assert game.map_size == (org_map_size + game.step_size)

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  teams = team_helper.make_teams(test_config, num_teams=7)
  test_config.set("TEAMS", teams)
  test_env = nmmo.Env(test_config)
  CommTogether.test(test_env)

  # performance test
  # from tests.testhelpers import profile_env_step
  # tasks = task_spec.make_task_from_spec(teams, [seek_task(5)]*len(teams))
  # profile_env_step(tasks=tasks)
