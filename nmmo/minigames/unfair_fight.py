# pylint: disable=duplicate-code, invalid-name
from nmmo.core.game_api import TeamBattle
from nmmo.task import task_spec
from nmmo.task.base_predicates import SeizeQuadCenter
from nmmo.lib import utils, team_helper, event_code


def SeizeBothQuads(gs, subject, num_ticks, quadrants):
  return 0.5*SeizeQuadCenter(gs, subject, num_ticks, quadrants[0]) +\
         0.5*SeizeQuadCenter(gs, subject, num_ticks, quadrants[1])

def sieze_task(num_ticks, quadrants):
  return task_spec.TaskSpec(
    eval_fn=SeizeBothQuads,
    eval_fn_kwargs={"num_ticks": num_ticks, "quadrants": quadrants},
    reward_to="team")

class UnfairFight(TeamBattle):
  required_systems = ["TERRAIN", "COMBAT"]

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self._small_team_size = 4  # determines the difficulty
    self.step_size = 4  # also the minimum team size
    self.adaptive_difficulty = True
    self.map_size = 48
    self._seize_duration = 30
    self._spawn_keys = None

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but heavy
    self.task_embedding = utils.get_hash_embedding(sieze_task, self.config.TASK_EMBED_DIM)

  @property
  def small_team_size(self):
    return self._small_team_size

  def set_small_team_size(self, team_size):
    self._small_team_size = team_size

  @property
  def seize_duration(self):
    return self._seize_duration

  def set_seize_duration(self, seize_duration):
    self._seize_duration = seize_duration

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    super().reset(np_random, map_dict)
    self.history[-1]["small_team_size"] = self.small_team_size

  @property
  def teams(self):
    return {"small": list(range(1, self.small_team_size+1)),
            "large": list(range(self.small_team_size+1, self.config.PLAYER_N+1)),}

  def _set_config(self):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("HORIZON", 256)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)

    # Make the map small
    self.config.set_for_episode("MAP_CENTER", self.map_size)

    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.1)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.9)  # prop of stone tiles: 0.05

    # NO death fog
    self.config.set_for_episode("PLAYER_DEATH_FOG", None)

    # Disable +1 hp per tick
    self.config.set_for_episode("PLAYER_HEALTH_INCREMENT", False)

    self._determine_difficulty()  # sets the small team size
    self.config.set_for_episode("TEAMS", self.teams)

  def _determine_difficulty(self):
    # Determine the difficulty (the small team size) based on the previous results
    # If there are no winners in the previous game, the difficulty is not changed
    if self.adaptive_difficulty and len(self.history) > 0 \
       and "winners" in self.history[-1] and self.history[-1]["winners"]:
      is_small_won = 1 in self.history[-1]["winners"]
      if is_small_won:
        self._small_team_size = max(self.small_team_size - self.step_size,
                                    self.step_size)
      else:
        self._small_team_size = min(self.small_team_size + self.step_size,
                                    self.config.PLAYER_N//2)

  def _set_realm(self, np_random, map_dict):
    spawn_keys = np_random.choice(["first", "second", "third", "fourth"], 2, replace=False)
    # Set the seize targets
    self.realm.reset(np_random, map_dict, custom_spawn=True,
                     seize_targets=list(spawn_keys))
    # Also, one should make sure these locations are spawnable
    spawn_locs = [self.realm.map.quad_centers[key] for key in spawn_keys]
    for loc in spawn_locs:
      self.realm.map.make_spawnable(*loc)
    team_loader = team_helper.TeamLoader(self.config, np_random, spawn_locs)
    self.realm.players.spawn(team_loader)
    # Use the other teams' spawn locations as the seize targets
    self._spawn_keys = {"small": spawn_keys[0], "large": spawn_keys[1]}

  def _define_tasks(self, np_random):
    task = sieze_task(self.seize_duration, list(self._spawn_keys.values()))
    return task_spec.make_task_from_spec(self.teams, [task]*2)

  def _check_winners(self, dones):
    # Since the goal is to seize tile for a certain duration, no winner game is possible
    return self._who_completed_task()

  @property
  def winning_score(self):
    if self._winners:
      kill_log = self.realm.event_log.get_data(
        agents=self._winners, event_code=event_code.EventCode.PLAYER_KILL)
      kill_bonus = kill_log.shape[0] / (self.config.PLAYER_N - len(self._winners))  # hacky
      time_limit = self.config.HORIZON
      speed_bonus = (time_limit - self.realm.tick) / time_limit
      return kill_bonus + speed_bonus
    return 0.0

  @staticmethod
  def test(env, horizon=30):
    game = UnfairFight(env)
    game.set_small_team_size(20)
    env.reset(game=game)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is True
    assert config.PLAYER_DEATH_FOG is None
    assert config.ITEM_SYSTEM_ENABLED is False
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False

    for _ in range(horizon):
      env.step({})

    # pylint: disable=protected-access
    # These should run without errors
    game.history.append({"result": False})
    game._determine_difficulty()
    game.history.append({"result": True, "winners": None})
    game._determine_difficulty()

    # Test if the difficulty changes
    org_small_team = game.small_team_size
    game.history.append({"result": True, "winners": [60]})  # large team won
    game._determine_difficulty()
    assert game.small_team_size == (org_small_team + game.step_size)

    game.history.append({"result": True, "winners": [1]})  # small team won
    game._determine_difficulty()
    assert game.small_team_size == org_small_team

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(test_config)
  UnfairFight.test(test_env)
