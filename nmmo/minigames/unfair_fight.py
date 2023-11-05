# pylint: disable=duplicate-code
from nmmo.core.game_api import TeamBattle
from nmmo.task import task_spec, base_predicates
from nmmo.lib import utils, team_helper

TIME_LIMIT = 160  # ticks, arbitrary

def survival_task(def_over_off_ratio):
  multiplier = min(4.0, 1.0/def_over_off_ratio)  # smaller team gets higher reward
  return task_spec.TaskSpec(
    eval_fn=base_predicates.TickGE,
    eval_fn_kwargs={"num_tick": TIME_LIMIT},
    task_kwargs={"reward_multiplier": multiplier},
    reward_to="team")

elimination_task = task_spec.TaskSpec(
  eval_fn=base_predicates.CheckAgentStatus,
  eval_fn_kwargs={"target": "all_foes", "status": "dead"},
  reward_to="team")


class UnfairFight(TeamBattle):
  required_systems = ["TERRAIN", "COMBAT"]
  enable_death_fog = True
  time_limit = TIME_LIMIT

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self._defense_size = 4  # determines the difficulty
    self.adaptive_difficulty = True
    self.step_size = 4

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but huge
    self.task_embedding = utils.get_hash_embedding(lambda: [survival_task, elimination_task],
                                                   self.config.TASK_EMBED_DIM)

  @property
  def defense_size(self):
    return self._defense_size

  def set_defense_size(self, defense_size):
    self._defense_size = defense_size

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    assert self.defense_size <= self.config.PLAYER_N//2, \
      f"self.defense_size({self.defense_size}) must be <= {self.config.PLAYER_N//2}"
    super().reset(np_random, map_dict)
    self.history[-1]["defense_size"] = self.defense_size
    self.history[-1]["def_over_off_ratio"] = self.def_over_off_ratio

  @property
  def teams(self):
    return {"defense": list(range(1, self.defense_size+1)),
            "offense": list(range(self.defense_size+1, self.config.PLAYER_N+1)),}

  @property
  def def_over_off_ratio(self):
    return self.defense_size / (self.config.PLAYER_N-self.defense_size)

  def _set_config(self):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)

    # Make the map small
    self.config.set_for_episode("MAP_CENTER", 24)

    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.1)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.9)  # prop of stone tiles: 0.05

    # Activate death fog
    self.config.set_for_episode("PLAYER_DEATH_FOG", 32 if self.enable_death_fog else None)
    self.config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/6)
    # Only the center tile is safe
    self.config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", 8)

    # Disable +1 hp per tick
    self.config.set_for_episode("PLAYER_HEALTH_INCREMENT", False)

    self._determine_difficulty()  # sets the map_center
    self.config.set_for_episode("TEAMS", self.teams)

  def _determine_difficulty(self):
    # Determine the difficulty (the defense size) based on the previous results: 1 up - 1 down
    if self.adaptive_difficulty and self.history and self.history[-1]["result"]:
      # agent 1 always play the defense team
      if 1 in self.history[-1]["winners"]:
        # if the defense won, decrease the defense size
        self._defense_size = max(self.defense_size - self.step_size, self.step_size)
      else:
        # if the offense won, increase the defense size
        self._defense_size = min(self.defense_size + self.step_size, self.config.PLAYER_N//2)

  def _define_tasks(self, np_random):
    return task_spec.make_task_from_spec(
      self.teams,  # in the order of defense, offense
      [survival_task(self.def_over_off_ratio), elimination_task])

  def _set_realm(self, np_random, map_dict):
    self.realm.reset(np_random, map_dict, custom_spawn=True)
    center = self.config.MAP_SIZE // 2
    radius = self.config.PLAYER_VISION_RADIUS
    # Custom spawning: candidate_locs should be a list of list of (row, col) tuples
    r_offset = np_random.integers(radius-2, radius+3)
    c_offset = np_random.integers(radius-2, radius+3)
    candidate_locs = [[(center-r_offset, center-c_offset)],
                      [(center+r_offset, center+c_offset)]]
    np_random.shuffle(candidate_locs)
    # Also, one should make sure these locations are spawnable
    for loc_list in candidate_locs:
      for loc in loc_list:
        self.realm.map.make_spawnable(*loc)
    team_loader = team_helper.TeamLoader(self.config, np_random, candidate_locs)
    self.realm.players.spawn(team_loader)

  @property
  def winning_score(self):
    # sum of the difficulty, the speed
    if self._winners:
      alive_members = sum(1.0 for agent_id in self._winners if agent_id in self.realm.players)\
                      / len(self._winners)
      difficulty = 1 / self.def_over_off_ratio if self._winners[0] in self.teams["defense"]\
                    else self.def_over_off_ratio
      # NOTE: speed bonus is only for the offense team
      speed_bonus = (self.time_limit - self.realm.tick) / self.time_limit\
                      if self._winners[0] in self.teams["offense"] else 0.0
      return difficulty + 3*speed_bonus + 0.5*alive_members  # prioritize speed
    # No one reached the center
    return 0.0

  @staticmethod
  def test(env, horizon=30):
    game = UnfairFight(env)
    env.reset(game=game)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is True
    assert config.ITEM_SYSTEM_ENABLED is False
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False

    for _ in range(horizon):
      env.step({})

    # Check the tasks
    for eid in game.teams["defense"]:
      assert "TickGE" in env.agent_task_map[eid][0].name,\
        "TickGE must be assigned to the defense team"
    for eid in game.teams["offense"]:
      assert "CheckAgentStatus" in env.agent_task_map[eid][0].name,\
        "CheckAgentStatus must be assigned to the offense team"

    # pylint: disable=protected-access
    # Test if the difficulty increases
    org_def_size = game.defense_size
    game.history.append({"result": True, "winners": [60]})  # the offense won
    game._determine_difficulty()
    assert game.defense_size == (org_def_size + game.step_size)

    game.history.append({"result": True, "winners": [1]})  # the defense won
    game._determine_difficulty()
    assert game.defense_size == org_def_size

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(test_config)
  UnfairFight.test(test_env)
