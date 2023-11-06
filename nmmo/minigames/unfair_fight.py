# pylint: disable=duplicate-code
import numpy as np
from nmmo.core.game_api import TeamBattle
from nmmo.task import task_spec
from nmmo.lib import utils, team_helper, event_code

# pylint: disable=invalid-name,unused-argument
def DestoryAllTargets(gs, subject, target):
  # Only counting the kills by the subject
  destroy_count = sum(np.in1d(subject.event.PLAYER_KILL.target_ent, target))
  return float(destroy_count)/len(target)

elimination_task = task_spec.TaskSpec(
  eval_fn=DestoryAllTargets,
  eval_fn_kwargs={"target": "all_foes"},
  reward_to="team")


class UnfairFight(TeamBattle):
  required_systems = ["TERRAIN", "COMBAT"]

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self._time_limit = 160  # determines the difficulty
    self.max_time_limit = 300
    self.step_size = 20
    self.num_cont_win = 3  # at the same duration, then change the difficulty
    self.adaptive_difficulty = True
    self._team_split = (self.config.PLAYER_N+1)//2
    self.safe_zone = 1

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but heavy
    self.task_embedding = utils.get_hash_embedding(lambda: [elimination_task]*2,
                                                   self.config.TASK_EMBED_DIM)

  @property
  def time_limit(self):
    return self._time_limit

  def set_time_limit(self, time_limit):
    self._time_limit = time_limit

  @property
  def team_split(self):
    return self._team_split

  def set_team_split(self, team_split):
    self._team_split = team_split

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    super().reset(np_random, map_dict)
    self.history[-1]["time_limit"] = self.time_limit

  @property
  def teams(self):
    return {"small": list(range(1, self.team_split)),
            "large": list(range(self.team_split, self.config.PLAYER_N+1)),}

  def _set_config(self):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("TEAMS", self.teams)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)

    # Make the map small
    self.config.set_for_episode("MAP_CENTER", 24)

    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.1)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.9)  # prop of stone tiles: 0.05

    # Activate death fog
    self.config.set_for_episode("PLAYER_DEATH_FOG", 16)
    self.config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/6)
    # Very small area is safe: 3 x 3
    self.config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", self.safe_zone)

    # Disable +1 hp per tick
    self.config.set_for_episode("PLAYER_HEALTH_INCREMENT", False)

    self._determine_difficulty()  # sets the time limit
    self.config.set_for_episode("HORIZON", self.time_limit)

  def _determine_difficulty(self):
    # Determine the difficulty (the seize duration) based on the previous results
    if self.adaptive_difficulty and len(self.history) > self.num_cont_win:
      prev_results = [1 in r["winners"] for r in self.history[-self.num_cont_win:]
                      if r["result"] is True and r["winners"]]
      if sum(prev_results) >= self.num_cont_win:  # small won all
        self._time_limit = min(self.time_limit + self.step_size, self.max_time_limit)
      if sum(prev_results) == 0:  # large won all
        self._time_limit = max(self.time_limit - self.step_size, 60)

  def _define_tasks(self, np_random):
    return task_spec.make_task_from_spec(self.teams, [elimination_task]*2)

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

  def _check_winners(self, dones):
    # If the time is up or all died at the same time, the small team wins
    if self.realm.tick >= self.time_limit or self.realm.num_players == 0:
      return self.teams["small"]
    # TeamBattle._check_winners() also checks if a team is eliminated
    return super()._check_winners(dones)

  @property
  def winning_score(self):
    if self._winners:
      kill_log = self.realm.event_log.get_data(
        agents=self._winners, event_code=event_code.EventCode.PLAYER_KILL)
      kill_bonus = kill_log.shape[0] / (self.config.PLAYER_N - len(self._winners))  # hacky
      speed_bonus = (self.time_limit - self.realm.tick) / self.time_limit
      # This will results in no bonus when the small team wins by time limit
      return kill_bonus + speed_bonus
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

    # pylint: disable=protected-access

    # These should run without errors
    game.history.append({"result": False, "time_limit": game.time_limit})
    game._determine_difficulty()
    game.history.append({"result": True, "winners": None, "time_limit": game.time_limit})
    game._determine_difficulty()

    # Test if the difficulty increases
    org_time_limit = game.time_limit
    for _ in range(game.num_cont_win):
      # the small won all
      game.history.append({"result": True, "winners": [1], "time_limit": game.time_limit})
      game._determine_difficulty()
    assert game.time_limit == (org_time_limit + game.step_size)

    for _ in range(game.num_cont_win):
      # the large won all
      game.history.append({"result": True, "winners": [60], "time_limit": game.time_limit})
      game._determine_difficulty()
    assert game.time_limit == org_time_limit

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(test_config)
  UnfairFight.test(test_env)
