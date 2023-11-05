# pylint: disable=invalid-name, duplicate-code
from nmmo.core.game_api import TeamBattle
from nmmo.task import task_spec
from nmmo.lib import utils, team_helper

def SeizeCenter(gs, subject, num_ticks):
  if not any(subject.health > 0):  # subject should be alive
    return 0.0
  progress_bonus = 0.3
  center_tile = (gs.config.MAP_SIZE//2, gs.config.MAP_SIZE//2)

  # if subject seized the center tile, start counting ticks
  if center_tile in gs.seize_status and gs.seize_status[center_tile][0] in subject.agents:
    seize_duration = gs.current_tick - gs.seize_status[center_tile][1]
    return progress_bonus + (1.0 - progress_bonus) * seize_duration/num_ticks

  # motivate agents to seize the center tile
  max_dist = center_tile[0] - gs.config.MAP_BORDER
  r = subject.row
  c = subject.col
  # distance to the center tile, so dist = 0 when subject is on the center tile
  # NOTE: subject can be multiple agents (e.g., team), so taking the minimum
  dists = min(utils.linf(list(zip(r,c)), center_tile))
  return progress_bonus * (1.0 - dists/max_dist)

def seize_task(dur_to_win):
  return task_spec.TaskSpec(
    eval_fn=SeizeCenter,
    eval_fn_kwargs={"num_ticks": dur_to_win},
    reward_to="team")


class KingoftheHill(TeamBattle):
  required_systems = ["TERRAIN", "COMBAT", "RESOURCE"]

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self._seize_duration = 10  # determines the difficulty
    self.dur_step_size = 10
    self.max_seize_duration = 100
    self.adaptive_difficulty = True
    self.num_game_won = 1  # at the same duration, threshold to increase the difficulty
    self.map_size = 40
    self.score_scaler = .5

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but huge
    self.task_embedding = utils.get_hash_embedding(SeizeCenter,
                                                   self.config.TASK_EMBED_DIM)

  @property
  def seize_duration(self):
    return self._seize_duration

  def set_seize_duration(self, seize_duration):
    self._seize_duration = seize_duration

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    super().reset(np_random, map_dict)
    self.history[-1]["map_size"] = self.map_size
    self.history[-1]["seize_duration"] = self.seize_duration

  def _set_config(self):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)

    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.05)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.95)  # prop of stone tiles: 0.05
    self.config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", True)

    # Activate death fog
    self.config.set_for_episode("PLAYER_DEATH_FOG", 32)
    self.config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/6)
    # Only the center tile is safe
    self.config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", 12)

    self._determine_difficulty()  # sets the seize duration
    self.config.set_for_episode("MAP_CENTER", self.map_size)

  def _determine_difficulty(self):
    # Determine the difficulty (the map center) based on the previous results
    if self.adaptive_difficulty and self.history \
       and self.history[-1]["result"]:  # the last game was won
      last_results = [r["result"] for r in self.history
                      if r["seize_duration"] == self.seize_duration]
      if sum(last_results) >= self.num_game_won:
        self._seize_duration = min(self.seize_duration + self.dur_step_size,
                                   self.max_seize_duration)

  def _set_realm(self, np_random, map_dict):
    self.realm.reset(np_random, map_dict, custom_spawn=True,
                     seize_targets=[(self.config.MAP_SIZE//2,self.config.MAP_SIZE//2)])
    # team spawn requires custom spawning
    team_loader = team_helper.TeamLoader(self.config, np_random)
    self.realm.players.spawn(team_loader)

  def _define_tasks(self, np_random):
    spec_list = [seize_task(self.seize_duration)] * len(self.teams)
    return task_spec.make_task_from_spec(self.teams, spec_list)

  @property
  def winning_score(self):
    if self._winners:
      speed_score = (self.map_size * (self.seize_duration**self.score_scaler))\
                    / max(self.realm.tick, 1)
      alive_bonus = sum(1.0 for agent_id in self._winners if agent_id in self.realm.players)\
                    / len(self._winners)
      return speed_score + alive_bonus
    # No one succeeded
    return 0.0

  def _check_winners(self, dones):
    return self._who_completed_task()

  @staticmethod
  def test(env, horizon=30):
    game = KingoftheHill(env)
    env.reset(game=game)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.TERRAIN_SYSTEM_ENABLED is True
    assert config.RESOURCE_SYSTEM_ENABLED is True
    assert config.COMBAT_SYSTEM_ENABLED is True
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False
    assert config.PLAYER_DEATH_FOG == 32
    assert env.realm.map.seize_targets == [(config.MAP_SIZE//2, config.MAP_SIZE//2)]

    for _ in range(horizon):
      env.step({})

    # Test if the difficulty increases
    org_seize_dur = game.seize_duration
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append({"result": result, "seize_duration": game.seize_duration})
      game._determine_difficulty()  # pylint: disable=protected-access
    assert game.seize_duration == (org_seize_dur + game.dur_step_size)

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_config.set("TEAMS", team_helper.make_teams(test_config, num_teams=7))
  test_env = nmmo.Env(test_config)
  KingoftheHill.test(test_env)
