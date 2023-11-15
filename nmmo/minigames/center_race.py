# pylint: disable=invalid-name, duplicate-code
from nmmo.core.game_api import Game
from nmmo.task import task_api
from nmmo.lib import utils


def ProgressTowardCenter(gs, subject):
  if not any(subject.health > 0):  # subject should be alive
    return 0.0

  center = gs.config.MAP_SIZE // 2
  max_dist = center - gs.config.MAP_BORDER

  r = subject.row
  c = subject.col
  # distance to the center tile, so dist = 0 when subject is on the center tile
  # NOTE: subject can be multiple agents (e.g., team), so taking the minimum
  dists = min(utils.linf(list(zip(r,c)), (center, center)))

  return 1.0 - dists/max_dist

class RacetoCenter(Game):
  required_systems = ["TERRAIN", "RESOURCE"]

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self._map_size = 32  # determines the difficulty
    self.score_scaler = 1.3
    self.adaptive_difficulty = True
    self.num_game_won = 1  # at the same map size, threshold to increase the difficulty
    self.step_size = 8

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but huge
    self.task_embedding = utils.get_hash_embedding(ProgressTowardCenter,
                                                   self.config.TASK_EMBED_DIM)

  @property
  def map_size(self):
    return self._map_size

  def set_map_size(self, map_size):
    self._map_size = map_size

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    assert self.map_size >= self.config.PLAYER_N//4,\
      f"self.map_size({self.map_size}) must be >= {self.config.PLAYER_N//4}"
    map_dict["mark_center"] = True  # mark the center tile
    super().reset(np_random, map_dict)
    self.history[-1]["map_size"] = self.map_size

  def _set_config(self, np_random):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)

    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.05)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.95)  # prop of stone tiles: 0.05
    self.config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", True)

    # Activate death fog
    self.config.set_for_episode("DEATH_FOG_ONSET", 32)
    self.config.set_for_episode("DEATH_FOG_SPEED", 1/6)
    # Only the center tile is safe
    self.config.set_for_episode("DEATH_FOG_FINAL_SIZE", 0)

    self._determine_difficulty()  # sets the map_size
    self.config.set_for_episode("MAP_CENTER", self.map_size)

  def _determine_difficulty(self):
    # Determine the difficulty (the map size) based on the previous results
    if self.adaptive_difficulty and self.history \
       and self.history[-1]["result"]:  # the last game was won
      last_results = [r["result"] for r in self.history if r["map_size"] == self.map_size]
      if sum(last_results) >= self.num_game_won \
        and self.map_size <= self.config.original["MAP_CENTER"] - self.step_size:
        self._map_size += self.step_size

  def _define_tasks(self, np_random):
    return task_api.make_same_task(ProgressTowardCenter, self.config.POSSIBLE_AGENTS,
                                   task_kwargs={"embedding": self.task_embedding})

  @property
  def winning_score(self):
    if self._winners:
      return (self.map_size**self.score_scaler)/max(self.realm.tick, 1)
    # No one reached the center
    return 0.0

  def _check_winners(self, dones):
    return self._who_completed_task()

  @staticmethod
  def test(env, horizon=30):
    game = RacetoCenter(env)
    env.reset(game=game)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is False
    assert config.DEATH_FOG_ONSET == 32
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False

    for _ in range(horizon):
      env.step({})

    # Test if the difficulty increases
    org_map_size = game.map_size
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append({"result": result, "map_size": game.map_size})
      game._determine_difficulty()  # pylint: disable=protected-access
    assert game.map_size == (org_map_size + game.step_size)

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(test_config)
  RacetoCenter.test(test_env)
