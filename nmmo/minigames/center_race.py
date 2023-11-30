# pylint: disable=invalid-name, duplicate-code
import time
from nmmo.core.game_api import Game
from nmmo.task import task_api
from nmmo.task.base_predicates import ProgressTowardCenter
from nmmo.lib import utils


class RacetoCenter(Game):
  required_systems = ["TERRAIN", "RESOURCE"]

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self._map_size = 32  # determines the difficulty
    self.adaptive_difficulty = True
    self.num_game_won = 1  # at the same map size, threshold to increase the difficulty
    self.step_size = 8
    self.num_player_resurrect = 0

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
    self.num_player_resurrect = 0

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
    self.config.set_for_episode("DEATH_FOG_ONSET", None)  # 32
    # self.config.set_for_episode("DEATH_FOG_SPEED", 1/6)
    # # Only the center tile is safe
    # self.config.set_for_episode("DEATH_FOG_FINAL_SIZE", 0)

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

  def _set_realm(self, map_dict):
    # NOTE: this game respawns dead players at the edge, so setting delete_dead_entity=False
    self.realm.reset(self._np_random, map_dict, delete_dead_player=False)

  def _define_tasks(self):
    return task_api.make_same_task(ProgressTowardCenter, self.config.POSSIBLE_AGENTS,
                                   task_kwargs={"embedding": self.task_embedding})

  def _process_dead_players(self, dones, dead_players):
    # Respawn dead players at the edge
    for player in dead_players.values():
      player.resurrect(freeze_duration=10, health_prop=1, edge_spawn=True)
      self.num_player_resurrect += 1

  @property
  def winning_score(self):
    if self._winners:
      time_limit = self.config.HORIZON
      return (time_limit - self.realm.tick) / time_limit  # speed bonus
    # No one reached the center
    return 0.0

  def _check_winners(self, dones):
    return self._who_completed_task()

  @staticmethod
  def test(env, horizon=30, seed=0):
    game = RacetoCenter(env)
    env.reset(game=game, seed=seed)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is False
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False

    start_time = time.time()
    for _ in range(horizon):
      _, r, d, _ = env.step({})
    print(f"Time taken: {time.time() - start_time:.3f} s")  # pylint: disable=bad-builtin

    # Test if the difficulty increases
    org_map_size = game.map_size
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append({"result": result, "map_size": game.map_size})
      game._determine_difficulty()  # pylint: disable=protected-access
    assert game.map_size == (org_map_size + game.step_size)

    # Check if returns of resurrect/frozen players are correct
    for agent_id, player in env._dead_this_tick.items():  # pylint: disable=protected-access
      assert player.alive, "Resurrected players should be alive"
      assert player.status.frozen, "Resurrected players should be frozen"
      assert player.my_task.progress == 0, "Resurrected players should have 0 progress"
      assert d[agent_id], "Resurrected players should be done = True"
      assert r[agent_id] == -1, "Resurrected players should have -1 reward"

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(test_config)
  RacetoCenter.test(test_env)  # 0.85 s

  # performance test
  from tests.testhelpers import profile_env_step
  test_tasks = task_api.make_same_task(ProgressTowardCenter, test_env.possible_agents)
  profile_env_step(tasks=test_tasks)
  # env._compute_rewards(): 1.9577480710031523
