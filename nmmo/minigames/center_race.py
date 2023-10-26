from nmmo.core.game_api import Game
from nmmo.task import task_api
from nmmo.lib import utils


# pylint: disable=invalid-name
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

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but huge
    self.task_embedding = utils.get_hash_embedding(ProgressTowardCenter,
                                                   self.config.TASK_EMBED_DIM)

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def _set_config(self):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)

    # Activate death fog
    self.config.set_for_episode("PLAYER_DEATH_FOG", 32)
    self.config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/4)
    # Only the center tile is safe
    self.config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", 0)

  def _define_tasks(self, np_random):
    return task_api.make_same_task(ProgressTowardCenter, self.config.POSSIBLE_AGENTS,
                                   task_kwargs={"embedding": self.task_embedding})

  def update(self, dones, dead_this_tick):
    super().update(dones, dead_this_tick)
    self._winners = self._who_completed_task()


if __name__ == "__main__":
  import nmmo
  config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(config)

  game = RacetoCenter(test_env)
  test_env.reset(game=game)

  # Check configs
  assert config.are_systems_enabled(game.required_systems)
  assert config.COMBAT_SYSTEM_ENABLED is False
  assert config.ITEM_SYSTEM_ENABLED is False
  assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False
  assert config.PLAYER_DEATH_FOG == 32

  for _ in range(30):
    test_env.step({})
