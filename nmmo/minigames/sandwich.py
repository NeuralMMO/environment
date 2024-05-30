import time
import numpy as np

from nmmo.core.game_api import TeamBattle, team_survival_task
from nmmo.task import task_spec
from nmmo.lib import team_helper


def secure_order(pos, radius=5):
  return {"secure": {"position": pos, "radius": radius}}

class Sandwich(TeamBattle):
  required_systems = ["TERRAIN", "COMBAT", "NPC", "COMMUNICATION"]
  num_teams = 8

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self.map_size = 80
    self._inner_npc_num = 2  # determines the difficulty
    self._outer_npc_num = None  # these npcs rally to the center
    self.npc_step_size = 2
    self.adaptive_difficulty = True
    self.num_game_won = 2  # at the same duration, threshold to increase the difficulty
    self.max_npc_num = self.config.PLAYER_N // self.num_teams
    self.survival_crit = 500  # to win, agents must survive this long
    self._grass_map = False

  @property
  def teams(self):
    team_size = self.config.PLAYER_N // self.num_teams
    teams = {i: list(range((i-1)*team_size+1, i*team_size+1))
             for i in range(1, self.num_teams)}
    teams[self.num_teams] = \
      list(range((self.num_teams-1)*team_size+1, self.config.PLAYER_N+1))
    return teams

  @property
  def inner_npc_num(self):
    return self._inner_npc_num

  def set_inner_npc_num(self, inner_npc_num):
    self._inner_npc_num = inner_npc_num

  @property
  def outer_npc_num(self):
    return self._outer_npc_num or min(self._inner_npc_num*self.num_teams, self.map_size*2)

  def set_outer_npc_num(self, outer_npc_num):
    self._outer_npc_num = outer_npc_num

  def set_grass_map(self, grass_map):
    self._grass_map = grass_map

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    super().reset(np_random, map_dict)
    self.history[-1]["inner_npc_num"] = self.inner_npc_num
    self.history[-1]["outer_npc_num"] = self.outer_npc_num
    self._grass_map = False  # reset to default

  def _set_config(self):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("TEAMS", self.teams)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)
    self.config.set_for_episode("NPC_DEFAULT_REFILL_DEAD_NPCS", False)
    # Make the map small
    self.config.set_for_episode("MAP_CENTER", self.map_size)
    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.1)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.9)
    self.config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", False)
    self.config.set_for_episode("TERRAIN_RESET_TO_GRASS", self._grass_map)
    # Activate death fog from the onset
    self.config.set_for_episode("DEATH_FOG_ONSET", 1)
    self.config.set_for_episode("DEATH_FOG_SPEED", 1/10)
    self.config.set_for_episode("DEATH_FOG_FINAL_SIZE", 5)
    # Enable +1 hp per tick
    self.config.set_for_episode("PLAYER_HEALTH_INCREMENT", 1)
    self._determine_difficulty()  # sets the seize duration

  def _determine_difficulty(self):
    # Determine the difficulty based on the previous results
    if self.adaptive_difficulty and self.history \
       and self.history[-1]["result"]:  # the last game was won
      last_results = [r["result"] for r in self.history
                      if r["inner_npc_num"] == self.inner_npc_num]
      if sum(last_results) >= self.num_game_won:
        # Increase the npc num, when there were only few npcs left at the end
        self._inner_npc_num += self.npc_step_size
        self._inner_npc_num = min(self._inner_npc_num, self.max_npc_num)

  def _generate_spawn_locs(self):
    center = self.config.MAP_SIZE // 2
    radius = self.map_size // 4
    angles = np.linspace(0, 2*np.pi, self.num_teams, endpoint=False)
    return [(center + int(radius*np.cos(a)), center + int(radius*np.sin(a))) for a in angles]

  def _set_realm(self, map_dict):
    self.realm.reset(self._np_random, map_dict, custom_spawn=True)
    # team spawn requires custom spawning
    spawn_locs = self._generate_spawn_locs()
    team_loader = team_helper.TeamLoader(self.config, self._np_random, spawn_locs)
    self.realm.players.spawn(team_loader)

    # spawn NPCs
    npc_manager = self.realm.npcs
    center = self.config.MAP_SIZE // 2
    offset = self.config.MAP_CENTER // 8
    for i in range(self.num_teams):
      r, c = spawn_locs[i]
      if r < center:
        r_min, r_max = center - offset, center - 1
      else:
        r_min, r_max = center + 1, center + offset
      if c < center:
        c_min, c_max = center - offset, center - 1
      else:
        c_min, c_max = center + 1, center + offset
      # pylint: disable=cell-var-from-loop
      npc_manager.area_spawn(r_min, r_max, c_min, c_max, self.inner_npc_num,
                             lambda r, c: npc_manager.spawn_npc(
                               r, c, name=f"NPC{i+1}", order={"rally": spawn_locs[i]}))
    npc_manager.edge_spawn(self.outer_npc_num,
                           lambda r, c: npc_manager.spawn_npc(
                              r, c, name="NPC5", order={"rally": (center,center)}))

  def _process_dead_npcs(self, dead_npcs):
    npc_manager = self.realm.npcs
    target_num = min(self.realm.num_players, self.inner_npc_num) // 2
    if len(npc_manager) < target_num:
      center = self.config.MAP_SIZE // 2
      offset = self.config.MAP_CENTER // 6
      r_min = c_min = center - offset
      r_max = c_max = center + offset
      num_spawn = target_num - len(npc_manager)
      npc_manager.area_spawn(r_min, r_max, c_min, c_max, num_spawn,
                             lambda r, c: npc_manager.spawn_npc(
                               r, c, name="NPC5", order={"rally": (center,center)}))

  @property
  def winning_score(self):
    if self._winners:
      time_limit = self.config.HORIZON
      speed_bonus = (time_limit - self.realm.tick) / time_limit
      return speed_bonus  # set max to 1.0
    # No one succeeded
    return 0.0

  def _check_winners(self, terminated):
    # Basic survival criteria
    if self.realm.tick < self.survival_crit:
      return None
    return super()._check_winners(terminated)

  @staticmethod
  def test(env, horizon=30, seed=0):
    game = Sandwich(env)
    env.reset(game=game, seed=seed)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.TERRAIN_SYSTEM_ENABLED is True
    assert config.RESOURCE_SYSTEM_ENABLED is False
    assert config.COMBAT_SYSTEM_ENABLED is True
    assert config.NPC_SYSTEM_ENABLED is True
    assert config.NPC_DEFAULT_REFILL_DEAD_NPCS is False
    assert config.EQUIPMENT_SYSTEM_ENABLED is False  # equipment is used to set npc stats
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False

    start_time = time.time()
    for _ in range(horizon):
      env.step({})
    print(f"Time taken: {time.time() - start_time:.3f} s")  # pylint: disable=bad-builtin

    # Test if the difficulty increases
    org_inner_npc_num = game.inner_npc_num
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append(
        {"result": result, "inner_npc_num": game.inner_npc_num})
      game._determine_difficulty()  # pylint: disable=protected-access
    assert game.inner_npc_num == (org_inner_npc_num + game.npc_step_size)

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(test_config)
  Sandwich.test(test_env)  # 0.74 s

  # performance test
  from tests.testhelpers import profile_env_step
  test_tasks = task_spec.make_task_from_spec(test_config.TEAMS,
                                        [team_survival_task(30)]*len(test_config.TEAMS))
  profile_env_step(tasks=test_tasks)
  # env._compute_rewards(): 0.1768564050034911
