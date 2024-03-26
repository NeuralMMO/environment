# pylint: disable=duplicate-code, invalid-name, unused-argument
import time
from nmmo.core.game_api import TeamBattle
from nmmo.task import task_spec
from nmmo.task.base_predicates import DefeatEntity
from nmmo.lib import utils, team_helper


def hunt_task(num_npc):
  return task_spec.TaskSpec(
    eval_fn=DefeatEntity,
    eval_fn_kwargs={"agent_type": "npc", "level": 0, "num_agent": num_npc},
    reward_to="team")

class RadioRaid(TeamBattle):
  required_systems = ["TERRAIN", "COMBAT", "COMMUNICATION", "NPC"]
  num_teams = 6

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)

    self._goal_num_npc = 5  # determines the difficulty
    self.adaptive_difficulty = True
    self.num_game_won = 2  # at the same map size, threshold to increase the difficulty
    self.step_size = 5
    self.quad_centers = None
    self._grass_map = False

    # npc danger: 0=all npc are passive, 1=all npc are aggressive
    self._npc_danger = 0  # increase by .1 per wave
    self._danger_step_size = .15
    self._spawn_center_crit = 0.4  # if danger is less than crit, spawn at center
    self.npc_wave_num = 10  # number of npc to spawn per wave
    self._last_wave_tick = 0
    self.npc_spawn_crit = 3
    self.npc_spawn_radius = 5
    self.max_wave_interval = 25

    # These will probably affect the difficulty
    self.map_size = 48
    self.spawn_immunity = self.config.HORIZON

    # NOTE: This is a hacky way to get a hash embedding for a function
    # TODO: Can we get more meaningful embedding? coding LLMs are good but heavy
    self.task_embedding = utils.get_hash_embedding(hunt_task, self.config.TASK_EMBED_DIM)

  @property
  def teams(self):
    team_size = self.config.PLAYER_N // self.num_teams
    teams = {f"Team{i}": list(range((i-1)*team_size+1, i*team_size+1))
             for i in range(1, self.num_teams)}
    teams[f"Team{self.num_teams}"] = \
      list(range((self.num_teams-1)*team_size+1, self.config.PLAYER_N+1))
    return teams

  @property
  def goal_num_npc(self):
    return self._goal_num_npc

  def set_goal_num_npc(self, goal_num_npc):
    self._goal_num_npc = goal_num_npc

  def set_grass_map(self, grass_map):
    self._grass_map = grass_map

  def is_compatible(self):
    return self.config.are_systems_enabled(self.required_systems)

  def reset(self, np_random, map_dict, tasks=None):
    super().reset(np_random, map_dict)
    self.history[-1]["goal_num_npc"] = self.goal_num_npc
    self._npc_danger = 0
    self._last_wave_tick = 0

  def _set_config(self):
    self.config.reset()
    self.config.toggle_systems(self.required_systems)
    self.config.set_for_episode("MAP_CENTER", self.map_size)
    self.config.set_for_episode("COMBAT_SPAWN_IMMUNITY", self.spawn_immunity)
    self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)
    self.config.set_for_episode("TEAMS", self.teams)
    self.config.set_for_episode("NPC_DEFAULT_REFILL_DEAD_NPCS", False)
    # Regenerate the map from fractal to have less obstacles
    self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    self.config.set_for_episode("TERRAIN_WATER", 0.1)
    self.config.set_for_episode("TERRAIN_FOILAGE", 0.95)
    self.config.set_for_episode("TERRAIN_RESET_TO_GRASS", self._grass_map)
    # NO death fog
    self.config.set_for_episode("DEATH_FOG_ONSET", None)
    # Enable +1 hp per tick -- restore health by eat/drink
    self.config.set_for_episode("PLAYER_HEALTH_INCREMENT", 1)
    # Make NPCs more aggressive
    self.config.set_for_episode("NPC_SPAWN_NEUTRAL", 0.3)
    self.config.set_for_episode("NPC_SPAWN_AGGRESSIVE", 0.8)

    self._determine_difficulty()  # sets the goal_num_npc

  def _determine_difficulty(self):
    # Determine the difficulty (the map size) based on the previous results
    if self.adaptive_difficulty and self.history \
       and self.history[-1]["result"]:  # the last game was won
      last_results = [r["result"] for r in self.history if r["goal_num_npc"] == self.goal_num_npc]
      if sum(last_results) >= self.num_game_won:
        self._goal_num_npc = self._goal_num_npc + self.step_size

  def _set_realm(self, map_dict):
    self.realm.reset(self._np_random, map_dict, custom_spawn=True)
    # team spawn requires custom spawning
    team_loader = team_helper.TeamLoader(self.config, self._np_random)
    self.realm.players.spawn(team_loader)

    # from each team, pick 4 agents and place on each quad center as recons
    self.quad_centers = list(self.realm.map.quad_centers.values())
    for members in self.teams.values():
      recons = self._np_random.choice(members, size=4, replace=False)
      for idx, agent_id in enumerate(recons):
        self.realm.players[agent_id].make_recon(new_pos=self.quad_centers[idx])

  def _define_tasks(self):
    spec_list = [hunt_task(self.goal_num_npc)] * len(self.teams)
    return task_spec.make_task_from_spec(self.teams, spec_list)

  def _process_dead_npcs(self, dead_npcs):
    npc_manager = self.realm.npcs
    diff_player_npc = (self.realm.num_players - self.num_teams*4) - len(npc_manager)
    # Spawn more NPCs if there are more players than NPCs
    # If the gap is large, spawn in waves
    # If the gap is small, spawn in small batches
    if diff_player_npc >= 0 and (len(npc_manager) <= self.npc_spawn_crit or \
       self.realm.tick - self._last_wave_tick > self.max_wave_interval):
      if self._npc_danger < self._spawn_center_crit:
        spawn_pos = self.realm.map.center_coord
      else:
        spawn_pos = self._np_random.choice(self.quad_centers)
      r_min, r_max = spawn_pos[0] - self.npc_spawn_radius, spawn_pos[0] + self.npc_spawn_radius
      c_min, c_max = spawn_pos[1] - self.npc_spawn_radius, spawn_pos[1] + self.npc_spawn_radius
      npc_manager.area_spawn(r_min, r_max, c_min, c_max, self.npc_wave_num,
                             lambda r, c: npc_manager.spawn_npc(r, c, danger=self._npc_danger))
      self._npc_danger += min(self._danger_step_size, 1)  # max danger = 1
      self._last_wave_tick = self.realm.tick

  def _check_winners(self, terminated):
    # No winner game is possible
    return self._who_completed_task()

  @property
  def is_over(self):
    return self.winners is not None or self.realm.tick >= self.config.HORIZON or \
           self.realm.num_players <= (self.num_teams*4)  # 4 immortal recons per team

  @property
  def winning_score(self):
    if self._winners:
      time_limit = self.config.HORIZON
      speed_bonus = (time_limit - self.realm.tick) / time_limit
      alive_bonus = sum(1.0 for agent_id in self._winners if agent_id in self.realm.players)\
                    / len(self._winners)
      return (speed_bonus + alive_bonus) / 2  # set max to 1.0
    return 0.0

  @staticmethod
  def test(env, horizon=30, seed=0):
    game = RadioRaid(env)
    env.reset(game=game, seed=seed)

    # Check configs
    config = env.config
    assert config.are_systems_enabled(game.required_systems)
    assert config.COMBAT_SYSTEM_ENABLED is True
    assert config.RESOURCE_SYSTEM_ENABLED is False
    assert config.COMMUNICATION_SYSTEM_ENABLED is True
    assert config.ITEM_SYSTEM_ENABLED is False
    assert config.DEATH_FOG_ONSET is None
    assert config.ALLOW_MOVE_INTO_OCCUPIED_TILE is False
    assert config.NPC_SYSTEM_ENABLED is True
    assert config.NPC_DEFAULT_REFILL_DEAD_NPCS is False

    start_time = time.time()
    for _ in range(horizon):
      env.step({})
    print(f"Time taken: {time.time() - start_time:.3f} s")  # pylint: disable=bad-builtin

    # pylint: disable=protected-access
    # These should run without errors
    game.history.append({"result": False, "goal_num_npc": game.goal_num_npc})
    game._determine_difficulty()

    # Test if the difficulty changes
    org_goal_npc = game.goal_num_npc
    for result in [False]*7 + [True]*game.num_game_won:
      game.history.append({"result": result, "goal_num_npc": game.goal_num_npc})
      game._determine_difficulty()  # pylint: disable=protected-access
    assert game.goal_num_npc == (org_goal_npc + game.step_size)

if __name__ == "__main__":
  import nmmo
  test_config = nmmo.config.Default()  # Medium, AllGameSystems
  test_env = nmmo.Env(test_config)
  RadioRaid.test(test_env)  # 0.60 s

  # performance test
  from tests.testhelpers import profile_env_step
  test_tasks = task_spec.make_task_from_spec(test_config.TEAMS,
                                             [hunt_task(30)]*len(test_config.TEAMS))
  profile_env_step(tasks=test_tasks)
  # env._compute_rewards(): 0.17201571099940338
