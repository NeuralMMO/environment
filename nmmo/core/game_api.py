# pylint: disable=no-member,bare-except
from abc import ABC, abstractmethod
import dill
import numpy as np
from nmmo.task import task_api, task_spec
from nmmo.lib import team_helper

GAME_MODE = ["agent_training", "team_training", "team_battle"]


class Game(ABC):
  game_mode = None

  def __init__(self, config, realm, sampling_weight=None):
    self.config = config
    self.realm = realm
    self.sampling_weight = sampling_weight
    self.tasks = None
    self._agent_stats = {}
    self._winners = None
    assert self.is_compatible(), "Game is not compatible with the config"

  @abstractmethod
  def is_compatible(self):
    """Check if the game is compatible with the config (e.g., required systems)"""
    raise NotImplementedError

  @property
  def winners(self):
    return self._winners

  def reset(self, np_random, map_np_array, tasks=None):
    self._set_config()
    self._set_realm(np_random, map_np_array)
    self.tasks = tasks if tasks else self._define_tasks(np_random)
    self._agent_stats.clear()
    self._winners = None

  def _set_config(self):
    """Set config for the episode. Can customize config using config.set_for_episode()"""
    self.config.reset()

  def _set_realm(self, np_random, map_np_array):
    """Set up the realm for the episode. Can customize map and spawn"""
    self.realm.reset(np_random, map_np_array, custom_spawn=False)

  @abstractmethod
  def _define_tasks(self, np_random):
    """Define tasks for the episode."""
    raise NotImplementedError

  def update(self, dones, dead_this_tick):
    """Update the game stats, e.g. agent stats, scores, winners, etc."""
    for agent_id in dones:
      if dones[agent_id]:
        agent = dead_this_tick[agent_id] if agent_id in dead_this_tick\
                                          else self.realm.players[agent_id]
        self._agent_stats[agent_id] = {"time_alive": self.realm.tick,
                                       "progress_to_center": agent.history.exploration}

    # Determine winners for the default task
    if self.realm.num_players == 1:  # only one survivor
      self._winners = list(self.realm.players.keys())
    elif all(dones.values()):  # several agents died at the same time
      self._winners = list(dones.keys())

  def get_episode_stats(self):
    """A helper function for trainers"""
    total_agent_steps = 0
    progress_to_center = 0
    max_progress = self.config.PLAYER_N * self.config.MAP_SIZE // 2
    for stat in self._agent_stats.values():
      total_agent_steps += stat['time_alive']
      progress_to_center += stat['progress_to_center']
    return {
      'total_agent_steps': total_agent_steps,
      'norm_progress_to_center': float(progress_to_center) / max_progress
    }

class DefaultGame(Game):
  """The default NMMO game"""
  game_mode = "agent_training"

  def is_compatible(self):
    return True

  def _define_tasks(self, np_random):
    return task_api.nmmo_default_task(self.config.POSSIBLE_AGENTS)

class AgentTraining(Game):
  """Game setting for agent training tasks"""
  game_mode = "agent_training"

  def is_compatible(self):
    try:
      assert self.config.COMBAT_SYSTEM_ENABLED, "Combat system must be enabled"
      # Check is the curriculum file exists and opens
      with open(self.config.CURRICULUM_FILE_PATH, 'rb') as f:
        dill.load(f) # a list of TaskSpec
    except:
      return False

    return True

  def _define_tasks(self, np_random):
    with open(self.config.CURRICULUM_FILE_PATH, 'rb') as f:
      # curriculum file may have been changed, so read the file when sampling
      curriculum = dill.load(f) # a list of TaskSpec
    cand_specs = [spec for spec in curriculum if spec.reward_to == "agent"]
    assert len(cand_specs) > 0, "No agent task is defined in the curriculum file"

    sampling_weights = [spec.sampling_weight for spec in cand_specs]
    sampled_spec = np_random.choice(cand_specs, size=self.config.PLAYER_N,
                                    p=sampling_weights/np.sum(sampling_weights))
    return task_spec.make_task_from_spec(self.config.POSSIBLE_AGENTS, sampled_spec)

class TeamGameTemplate(Game):
  """A helper class with common utils for team games"""
  def is_compatible(self):
    try:
      assert self.config.COMBAT_SYSTEM_ENABLED, "Combat system must be enabled"
      assert self.config.TEAMS is not None, "Team training mode requires TEAMS to be defined"
      num_agents = sum(len(v) for v in self.config.TEAMS.values())
      assert self.config.PLAYER_N == num_agents,\
        "PLAYER_N must match the number of agents in TEAMS"
      # Check is the curriculum file exists and opens
      with open(self.config.CURRICULUM_FILE_PATH, 'rb') as f:
        dill.load(f) # a list of TaskSpec
    except:
      return False

    return True

  def _set_realm(self, np_random, map_np_array):
    self.realm.reset(np_random, map_np_array, custom_spawn=True)
    # Custom spawning
    team_loader = team_helper.TeamLoader(self.config, np_random)
    self.realm.npcs.spawn()
    self.realm.players.spawn(team_loader)

  def _get_cand_team_tasks(self, np_random, num_tasks):
    # NOTE: use different file to store different set of tasks?
    with open(self.config.CURRICULUM_FILE_PATH, 'rb') as f:
      curriculum = dill.load(f) # a list of TaskSpec
    cand_specs = [spec for spec in curriculum if spec.reward_to == "team"]
    assert len(cand_specs) > 0, "No team task is defined in the curriculum file"

    sampling_weights = [spec.sampling_weight for spec in cand_specs]
    sampled_spec = np_random.choice(cand_specs, size=num_tasks,
                                    p=sampling_weights/np.sum(sampling_weights))
    return sampled_spec

class TeamTraining(TeamGameTemplate):
  """Game setting for team training tasks"""
  game_mode = "team_training"

  def _define_tasks(self, np_random):
    sampled_spec = self._get_cand_team_tasks(np_random, len(self.config.TEAMS))
    return task_spec.make_task_from_spec(self.config.TEAMS, sampled_spec)

class TeamBattle(TeamGameTemplate):
  """Game setting for team battle"""
  game_mode = "team_battle"

  def _define_tasks(self, np_random):
    sampled_spec = self._get_cand_team_tasks(np_random, num_tasks=1)[0]
    return task_spec.make_task_from_spec(self.config.TEAMS,
                                         [sampled_spec] * len(self.config.TEAMS))

  def update(self, dones, dead_this_tick):
    super().update(dones, dead_this_tick)
    self._winners = self._check_battle_winners()

  def _check_battle_winners(self):
    # A team is won, when their task is completed first or only one team remains
    assert self.config.TEAMS is not None, "Team battle mode requires TEAMS to be defined"
    current_teams = {}
    for team_id, team in self.config.TEAMS.items():
      alive_members = [agent_id for agent_id in team if agent_id in self.realm.players]
      if len(alive_members) > 0:
        current_teams[team_id] = alive_members
    if len(current_teams) == 1:
      winner_team = list(current_teams.keys())[0]
      return self.config.TEAMS[winner_team]

    # Return all assignees who completed their tasks
    # Assuming the episode gets ended externally
    winners = []
    for task in self.tasks:
      if task.completed:
        winners += task.assignee
    if len(winners) == 0:
      winners = None
    return winners
