# pylint: disable=no-member,bare-except
from abc import ABC, abstractmethod
from typing import Dict, List
import dill
import numpy as np

from nmmo.task import task_api, task_spec
from nmmo.lib import team_helper

GAME_MODE = ["agent_training", "team_training", "team_battle"]


class Game(ABC):
  game_mode = None

  def __init__(self, env, sampling_weight=None):
    self.config = env.config
    self.realm = env.realm
    self.sampling_weight = sampling_weight or 1.0
    self.tasks = None
    self._next_tasks = None
    self._agent_stats = {}
    self._winners = None
    self._game_done = False
    self.history: List[Dict] = []
    assert self.is_compatible(), "Game is not compatible with the config"

  @abstractmethod
  def is_compatible(self):
    """Check if the game is compatible with the config (e.g., required systems)"""
    raise NotImplementedError

  @property
  def winners(self):
    return self._winners

  @property
  def winning_score(self):
    if self._winners:
      # CHECK ME: should we return the winners" tasks" reward multiplier?
      return 1.0  # default score for task completion
    return 0.0

  def reset(self, np_random, map_dict, tasks=None):
    self._set_config()
    self._set_realm(np_random, map_dict)
    if tasks:
      # tasks comes from env.reset()
      self.tasks = tasks
    elif self._next_tasks:
      # env.reset() cannot take both game and tasks
      # so set next_tasks in the game first
      self.tasks = self._next_tasks
      self._next_tasks = None
    else:
      self.tasks = self._define_tasks(np_random)
    self._post_setup()
    self._reset_stats()

  def _set_config(self):
    """Set config for the episode. Can customize config using config.set_for_episode()"""
    self.config.reset()

  def _set_realm(self, np_random, map_dict):
    """Set up the realm for the episode. Can customize map and spawn"""
    self.realm.reset(np_random, map_dict, custom_spawn=False)

  def _post_setup(self):
    """Post-setup processes, e.g., attach team tags, etc."""

  def _reset_stats(self):
    """Reset stats for the episode"""
    self._agent_stats.clear()
    self._winners = None
    self._game_done = False
    # result = False means the game ended without a winner
    self.history.append({"result": False, "winners": None, "winning_score": None})

  @abstractmethod
  def _define_tasks(self, np_random):
    """Define tasks for the episode."""
    # NOTE: Task embeddings should be provided somehow, e.g., from curriculum file.
    # Otherwise, policies cannot be task-conditioned.
    raise NotImplementedError

  def set_next_tasks(self, tasks):
    """Set the next task to be completed"""
    self._next_tasks = tasks

  def update(self, dones, dead_this_tick):
    """Update the game stats, e.g. agent stats, scores, winners, etc."""
    for agent_id in dones:
      if dones[agent_id]:
        agent = dead_this_tick[agent_id] if agent_id in dead_this_tick\
                                          else self.realm.players[agent_id]
        self._agent_stats[agent_id] = {"time_alive": self.realm.tick,
                                       "progress_to_center": agent.history.exploration}

    self._winners = self._check_winners(dones)

    if self._winners and not self._game_done:
      self._game_done = self.history[-1]["result"] = True
      self.history[-1]["winners"] = self._winners
      self.history[-1]["winning_score"] = self.winning_score
      self.history[-1]["winning_tick"] = self.realm.tick
      self.history[-1].update(self.get_episode_stats())

  def _check_winners(self, dones):
    # Determine winners for the default task
    if self.realm.num_players == 1:  # only one survivor
      return list(self.realm.players.keys())
    if all(dones.values()) or self.realm.tick >= self.config.HORIZON:
      # several agents died at the same time or reached the time limit
      return list(dones.keys())
    return None

  def get_episode_stats(self):
    """A helper function for trainers"""
    total_agent_steps = 0
    progress_to_center = 0
    max_progress = self.config.PLAYER_N * self.config.MAP_SIZE // 2
    for stat in self._agent_stats.values():
      total_agent_steps += stat["time_alive"]
      progress_to_center += stat["progress_to_center"]
    return {
      "total_agent_steps": total_agent_steps,
      "norm_progress_to_center": float(progress_to_center) / max_progress
    }

  ############################
  # Helper functions for Game
  def _who_completed_task(self):
    # Return all assignees who completed their tasks
    winners = []
    for task in self.tasks:
      if task.completed:
        winners += task.assignee
    return winners or None


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
      with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
        dill.load(f) # a list of TaskSpec
    except:
      return False
    return True

  def _define_tasks(self, np_random):
    with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
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
      with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
        dill.load(f) # a list of TaskSpec
    except:
      return False
    return True

  def _set_realm(self, np_random, map_dict):
    self.realm.reset(np_random, map_dict, custom_spawn=True)
    # Custom spawning
    team_loader = team_helper.TeamLoader(self.config, np_random)
    self.realm.npcs.spawn()
    self.realm.players.spawn(team_loader)

  def _post_setup(self):
    self._attach_team_tag()

  def _attach_team_tag(self):
    # setup team names
    for team_id, members in self.config.TEAMS.items():
      for idx, agent_id in enumerate(members):
        self.realm.players[agent_id].name = f"{team_id}_{agent_id}"
        if idx == 0:
          self.realm.players[agent_id].name = f"{team_id}_leader"

  def _get_cand_team_tasks(self, np_random, num_tasks, tags=None):
    # NOTE: use different file to store different set of tasks?
    with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
      curriculum = dill.load(f) # a list of TaskSpec
    cand_specs = [spec for spec in curriculum if spec.reward_to == "team"]
    if tags:
      cand_specs = [spec for spec in cand_specs if tags in spec.tags]
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

  def _check_winners(self, dones):
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
    return self._who_completed_task()
