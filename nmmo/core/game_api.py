# pylint: disable=no-member,bare-except
from abc import ABC, abstractmethod
from typing import Dict, List
import dill
import numpy as np

from nmmo.task import task_api, task_spec, base_predicates
from nmmo.lib import team_helper, utils

GAME_MODE = ["agent_training", "team_training", "team_battle"]


class Game(ABC):
  game_mode = None

  def __init__(self, env, sampling_weight=None):
    self.config = env.config
    self.realm = env.realm
    self._np_random = env._np_random
    self.sampling_weight = sampling_weight or 1.0
    self.tasks = None
    self.assign_dead_reward = True
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
  def name(self):
    return self.__class__.__name__

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
    self._np_random = np_random
    self._set_config()
    self._set_realm(map_dict)
    if tasks:
      # tasks comes from env.reset()
      self.tasks = tasks
    elif self._next_tasks:
      # env.reset() cannot take both game and tasks
      # so set next_tasks in the game first
      self.tasks = self._next_tasks
      self._next_tasks = None
    else:
      self.tasks = self._define_tasks()
    self._post_setup()
    self._reset_stats()

  def _set_config(self):  # pylint: disable=unused-argument
    """Set config for the episode. Can customize config using config.set_for_episode()"""
    self.config.reset()

  def _set_realm(self, map_dict):
    """Set up the realm for the episode. Can customize map and spawn"""
    self.realm.reset(self._np_random, map_dict, custom_spawn=False)

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
  def _define_tasks(self):
    """Define tasks for the episode."""
    # NOTE: Task embeddings should be provided somehow, e.g., from curriculum file.
    # Otherwise, policies cannot be task-conditioned.
    raise NotImplementedError

  def set_next_tasks(self, tasks):
    """Set the next task to be completed"""
    self._next_tasks = tasks

  def update(self, terminated, dead_players, dead_npcs):
    """Process dead players/npcs, update the game stats, winners, etc."""
    self._process_dead_players(terminated, dead_players)
    self._process_dead_npcs(dead_npcs)
    self._winners = self._check_winners(terminated)
    if self._winners and not self._game_done:
      self._game_done = self.history[-1]["result"] = True
      self.history[-1]["winners"] = self._winners
      self.history[-1]["winning_score"] = self.winning_score
      self.history[-1]["winning_tick"] = self.realm.tick
      self.history[-1].update(self.get_episode_stats())

  def _process_dead_players(self, terminated, dead_players):
    for agent_id in terminated:
      if terminated[agent_id]:
        agent = dead_players[agent_id] if agent_id in dead_players\
                                       else self.realm.players[agent_id]
        self._agent_stats[agent_id] = {"time_alive": self.realm.tick,
                                       "progress_to_center": agent.history.exploration}

  def _process_dead_npcs(self, dead_npcs):
    if self.config.NPC_SYSTEM_ENABLED and self.config.NPC_DEFAULT_REFILL_DEAD_NPCS:
      for npc in dead_npcs.values():
        if npc.spawn_danger:
          self.realm.npcs.spawn_dangers.append(npc.spawn_danger)
      # refill npcs to target config.NPC_N, within config.NPC_SPAWN_ATTEMPTS
      self.realm.npcs.default_spawn()

  def _check_winners(self, terminated):
    # Determine winners for the default task
    if self.realm.num_players == 1:  # only one survivor
      return list(self.realm.players.keys())
    if all(terminated.values()):
      # declare all winners when they died at the same time
      return list(terminated.keys())
    if self.realm.tick >= self.config.HORIZON:
      # declare all survivors as winners when the time is up
      return [agent_id for agent_id, done in terminated.items() if not done]
    return None

  @property
  def is_over(self):
    return self.winners is not None or self.realm.num_players == 0 or \
           self.realm.tick >= self.config.HORIZON

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

  def _define_tasks(self):
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

  def _define_tasks(self):
    with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
      # curriculum file may have been changed, so read the file when sampling
      curriculum = dill.load(f) # a list of TaskSpec
    cand_specs = [spec for spec in curriculum if spec.reward_to == "agent"]
    assert len(cand_specs) > 0, "No agent task is defined in the curriculum file"

    sampling_weights = [spec.sampling_weight for spec in cand_specs]
    sampled_spec = self._np_random.choice(cand_specs, size=self.config.PLAYER_N,
                                          p=sampling_weights/np.sum(sampling_weights))
    return task_spec.make_task_from_spec(self.config.POSSIBLE_AGENTS, sampled_spec)

class TeamGameTemplate(Game):
  """A helper class with common utils for team games"""
  assign_dead_reward = False  # Do NOT always assign -1 to dead agents

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

  def _set_realm(self, map_dict):
    self.realm.reset(self._np_random, map_dict, custom_spawn=True)
    # Custom spawning
    team_loader = team_helper.TeamLoader(self.config, self._np_random)
    self.realm.players.spawn(team_loader)
    self.realm.npcs.default_spawn()

  def _post_setup(self):
    self._attach_team_tag()

  @property
  def teams(self):
    return self.config.TEAMS

  def _attach_team_tag(self):
    # setup team names
    for team_id, members in self.teams.items():
      if isinstance(team_id, int):
        team_id = f"Team{team_id:02d}"
      for idx, agent_id in enumerate(members):
        self.realm.players[agent_id].name = f"{team_id}_{agent_id}"
        if idx == 0:
          self.realm.players[agent_id].name = f"{team_id}_leader"

  def _get_cand_team_tasks(self, num_tasks, tags=None):
    # NOTE: use different file to store different set of tasks?
    with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
      curriculum = dill.load(f) # a list of TaskSpec
    cand_specs = [spec for spec in curriculum if spec.reward_to == "team"]
    if tags:
      cand_specs = [spec for spec in cand_specs if tags in spec.tags]
    assert len(cand_specs) > 0, "No team task is defined in the curriculum file"

    sampling_weights = [spec.sampling_weight for spec in cand_specs]
    sampled_spec = self._np_random.choice(cand_specs, size=num_tasks,
                                          p=sampling_weights/np.sum(sampling_weights))
    return sampled_spec

class TeamTraining(TeamGameTemplate):
  """Game setting for team training tasks"""
  game_mode = "team_training"

  def _define_tasks(self):
    sampled_spec = self._get_cand_team_tasks(len(self.config.TEAMS))
    return task_spec.make_task_from_spec(self.config.TEAMS, sampled_spec)

def team_survival_task(num_tick, embedding=None):
  return task_spec.TaskSpec(
    eval_fn=base_predicates.TickGE,
    eval_fn_kwargs={"num_tick": num_tick},
    reward_to="team",
    embedding=embedding)

class TeamBattle(TeamGameTemplate):
  """Game setting for team battle"""
  game_mode = "team_battle"

  def __init__(self, env, sampling_weight=None):
    super().__init__(env, sampling_weight)
    self.task_embedding = utils.get_hash_embedding(base_predicates.TickGE,
                                                   self.config.TASK_EMBED_DIM)

  def is_compatible(self):
    return True

  def _define_tasks(self):
    # NOTE: Teams can win by eliminating all other teams,
    # or fully cooperating to survive for the entire episode
    survive_task = team_survival_task(self.config.HORIZON, self.task_embedding)
    return task_spec.make_task_from_spec(self.config.TEAMS,
                                         [survive_task] * len(self.config.TEAMS))

  def _check_winners(self, terminated):
    # A team is won, when their task is completed first or only one team remains
    assert self.config.TEAMS is not None, "Team battle mode requires TEAMS to be defined"
    current_teams = self._check_remaining_teams()
    if len(current_teams) == 1:
      winner_team = list(current_teams.keys())[0]
      return self.config.TEAMS[winner_team]

    # Return all assignees who completed their tasks
    # Assuming the episode gets ended externally
    return self._who_completed_task()

  def _check_remaining_teams(self):
    current_teams = {}
    for team_id, team in self.config.TEAMS.items():
      alive_members = [agent_id for agent_id in team if agent_id in self.realm.players]
      if len(alive_members) > 0:
        current_teams[team_id] = alive_members
    return current_teams
