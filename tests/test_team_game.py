# pylint: disable=protected-access
import unittest

import nmmo
from nmmo.core.game_api import AgentTraining, TeamTraining, TeamBattle
from nmmo.lib.team_helper import TeamHelper


NUM_TEAMS = 16
TEAM_SIZE = 8

class TeamConfig(nmmo.config.Small, nmmo.config.AllGameSystems):
  PLAYER_N = NUM_TEAMS * TEAM_SIZE
  TEAMS = {"Team" + str(i+1): [i*TEAM_SIZE+j+1 for j in range(TEAM_SIZE)]
           for i in range(NUM_TEAMS)}
  CURRICULUM_FILE_PATH = "tests/task/sample_curriculum.pkl"

class TestTeamGame(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = TeamConfig()
    cls.env = nmmo.Env(cls.config)

  def test_num_agents_in_teams(self):
    # raise error if PLAYER_N is not equal to the number of agents in TEAMS
    config = TeamConfig()
    config.set("PLAYER_N", 127)
    env = nmmo.Env(config)
    self.assertRaises(AssertionError, lambda: TeamTraining(env))

  def test_agent_training_game(self):
    game = AgentTraining(self.env)
    self.env.reset(game=game)

    # this should use the DefaultGame setup
    self.assertTrue(isinstance(self.env.game, AgentTraining))
    for task in self.env.tasks:
      self.assertEqual(task.reward_to, "agent")  # all tasks are for agents

    # every agent is assigned a task
    self.assertEqual(len(self.env.possible_agents), len(self.env.tasks))
    # for the training tasks, the task assignee and subject should be the same
    for task in self.env.tasks:
      self.assertEqual(task.assignee, task.subject)

    # winners should be none when not determined
    self.assertEqual(self.env.game.winners, None)

    # make agent 1 a winner by destroying all other agents
    for agent_id in self.env.possible_agents[1:]:
      self.env.realm.players[agent_id].resources.health.update(0)
    self.env.step({})
    self.assertEqual(self.env.game.winners, [1])

  def test_team_training_game_spawn(self):
    # when TEAMS is set, the possible agents should include all agents
    team_helper = TeamHelper(self.config.TEAMS)
    self.assertListEqual(self.env.possible_agents,
                         list(team_helper.team_and_position_for_agent.keys()))

    game = TeamTraining(self.env)
    self.env.reset(game=game)

    for task in self.env.tasks:
      self.assertEqual(task.reward_to, "team")  # all tasks are for teams

    # agents in the same team should spawn together
    team_locs = {}
    for team_id, team_members in self.env.config.TEAMS.items():
      team_locs[team_id] = self.env.realm.players[team_members[0]].pos
      for agent_id in team_members:
        self.assertEqual(team_locs[team_id], self.env.realm.players[agent_id].pos)

    # teams should be apart from each other
    for team_a in self.config.TEAMS.keys():
      for team_b in self.config.TEAMS.keys():
        if team_a != team_b:
          self.assertNotEqual(team_locs[team_a], team_locs[team_b])

  def test_team_battle_mode(self):
    game = TeamBattle(self.env)
    self.env.reset(game=game)
    env = self.env

    # battle mode: all teams share the same task
    task_spec_name = env.tasks[0].spec_name
    for task in env.tasks:
      self.assertEqual(task.reward_to, "team")  # all tasks are for teams
      self.assertEqual(task.spec_name, task_spec_name)  # all tasks are the same in competition

    # set the first team to win
    winner_team = "Team1"
    for team_id, members in env.config.TEAMS.items():
      if team_id != winner_team:
        for agent_id in members:
          env.realm.players[agent_id].resources.health.update(0)
    env.step({})
    self.assertEqual(env.game.winners, env.config.TEAMS[winner_team])

  def test_competition_winner_task_completed(self):
    game = TeamBattle(self.env)
    self.env.reset(game=game)

    # The first two tasks get completed
    winners = []
    for task in self.env.tasks[:2]:
      task._completed_tick = 1
      self.assertEqual(task.completed, True)
      winners += task.assignee

    self.env.step({})
    self.assertEqual(self.env.game.winners, winners)

  def test_game_via_config(self):
    config = TeamConfig()
    config.set("GAME_PACKS", [(AgentTraining, 1),
                              (TeamTraining, 1),
                              (TeamBattle, 1)])
    env = nmmo.Env(config)
    env.reset()
    for _ in range(3):
      env.step({})

    self.assertTrue(isinstance(env.game, game_cls)
                    for game_cls in [AgentTraining, TeamTraining, TeamBattle])

if __name__ == '__main__':
  unittest.main()
