import logging
import unittest

from tests.testhelpers import ScriptedAgentTestConfig
import numpy as np

import nmmo

# pylint: disable=import-error, unused-argument
from nmmo.core.env import Env as TaskEnv
from nmmo.task import sampler
from nmmo.task.task_api import Repeat
from nmmo.task.predicate import Predicate, predicate
from nmmo.task.group import Group
from nmmo.task.utils import TeamHelper
from nmmo.task.game_state import GameState

from nmmo.systems import item as Item
from nmmo.io import action as Action

@predicate
def Success(gs):
  return True

@predicate
def Failure(gs):
  return False

class FakePredicate(Predicate):
  def __init__(self, subject:Group, param1: int, param2: Item.Item, param3: Action.Style) -> None:
    super().__init__(subject, param1, param2, param3)
    self._param1 = param1
    self._param2 = param2
    self._param3 = param3

  def _evaluate(self, gs: GameState) -> bool:
    return False

class MockGameState(GameState):
  def __init__(self):
    # pylint: disable=super-init-not-called
    pass

class TestTaskAPI(unittest.TestCase):

  def test_operators(self):
    # pylint: disable=unsupported-binary-operation,invalid-unary-operand-type

    mock_gs = MockGameState()

    # AND (&), OR (|), NOT (~), IMPLY (>>)
    task1 = Success() & Failure() & Success()
    self.assertFalse(task1(mock_gs))

    task2 = Success() | Failure() | Success()
    self.assertTrue(task2(mock_gs))

    task3 = Success() &  ~ Failure() & Success()
    self.assertTrue(task3(mock_gs))

    task4 = Success() >> Success()
    self.assertTrue(task4(mock_gs))

    task5 = Success() >> ~ Success()
    self.assertFalse(task5(mock_gs))

    task6 = (Failure() >> Failure()) & Success()
    self.assertTrue(task6(mock_gs))

  def test_predicate_name(self):

    success = Success()
    failure = Failure()
    fake_task = FakePredicate(Group([2]), 1, Item.Hat, Action.Melee)
    combination = (success & ~ (failure | fake_task)) | (failure >> fake_task)

    self.assertEqual(combination.name,
      "OR(AND(Success,NOT(OR(Failure,FakePredicate_(2,)_1_Hat_Melee))),"
      "IMPLY(Failure->FakePredicate_(2,)_1_Hat_Melee))")

  def test_team_helper(self):
    # TODO(kywch): This test is true now but may change later.

    config = ScriptedAgentTestConfig()
    env = nmmo.Env(config)
    env.reset()

    team_helper = TeamHelper(list(range(1, config.PLAYER_N+1)), len(config.PLAYERS))

    # agents' population should match team_helper team id
    for ent_id, ent in env.realm.players.items():
      # pylint: disable=protected-access
      self.assertEqual(team_helper._ent_to_team[ent_id], ent.population)

  def test_team_assignment(self):
    team =  Group([1, 2, 8, 9], "TeamFoo")

    self.assertEqual(team.name, 'TeamFoo')
    self.assertEqual(team[2].name, "TeamFoo.2")
    self.assertEqual(team[2], (8,))

    # don't allow member of one-member team
    self.assertEqual(team[2][0].name, team[2].name)

  def test_random_task_sampler(self):
    rand_sampler = sampler.RandomTaskSampler()

    rand_sampler.add_task_spec(Success, [[Group([1]), Group([3])]])
    rand_sampler.add_task_spec(Failure, [[Group([2]), Group([1,3])]])
    rand_sampler.add_task_spec(FakePredicate, [
      [Group([1]), Group([2]), Group([1,2]), Group([3]), Group([1,3])],
      [Item.Hat, Item.Top, Item.Bottom],
      [1, 5, 10],
      [0.1, 0.2, 0.3, 0.4]
    ])

    rand_sampler.sample(max_clauses=4, max_clause_size=3, not_p=0.5)

  def test_completed_tasks_in_info(self):
    config = ScriptedAgentTestConfig()
    env = TaskEnv(config)

    # some team helper maybe necessary
    team_helper = TeamHelper(range(1, config.PLAYER_N+1), len(config.PLAYERS))

    fake_task = FakePredicate(team_helper.left_team(3), Item.Hat, 1, 0.1)
    task_assignment = \
      [(Repeat(assignee=Group([1]), predicate=Success(), reward=1),2),
        Repeat(assignee=Group([1]), predicate=Failure(), reward=1),
        Repeat(assignee=Group([1]), predicate=Success(), reward=-1),
        Repeat(assignee=team_helper.own_team(2), predicate=Success(), reward=1),
        Repeat(assignee=Group([3]), predicate=fake_task, reward=2)]
    env.change_task(task_assignment)
    _, _, _, infos = env.step({})
    logging.info(infos)

    # agent 1: task1 is always True
    self.assertEqual(infos[1]['task'][Success().name], 1)

    # agent 2 should have been assigned Success() but not FakePredicate()
    self.assertEqual(infos[2]['task'][Success().name], 1)
    self.assertTrue(fake_task.name not in infos[2]['task'])

    # agent 3 should have been assigned FakePredicate(), which is always False (0)
    self.assertEqual(infos[3]['task'][fake_task.name], 0)

    # all agents in the same team with agent 2 have Success()
    # other agents don't have any tasks assigned
    for ent_id in range(4, config.PLAYER_N+1):
      if Group([ent_id]) in team_helper.own_team(2):
        self.assertEqual(infos[ent_id]['task'][Success().name], 1)
      else:
        self.assertEqual(infos[ent_id]['task'], {})

  def test_task_embedding(self):
    env = TaskEnv()
    obs = env.reset()
    self.assertEqual(obs[1]['Task'].shape, 
                     env.observation_space(1)['Task'].shape)
    
    task = [Repeat(assignee=Group([1,2]),predicate=Success())]
    env.change_task(task,
                    task_encoding={1:np.array([1,2,3,4])},
                    embedding_size=4)
    obs = env.reset()
    self.assertTrue(all(obs[1]['Task']==np.array([1,2,3,4])))
    self.assertTrue(all(obs[2]['Task']==np.array([0,0,0,0])))

if __name__ == '__main__':
  unittest.main()
