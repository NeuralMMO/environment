# pylint: disable=import-error,unused-argument,invalid-name
# pylint: disable=no-member,no-value-for-parameter,not-callable,expression-not-assigned
import unittest
import numpy as np

import nmmo
from nmmo.core.env import Env
from nmmo.task.task_api import define_predicate, define_task
from nmmo.task.group import Group
from nmmo.task.team_helper import TeamHelper
from nmmo.task.constraint import InvalidConstraint, ScalarConstraint
from nmmo.task.base_predicates import TickGE, CanSeeGroup

from nmmo.systems import item as Item
from nmmo.core import action as Action

from tests.testhelpers import ScriptedAgentTestConfig

@define_predicate
def Success(gs, subject: Group):
  return True

@define_predicate
def Failure(gs, subject: Group):
  return False

@define_task
def Fake(gs, subject, a,b,c):
  return False

class MockGameState():
  def __init__(self):
    # pylint: disable=super-init-not-called
    self.config = nmmo.config.Default()
    self.cache_result = {}
    self.get_subject_view = lambda _: None
class TestTaskAPI(unittest.TestCase):

  def test_operators(self):
    # pylint: disable=unsupported-binary-operation,invalid-unary-operand-type

    mock_gs = MockGameState()
    SUCCESS = Success(Group([0]))
    FAILURE = Failure(Group([0]))
    # AND (&), OR (|), NOT (~), IMPLY (>>)
    task1 = SUCCESS & FAILURE
    self.assertFalse(task1(mock_gs))

    task2 = SUCCESS | FAILURE | SUCCESS
    self.assertTrue(task2(mock_gs))

    task3 = SUCCESS & ~ FAILURE & SUCCESS
    self.assertTrue(task3(mock_gs))

    task4 = SUCCESS >> SUCCESS
    self.assertTrue(task4(mock_gs))

    task5 = SUCCESS >> ~ SUCCESS
    self.assertFalse(task5(mock_gs))

    task6 = (FAILURE >> FAILURE) & SUCCESS
    self.assertTrue(task6(mock_gs))

    task7 = SUCCESS + SUCCESS
    self.assertEqual(task7(mock_gs),2)

    task8 = SUCCESS * 3
    self.assertEqual(task8(mock_gs),3)

    self.assertEqual(task6.name, "(PAND_(IMPLY_(Failure_(0,))_(Failure_(0,)))_(Success_(0,)))")

  def test_team_assignment(self):
    team =  Group([1, 2, 8, 9], "TeamFoo")

    self.assertEqual(team.name, 'TeamFoo')
    self.assertEqual(team[2].name, "TeamFoo.2")
    self.assertEqual(team[2], (8,))

    # don't allow member of one-member team
    self.assertEqual(team[2][0].name, team[2].name)

  def test_task_name(self):
    SUCCESS = Success(Group([0]))
    FAILURE = Failure(Group([0]))
    fake_task = Fake(Group([2]), 1, Item.Hat, Action.Melee)
    combination = (SUCCESS & ~ (FAILURE | fake_task)) | (FAILURE >> fake_task)
    self.assertEqual(combination.name,
      "(POR_(PAND_(Success_(0,))_(PNOT_(POR_(Failure_(0,))_(Fake_(2,)_1_Hat_Melee))))_\
(IMPLY_(Failure_(0,))_(Fake_(2,)_1_Hat_Melee)))")

  def test_constraint(self):
    mock_gs = MockGameState()
    good = Success(Group([0]))
    bad = Success(Group([99999]))
    good(mock_gs)
    self.assertRaises(InvalidConstraint,lambda: bad(mock_gs))

    scalar = ScalarConstraint(low=-10,high=10)
    for _ in range(10):
      self.assertTrue(scalar.sample(mock_gs.config)<10)
      self.assertTrue(scalar.sample(mock_gs.config)>=-10)

    bad = TickGE(Group([0]), -1)
    self.assertRaises(InvalidConstraint, lambda: bad(mock_gs))

  def test_sample_task(self):
    task = CanSeeGroup() & TickGE()
    self.assertEqual(task.name,
                     "(PAND_(CanSeeGroup_subject:GroupConstraint_target:GroupConstraint)_\
(TickGE_subject:GroupConstraint_num_tick:ScalarConstraint))")
    config = nmmo.config.Default()
    TickGE().sample(config)
    task.sample(config).name

    # DONE

  def test_completed_tasks_in_info(self):
    config = ScriptedAgentTestConfig()
    env = Env(config)
    team_helper = TeamHelper.generate_from_config(config)
    fake_task = Fake(Group([3]), 1, Item.Hat, Action.Melee)
    task_assignment = \
      [(Success(Group([1])),2),
       Failure(Group([1])),
       Success(Group([1])) * -1,
       3 * Success(Group([1])),
       Success(team_helper.own_team(2)),
       fake_task
      ]
    env.change_task(task_assignment)
    _, _, _, infos = env.step({})

    # agent 1: task1 is always True
    self.assertEqual(infos[1]['task'][Success(Group([1])).name], 1.0)
    self.assertEqual(infos[1]['task'][(Success(Group([1])) * -1).name], -1.0)
    self.assertEqual(infos[1]['task'][(3*Success(Group([1]))).name], 3.0)

    # agent 2 should have been assigned Success but not Fake()
    self.assertEqual(infos[2]['task'][Success(team_helper.own_team(2)).name], 1)
    self.assertTrue(fake_task.name not in infos[2]['task'])

    # agent 3 should have been assigned Fake(), which is always False (0)
    self.assertEqual(infos[3]['task'][fake_task.name], 0)

    # all agents in the same team with agent 2 have SUCCESS
    # other agents don't have any tasks assigned
    group_name = Success(team_helper.own_team(2)).name
    for ent_id in range(4, config.PLAYER_N+1):
      if Group([ent_id]) in team_helper.own_team(2):
        self.assertEqual(infos[ent_id]['task'][group_name], 1)
      else:
        self.assertEqual(infos[ent_id]['task'], {})

    # DONE

  def test_task_embedding(self):
    env = Env()
    obs = env.reset()
    self.assertRaises(KeyError, lambda: obs[1]['Task'])

    task = [Success([1,2])]
    env.change_task(task,
                    task_encoding={1:np.array([1,2,3,4])},
                    embedding_size=4)
    obs = env.reset()
    self.assertTrue(all(obs[1]['Task']==np.array([1,2,3,4])))
    self.assertTrue(all(obs[2]['Task']==np.array([0,0,0,0])))

    # DONE

if __name__ == '__main__':
  unittest.main()
