# pylint: disable=unused-argument,invalid-name
import unittest

import nmmo
from nmmo.core.env import Env
from nmmo.task.predicate_api import define_predicate
from nmmo.task.task_api import Task, nmmo_default_task, make_same_tasks
from nmmo.task.group import Group
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

@define_predicate
def Fake(gs, subject, a,b,c):
  return False

class MockGameState():
  def __init__(self):
    # pylint: disable=super-init-not-called
    self.config = nmmo.config.Default()
    self.cache_result = {}
    self.get_subject_view = lambda _: None

class TestTaskAPI(unittest.TestCase):
  def test_predicate_operators(self):
    # pylint: disable=unsupported-binary-operation,invalid-unary-operand-type
    # pylint: disable=no-value-for-parameter,not-callable,no-member

    mock_gs = MockGameState()
    SUCCESS = Success(Group(0))
    FAILURE = Failure(Group(0))

    # AND (&), OR (|), NOT (~), IMPLY (>>)
    pred1 = SUCCESS & FAILURE
    self.assertFalse(pred1(mock_gs))

    pred2 = SUCCESS | FAILURE | SUCCESS
    self.assertTrue(pred2(mock_gs))

    pred3 = SUCCESS & ~ FAILURE & SUCCESS
    self.assertTrue(pred3(mock_gs))

    pred4 = SUCCESS >> SUCCESS
    self.assertTrue(pred4(mock_gs))

    pred5 = SUCCESS >> ~ SUCCESS
    self.assertFalse(pred5(mock_gs))

    pred6 = (FAILURE >> FAILURE) & SUCCESS
    self.assertTrue(pred6(mock_gs))
    self.assertEqual(pred6.name,
                     "(PAND_(IMPLY_(Failure_(0,))_(Failure_(0,)))_(Success_(0,)))")

    # predicate math
    pred7 = 0.1 * SUCCESS + 0.3
    self.assertEqual(pred7(mock_gs), 0.4)
    self.assertEqual(pred7.name,
                     "(PADD_(PMUL_(Success_(0,))_0.1)_0.3)")

    pred8 = 0.3 * SUCCESS - 1
    self.assertEqual(pred8(mock_gs), 0.0) # cannot go below 0

    pred9 = 0.3 * SUCCESS + 1
    self.assertEqual(pred9(mock_gs), 1.0) # cannot go over 1

  def test_team_assignment(self):
    team =  Group([1, 2, 8, 9], "TeamFoo")

    self.assertEqual(team.name, 'TeamFoo')
    self.assertEqual(team[2].name, "TeamFoo.2")
    self.assertEqual(team[2], (8,))

    # don't allow member of one-member team
    self.assertEqual(team[2][0].name, team[2].name)

  def test_predicate_name(self):
    # pylint: disable=no-value-for-parameter,no-member
    SUCCESS = Success(Group([0,2]))
    FAILURE = Failure(Group(0))
    fake_pred = Fake(Group(2), 1, Item.Hat, Action.Melee)
    combination = (SUCCESS & ~ (FAILURE | fake_pred)) | (FAILURE >> fake_pred)
    self.assertEqual(combination.name,
      "(POR_(PAND_(Success_(0,2))_(PNOT_(POR_(Failure_(0,))_(Fake_(2,)_1_Hat_Melee))))_"+\
      "(IMPLY_(Failure_(0,))_(Fake_(2,)_1_Hat_Melee)))")

  def test_constraint(self):
    # pylint: disable=not-callable,no-value-for-parameter
    mock_gs = MockGameState()
    good = Success(Group(0))
    bad = Success(Group(99999))
    good(mock_gs)
    self.assertRaises(InvalidConstraint,lambda: bad(mock_gs))

    scalar = ScalarConstraint(low=-10,high=10)
    for _ in range(10):
      self.assertTrue(scalar.sample(mock_gs.config)<10)
      self.assertTrue(scalar.sample(mock_gs.config)>=-10)

    bad = TickGE(Group(0), -1)
    self.assertRaises(InvalidConstraint, lambda: bad(mock_gs))

  def test_sample_predicate(self):
    # pylint: disable=no-value-for-parameter,expression-not-assigned
    predicate = CanSeeGroup() & TickGE()
    self.assertEqual(predicate.name,
                     "(PAND_(CanSeeGroup_subject:GroupConstraint_target:GroupConstraint)_"+\
                     "(TickGE_subject:GroupConstraint_num_tick:ScalarConstraint))")
    config = nmmo.config.Default()
    TickGE().sample(config)
    predicate.sample(config).name

    # DONE

  def test_task_api_with_predicate(self):
    # pylint: disable=no-value-for-parameter
    mock_gs = MockGameState()
    pred = Fake(Group(2), 1, Item.Hat, Action.Melee)
    assignee = [1,2,3] # list of agent ids
    task = Task(pred, assignee)
    rewards, infos = task.compute_rewards(mock_gs)

    self.assertEqual(task.name, # contains predicate name and assignee list
                     "(Task_eval_fn:(Fake_(2,)_1_Hat_Melee)_assignee:(1,2,3))")
    for agent_id in assignee:
      self.assertEqual(rewards[agent_id], 0)
      self.assertEqual(infos[agent_id]['progress'], 0) # progress (False -> 0)
      self.assertFalse(task.completed)

  def test_task_api_with_function(self):
    mock_gs = MockGameState()
    def eval_with_subject_fn(subject: Group):
      def is_agent_1(gs):
        return any(agent_id == 1 for agent_id in subject.agents)
      return is_agent_1

    assignee = [1,2,3] # list of agent ids
    task = Task(eval_with_subject_fn(Group(assignee)), assignee)
    rewards, infos = task.compute_rewards(mock_gs)

    self.assertEqual(task.name, # contains predicate name and assignee list
                     "(Task_eval_fn:is_agent_1_assignee:(1,2,3))")
    for agent_id in assignee:
      self.assertEqual(rewards[agent_id], 1)
      self.assertEqual(infos[agent_id]['progress'], 1) # progress (True -> 1)
      self.assertTrue(task.completed)

  def test_nmmo_default_task(self):
    config = ScriptedAgentTestConfig()
    env = Env(config)

    dafault_tasks = nmmo_default_task(env.possible_agents)
    env.reset(new_tasks=dafault_tasks)
    for _ in range(3):
      env.step({})

    # DONE

  def test_completed_tasks_in_info(self):
    # pylint: disable=no-value-for-parameter
    config = ScriptedAgentTestConfig()
    env = Env(config)

    # Use make_atomic_task(predicate, agent_list) when
    # the predicate's subject is the same as the task assignee
    same_team = [1, 2, 3, 4]
    fake_pred = Fake(Group(3), 1, Item.Hat, Action.Melee)
    tasks = make_same_tasks(pred=Success, assignee=1) # task 1
    tasks += make_same_tasks(pred=Failure, assignee=2) # task 2
    tasks += [Task(fake_pred, assignee=3), # task 3: fake_pred is already instantiated
              Task(Success(Group(same_team)), assignee=same_team)] # task 4: team task

    # tasks are all instantiated with the agent ids
    env.reset(new_tasks=tasks)
    _, _, _, infos = env.step({})

    # agent 1: assigned only task 1, which is always True
    self.assertEqual(infos[1]['task'][tasks[0].name]['reward'], 1.0)
    for i in [1, 2]: # task 2 and 3
      self.assertTrue(tasks[i].name not in infos[1]['task'])

    # agent 2: assigned task 2 (Failure) and task 4 (Success)
    self.assertEqual(infos[2]['task'][tasks[1].name]['reward'], 0.0) # task 2
    self.assertEqual(infos[2]['task'][tasks[3].name]['reward'], 1.0) # task 4

    # agent 3 assigned task 3, Fake(), which is always False (0)
    self.assertEqual(infos[3]['task'][tasks[2].name]['reward'], 0.0) # task 3

    # all agents in the same team with agent 2 have SUCCESS
    # other agents don't have any tasks assigned
    for ent_id in env.possible_agents:
      if ent_id in same_team:
        self.assertEqual(infos[ent_id]['task'][tasks[3].name]['reward'], 1.0)
      else:
        self.assertTrue(tasks[3].name not in infos[ent_id]['task'])

    # DONE

if __name__ == '__main__':
  unittest.main()
