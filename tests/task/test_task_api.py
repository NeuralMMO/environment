# pylint: disable=unused-argument,invalid-name
import unittest
from types import FunctionType

import nmmo
from nmmo.core.env import Env
from nmmo.task.predicate_api import make_predicate, Predicate
from nmmo.task.task_api import Task, make_team_tasks
from nmmo.task.group import Group
from nmmo.task.constraint import InvalidConstraint, ScalarConstraint
from nmmo.task.base_predicates import TickGE, CanSeeGroup, AllMembersWithinRange

from nmmo.systems import item as Item
from nmmo.core import action as Action

from scripted.baselines import Sleeper
from tests.testhelpers import ScriptedAgentTestConfig, change_spawn_pos

# define predicates in the function form
#   with the required signatures: gs, subject
def Success(gs, subject: Group):
  return True

def Failure(gs, subject: Group):
  return False

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

    self.assertTrue(isinstance(Success, FunctionType))
    self.assertTrue(isinstance(Failure, FunctionType))

    # make predicate class from function
    success_pred_cls = make_predicate(Success)
    failure_pred_cls = make_predicate(Failure)
    self.assertTrue(isinstance(success_pred_cls, type)) # class
    self.assertTrue(isinstance(failure_pred_cls, type))

    # then instantiate predicates
    SUCCESS = success_pred_cls(Group(0))
    FAILURE = failure_pred_cls(Group(0))
    self.assertTrue(isinstance(SUCCESS, Predicate))
    self.assertTrue(isinstance(FAILURE, Predicate))

    # NOTE: only the instantiated predicate can be used with operators like below
    mock_gs = MockGameState()

    # AND (&), OR (|), NOT (~)
    pred1 = SUCCESS & FAILURE
    self.assertFalse(pred1(mock_gs))

    pred2 = SUCCESS | FAILURE | SUCCESS
    self.assertTrue(pred2(mock_gs))

    pred3 = SUCCESS & ~ FAILURE & SUCCESS
    self.assertTrue(pred3(mock_gs))

    # predicate math
    pred4 = 0.1 * SUCCESS + 0.3
    self.assertEqual(pred4(mock_gs), 0.4)
    self.assertEqual(pred4.name,
                     "(ADD_(MUL_(Success_(0,))_0.1)_0.3)")

    pred5 = 0.3 * SUCCESS - 1
    self.assertEqual(pred5(mock_gs), 0.0) # cannot go below 0

    pred6 = 0.3 * SUCCESS + 1
    self.assertEqual(pred6(mock_gs), 1.0) # cannot go over 1

  def test_team_assignment(self):
    team =  Group([1, 2, 8, 9], "TeamFoo")

    self.assertEqual(team.name, 'TeamFoo')
    self.assertEqual(team[2].name, "TeamFoo.2")
    self.assertEqual(team[2], (8,))

    # don't allow member of one-member team
    self.assertEqual(team[2][0].name, team[2].name)

  def test_predicate_name(self):
    # pylint: disable=no-value-for-parameter,no-member
    # make predicate class from function
    success_pred_cls = make_predicate(Success)
    failure_pred_cls = make_predicate(Failure)
    fake_pred_cls = make_predicate(Fake)

    # instantiate the predicates
    SUCCESS = success_pred_cls(Group([0,2]))
    FAILURE = failure_pred_cls(Group(0))
    fake_pred = fake_pred_cls(Group(2), 1, Item.Hat, Action.Melee)
    combination = (SUCCESS & ~ (FAILURE | fake_pred)) | (FAILURE * fake_pred + .3) - .4
    self.assertEqual(combination.name,
      "(OR_(AND_(Success_(0,2))_(NOT_(OR_(Failure_(0,))_(Fake_(2,)_1_Hat_Melee))))_"+\
      "(SUB_(ADD_(MUL_(Failure_(0,))_(Fake_(2,)_1_Hat_Melee))_0.3)_0.4))")

  def test_constraint(self):
    # pylint: disable=not-callable,no-value-for-parameter
    # define predicate classes from functions

    # make predicate class from function
    success_pred_cls = make_predicate(Success)
    tickge_pred_cls = make_predicate(TickGE)
    self.assertTrue(isinstance(TickGE, FunctionType))

    mock_gs = MockGameState()
    good = success_pred_cls(Group(0))
    bad = success_pred_cls(Group(99999))
    good(mock_gs)
    self.assertRaises(InvalidConstraint,lambda: bad(mock_gs))

    scalar = ScalarConstraint(low=-10,high=10)
    for _ in range(10):
      self.assertTrue(scalar.sample(mock_gs.config)<10)
      self.assertTrue(scalar.sample(mock_gs.config)>=-10)

    bad = tickge_pred_cls(Group(0), -1)
    self.assertRaises(InvalidConstraint, lambda: bad(mock_gs))

  def test_sample_predicate(self):
    # pylint: disable=no-value-for-parameter,expression-not-assigned
    # make predicate class from function
    canseegrp_pred_cls = make_predicate(CanSeeGroup)
    tickge_pred_cls = make_predicate(TickGE)

    # if the predicate class is instantiated without the subject,
    mock_gs = MockGameState()
    predicate = canseegrp_pred_cls() & tickge_pred_cls()
    self.assertEqual(predicate.name,
                     "(AND_(CanSeeGroup_subject:GroupConstraint_target:AgentListConstraint)_"+\
                     "(TickGE_subject:GroupConstraint_num_tick:ScalarConstraint))")

    # this predicate cannot calculate progress becuase it has no subject
    with self.assertRaises(AttributeError):
      predicate(mock_gs)

    # this predicate supports sampling with valid arguments
    config = nmmo.config.Default()
    tickge_pred_cls().sample(config)
    predicate.sample(config).name

    # DONE

  def test_task_api_with_predicate(self):
    # pylint: disable=no-value-for-parameter,no-member
    fake_pred_cls = make_predicate(Fake)

    mock_gs = MockGameState()
    predicate = fake_pred_cls(Group(2), 1, Item.Hat, Action.Melee)
    assignee = [1,2,3] # list of agent ids
    task = predicate.create_task(assignee=assignee)
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

  def test_predicate_fn_using_other_predicate_fn(self):
    # define a predicate: to form a tight formation, for a certain number of ticks
    def PracticeFormation(gs, subject, dist, num_tick):
      return AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick)

    # team should stay together within 1 tile for 10 ticks
    goal_tick = 10
    task_spec = ('team', PracticeFormation, {'dist': 1, 'num_tick': goal_tick})

    # create the test task from the task spec
    teams = {0:[1,2,3], 1:[4,5], 2:[6,7], 3:[8,9], 4:[10,11]}

    config = ScriptedAgentTestConfig()
    config.PLAYERS =[Sleeper]
    config.IMMORTAL = True

    env = Env(config)
    env.reset(make_task_fn=lambda: make_team_tasks(teams, [task_spec]))

    # move agent 2, 3 to agent 1's pos
    for agent_id in [2,3]:
      change_spawn_pos(env.realm, agent_id,
                       env.realm.players[1].pos)

    for tick in range(goal_tick+2):
      _, rewards, _, infos = env.step({})

      if tick < 10:
        self.assertAlmostEqual(rewards[1], 1/goal_tick)
        self.assertAlmostEqual((1+tick)/goal_tick,
                               infos[1]['task'][env.tasks[0].name]['progress'])
      else:
        # tick 11, task should be completed
        self.assertEqual(rewards[1], 0)
        self.assertEqual(infos[1]['task'][env.tasks[0].name]['progress'], 1)
        self.assertEqual(infos[1]['task'][env.tasks[0].name]['completed'], True)

  def test_completed_tasks_in_info(self):
    # pylint: disable=no-value-for-parameter,no-member
    config = ScriptedAgentTestConfig()
    env = Env(config)

    # make predicate class from function
    success_pred_cls = make_predicate(Success)
    failure_pred_cls = make_predicate(Failure)
    fake_pred_cls = make_predicate(Fake)

    # instantiate the predicates
    same_team = [1, 2, 3, 4]
    predicates = [
      success_pred_cls(Group(1)), # task 1
      failure_pred_cls(Group(2)), # task 2
      fake_pred_cls(Group(3), 1, Item.Hat, Action.Melee), # task 3
      success_pred_cls(Group(same_team))] # task 4

    # tasks can be created directly from predicate instances
    test_tasks = [pred.create_task() for pred in predicates]

    # tasks are all instantiated with the agent ids
    env.reset(make_task_fn=lambda: test_tasks)
    _, _, _, infos = env.step({})

    # agent 1: assigned only task 1, which is always True
    self.assertEqual(infos[1]['task'][env.tasks[0].name]['reward'], 1.0)
    for i in [1, 2]: # task 2 and 3
      self.assertTrue(env.tasks[i].name not in infos[1]['task'])

    # agent 2: assigned task 2 (Failure) and task 4 (Success)
    self.assertEqual(infos[2]['task'][env.tasks[1].name]['reward'], 0.0) # task 2
    self.assertEqual(infos[2]['task'][env.tasks[3].name]['reward'], 1.0) # task 4

    # agent 3 assigned task 3, Fake(), which is always False (0)
    self.assertEqual(infos[3]['task'][env.tasks[2].name]['reward'], 0.0) # task 3

    # all agents in the same team with agent 2 have SUCCESS
    # other agents don't have any tasks assigned
    for ent_id in env.possible_agents:
      if ent_id in same_team:
        self.assertEqual(infos[ent_id]['task'][env.tasks[3].name]['reward'], 1.0)
      else:
        self.assertTrue(env.tasks[3].name not in infos[ent_id]['task'])

    # DONE

if __name__ == '__main__':
  unittest.main()
