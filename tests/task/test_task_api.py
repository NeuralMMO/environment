# pylint: disable=unused-argument,invalid-name
import unittest
from types import FunctionType
import numpy as np

import nmmo
from nmmo.core.env import Env
from nmmo.task.predicate_api import make_predicate, Predicate
from nmmo.task.task_api import Task, OngoingTask, HoldDurationTask
from nmmo.task.task_spec import TaskSpec, make_task_from_spec
from nmmo.task.group import Group
from nmmo.task.constraint import ScalarConstraint, GroupConstraint, AGENT_LIST_CONSTRAINT
from nmmo.task.base_predicates import (
    TickGE, CanSeeGroup, AllMembersWithinRange, StayAlive, HoardGold
)

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
    self.current_tick = -1
    self.cache_result = {}
    self.get_subject_view = lambda _: None

  def clear_cache(self):
    pass

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

    # get the individual predicate"s source code
    self.assertEqual(SUCCESS.get_source_code(),
                     "def Success(gs, subject: Group):\n  return True")
    self.assertEqual(FAILURE.get_source_code(),
                     "def Failure(gs, subject: Group):\n  return False")

    # AND (&), OR (|), NOT (~)
    pred1 = SUCCESS & FAILURE
    self.assertFalse(pred1(mock_gs))
    # NOTE: get_source_code() of the combined predicates returns the joined str
    #   of each predicate"s source code, which may NOT represent what the actual
    #   predicate is doing
    self.assertEqual(pred1.get_source_code(),
                     "def Success(gs, subject: Group):\n  return True\n\n"+
                     "def Failure(gs, subject: Group):\n  return False")

    pred2 = SUCCESS | FAILURE | SUCCESS
    self.assertTrue(pred2(mock_gs))
    self.assertEqual(pred2.get_source_code(),
                     "def Success(gs, subject: Group):\n  return True\n\n"+
                     "def Failure(gs, subject: Group):\n  return False\n\n"+
                     "def Success(gs, subject: Group):\n  return True")

    pred3 = SUCCESS & ~ FAILURE & SUCCESS
    self.assertTrue(pred3(mock_gs))
    # NOTE: demonstrating the above point -- it just returns the functions
    #   NOT what this predicate actually evaluates.
    self.assertEqual(pred2.get_source_code(),
                     pred3.get_source_code())

    # predicate math
    pred4 = 0.1 * SUCCESS + 0.3
    self.assertEqual(pred4(mock_gs), 0.4)
    self.assertEqual(pred4.name,
                     "(ADD_(MUL_(Success_(0,))_0.1)_0.3)")
    # NOTE: demonstrating the above point again, -- it just returns the functions
    #   NOT what this predicate actually evaluates.
    self.assertEqual(pred4.get_source_code(),
                     "def Success(gs, subject: Group):\n  return True")

    pred5 = 0.3 * SUCCESS - 1
    self.assertEqual(pred5(mock_gs), 0.0) # cannot go below 0

    pred6 = 0.3 * SUCCESS + 1
    self.assertEqual(pred6(mock_gs), 1.0) # cannot go over 1

  def test_team_assignment(self):
    team =  Group([1, 2, 8, 9], "TeamFoo")

    self.assertEqual(team.name, "TeamFoo")
    self.assertEqual(team[2].name, "TeamFoo.2")
    self.assertEqual(team[2], (8,))

    # don"t allow member of one-member team
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
    mock_gs = MockGameState()
    scalar = ScalarConstraint(low=-10,high=10)
    for _ in range(10):
      self.assertTrue(scalar.sample(mock_gs.config)<10)
      self.assertTrue(scalar.sample(mock_gs.config)>=-10)

  def test_sample_predicate(self):
    # pylint: disable=no-value-for-parameter,expression-not-assigned
    # make predicate class from function
    canseegrp_pred_cls = make_predicate(CanSeeGroup)
    tickge_pred_cls = make_predicate(TickGE)

    # if the predicate class is instantiated without the subject,
    mock_gs = MockGameState()
    predicate = canseegrp_pred_cls(subject=GroupConstraint, target=AGENT_LIST_CONSTRAINT) &\
                tickge_pred_cls(subject=GroupConstraint, num_tick=ScalarConstraint)
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
    group = Group(2)
    item = Item.Hat
    action = Action.Melee
    predicate = fake_pred_cls(group, a=1, b=item, c=action)
    self.assertEqual(predicate.get_source_code(),
                     "def Fake(gs, subject, a,b,c):\n  return False")
    self.assertEqual(predicate.get_signature(), ["gs", "subject", "a", "b", "c"])
    self.assertEqual(predicate.args, [group])
    self.assertDictEqual(predicate.kwargs, {"a": 1, "b": item, "c": action})

    assignee = [1,2,3] # list of agent ids
    task = predicate.create_task(assignee=assignee)
    rewards, infos = task.compute_rewards(mock_gs)

    self.assertEqual(task.name, # contains predicate name and assignee list
                     "(Task_eval_fn:(Fake_(2,)_a:1_b:Hat_c:Melee)_assignee:(1,2,3))")
    self.assertEqual(task.get_source_code(),
                     "def Fake(gs, subject, a,b,c):\n  return False")
    self.assertEqual(task.get_signature(), ["gs", "subject", "a", "b", "c"])
    self.assertEqual(task.args, [group])
    self.assertDictEqual(task.kwargs, {"a": 1, "b": item, "c": action})
    for agent_id in assignee:
      self.assertEqual(rewards[agent_id], 0)
      self.assertEqual(infos[agent_id]["progress"], 0) # progress (False -> 0)
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
    self.assertEqual(task.get_source_code(),
                     "def is_agent_1(gs):\n        " +
                     "return any(agent_id == 1 for agent_id in subject.agents)")
    self.assertEqual(task.get_signature(), ["gs"])
    self.assertEqual(task.args, [])
    self.assertDictEqual(task.kwargs, {})
    self.assertEqual(task.subject, tuple(assignee))
    self.assertEqual(task.assignee, tuple(assignee))
    for agent_id in assignee:
      self.assertEqual(rewards[agent_id], 1)
      self.assertEqual(infos[agent_id]["progress"], 1) # progress (True -> 1)
      self.assertTrue(task.completed)

  def test_predicate_fn_using_other_predicate_fn(self):
    # define a predicate: to form a tight formation, for a certain number of ticks
    def PracticeFormation(gs, subject, dist, num_tick):
      return AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick)

    # team should stay together within 1 tile for 10 ticks
    goal_tick = 10
    task_spec = TaskSpec(eval_fn=PracticeFormation,
                         eval_fn_kwargs={"dist": 1, "num_tick": goal_tick},
                         reward_to="team")

    # create the test task from the task spec
    teams = {1:[1,2,3], 3:[4,5], 6:[6,7], 9:[8,9], 14:[10,11]}
    team_ids= list(teams.keys())

    config = ScriptedAgentTestConfig()
    config.PLAYERS =[Sleeper]
    config.IMMORTAL = True

    env = Env(config)
    env.reset(make_task_fn=lambda: make_task_from_spec(teams, [task_spec]))

    # check the task information
    task = env.tasks[0]
    self.assertEqual(task.name,
                     "(Task_eval_fn:(PracticeFormation_(1,2,3)_dist:1_num_tick:10)"+
                     "_assignee:(1,2,3))")
    self.assertEqual(task.get_source_code(),
                     "def PracticeFormation(gs, subject, dist, num_tick):\n      "+
                     "return AllMembersWithinRange(gs, subject, dist) * "+
                     "TickGE(gs, subject, num_tick)")
    self.assertEqual(task.get_signature(), ["gs", "subject", "dist", "num_tick"])
    self.assertEqual(task.subject, tuple(teams[team_ids[0]]))
    self.assertEqual(task.kwargs, task_spec.eval_fn_kwargs)
    self.assertEqual(task.assignee, tuple(teams[team_ids[0]]))

    # check the agent-task map
    for agent_id, agent_tasks in env.agent_task_map.items():
      for task in agent_tasks:
        self.assertTrue(agent_id in task.assignee)

    # move agent 2, 3 to agent 1"s pos
    for agent_id in [2,3]:
      change_spawn_pos(env.realm, agent_id,
                       env.realm.players[1].pos)

    for tick in range(goal_tick+2):
      _, rewards, _, infos = env.step({})

      if tick < 10:
        target_reward = 1.0 if env.realm.tick == goal_tick else 1/goal_tick
        self.assertAlmostEqual(rewards[1], target_reward)
        self.assertAlmostEqual((1+tick)/goal_tick,
                               infos[1]["task"][env.tasks[0].name]["progress"])
      else:
        # tick 11, task should be completed
        self.assertEqual(rewards[1], 0)
        self.assertEqual(infos[1]["task"][env.tasks[0].name]["progress"], 1)
        self.assertEqual(infos[1]["task"][env.tasks[0].name]["completed"], True)

    # test the task_spec_with_embedding
    task_embedding = np.ones(config.TASK_EMBED_DIM, dtype=np.float16)
    task_spec_with_embedding = TaskSpec(eval_fn=PracticeFormation,
                                        eval_fn_kwargs={"dist": 1, "num_tick": goal_tick},
                                        reward_to="team",
                                        embedding=task_embedding)
    env.reset(make_task_fn=lambda: make_task_from_spec(teams, [task_spec_with_embedding]))

    task = env.tasks[0]
    self.assertEqual(task.spec_name, # without the subject and assignee agent ids
                     "Task_PracticeFormation_(dist:1_num_tick:10)_reward_to:team")
    self.assertEqual(task.name,
                     "(Task_eval_fn:(PracticeFormation_(1,2,3)_dist:1_num_tick:10)"+
                     "_assignee:(1,2,3))")
    self.assertEqual(task.get_source_code(),
                     "def PracticeFormation(gs, subject, dist, num_tick):\n      "+
                     "return AllMembersWithinRange(gs, subject, dist) * "+
                     "TickGE(gs, subject, num_tick)")
    self.assertEqual(task.get_signature(), ["gs", "subject", "dist", "num_tick"])
    self.assertEqual(task.subject, tuple(teams[team_ids[0]]))
    self.assertEqual(task.kwargs, task_spec.eval_fn_kwargs)
    self.assertEqual(task.assignee, tuple(teams[team_ids[0]]))
    self.assertTrue(np.array_equal(task.embedding, task_embedding))

    obs_spec = env.observation_space(1)
    self.assertTrue(obs_spec["Task"].contains(task.embedding))

  def test_completed_tasks_in_info(self):
    # pylint: disable=no-value-for-parameter,no-member
    config = ScriptedAgentTestConfig()
    config.ALLOW_MULTI_TASKS_PER_AGENT = True
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
    self.assertEqual(infos[1]["task"][env.tasks[0].name]["reward"], 1.0)
    for i in [1, 2]: # task 2 and 3
      self.assertTrue(env.tasks[i].name not in infos[1]["task"])

    # agent 2: assigned task 2 (Failure) and task 4 (Success)
    self.assertEqual(infos[2]["task"][env.tasks[1].name]["reward"], 0.0) # task 2
    self.assertEqual(infos[2]["task"][env.tasks[3].name]["reward"], 1.0) # task 4

    # agent 3 assigned task 3, Fake(), which is always False (0)
    self.assertEqual(infos[3]["task"][env.tasks[2].name]["reward"], 0.0) # task 3

    # all agents in the same team with agent 2 have SUCCESS
    # other agents don"t have any tasks assigned
    for ent_id in env.possible_agents:
      if ent_id in same_team:
        self.assertEqual(infos[ent_id]["task"][env.tasks[3].name]["reward"], 1.0)
      else:
        self.assertTrue(env.tasks[3].name not in infos[ent_id]["task"])

    # DONE

  def test_make_task_from_spec(self):
    teams = {0:[1,2,3], 1:[4,5,6]}
    test_embedding = np.array([1,2,3])
    task_spec = [
      TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": 20}),
      TaskSpec(eval_fn=StayAlive, eval_fn_kwargs={}, task_cls=OngoingTask),
      TaskSpec(eval_fn=StayAlive, eval_fn_kwargs={"target": "my_team_leader"},
               task_cls=OngoingTask, reward_to="team"),
      TaskSpec(eval_fn=StayAlive, eval_fn_kwargs={"target": "left_team"},
               task_cls=OngoingTask, task_kwargs={"reward_multiplier": 2},
               reward_to="team", embedding=test_embedding),
    ]

    task_list = []
    # testing each task spec, individually
    for single_spec in task_spec:
      task_list.append(make_task_from_spec(teams, [single_spec]))

    # check the task spec names
    self.assertEqual(task_list[0][0].spec_name,
                     "Task_TickGE_(num_tick:20)_reward_to:agent")
    self.assertEqual(task_list[1][0].spec_name,
                     "OngoingTask_StayAlive_()_reward_to:agent")
    self.assertEqual(task_list[2][0].spec_name,
                     "OngoingTask_StayAlive_(target:my_team_leader)_reward_to:team")
    self.assertEqual(task_list[3][0].spec_name,
                     "OngoingTask_StayAlive_(target:left_team)_reward_to:team")

    # check the task names
    self.assertEqual(task_list[0][0].name,
                     "(Task_eval_fn:(TickGE_(1,)_num_tick:20)_assignee:(1,))")
    self.assertEqual(task_list[1][0].name,
                     "(OngoingTask_eval_fn:(StayAlive_(1,))_assignee:(1,))")
    self.assertEqual(task_list[2][0].name,
                     "(OngoingTask_eval_fn:(StayAlive_(1,))_assignee:(1,2,3))")
    self.assertEqual(task_list[3][0].name,
                     "(OngoingTask_eval_fn:(StayAlive_(4,5,6))_assignee:(1,2,3))")
    self.assertEqual(task_list[3][0].reward_multiplier, 2)
    self.assertTrue(np.array_equal(task_list[3][0].embedding, np.array([1,2,3])))

  def test_hold_duration_task(self):
    # pylint: disable=protected-access
    # each agent should hoard gold for 10 ticks
    goal_tick = goal_gold = 10
    task_spec = [TaskSpec(eval_fn=HoardGold,
                          eval_fn_kwargs={"amount": goal_gold},
                          task_cls=HoldDurationTask,
                          task_kwargs={"hold_duration": goal_tick})] * 3

    config = ScriptedAgentTestConfig()
    config.PLAYERS =[Sleeper]
    config.IMMORTAL = True

    teams = {id: [id] for id in range(1,4)}
    env = Env(config)
    env.reset(make_task_fn=lambda: make_task_from_spec(teams, task_spec))

    # give agent 1, 2 enough gold
    for agent_id in [1,2]:
      env.realm.players[agent_id].gold.update(goal_gold+1)

    for _ in range(5):
      env.step({})

    # check the task information
    self.assertEqual(env.tasks[0].spec_name,
                     "HoldDurationTask_HoardGold_(amount:10)_reward_to:agent")
    for idx in [0, 1]:
      self.assertEqual(env.tasks[idx]._progress, 0.5) # agent 1 & 2 has enough gold
      self.assertEqual(env.tasks[idx]._max_progress, 0.5)
      self.assertEqual(env.tasks[idx].reward_signal_count, 5)
    self.assertTrue(env.tasks[2]._progress == 0.0) # agent 3 has no gold
    for task in env.tasks:
      self.assertTrue(task.completed is False) # not completed yet

    # take away gold from agent 2
    env.realm.players[2].gold.update(goal_gold-1)

    env.step({})
    self.assertEqual(env.tasks[0]._progress, 0.6) # agent 1 has enough gold
    self.assertEqual(env.tasks[0]._max_progress, 0.6)
    self.assertEqual(env.tasks[0].reward_signal_count, 6)
    self.assertEqual(env.tasks[1]._progress, 0) # agent 2 has not enough gold
    self.assertEqual(env.tasks[1]._max_progress, 0.5) # max values are preserved
    self.assertEqual(env.tasks[1]._positive_reward_count, 5)
    self.assertEqual(env.tasks[1].reward_signal_count, 6) # 5 positive + 1 negative

    for _ in range(4):
      env.step({})

    # only agent 1 successfully held 10 gold for 10 ticks
    self.assertTrue(env.tasks[0].completed is True)
    self.assertTrue(env.tasks[1].completed is False)
    self.assertTrue(env.tasks[2].completed is False)

  def test_task_spec_with_predicate(self):
    teams = {0:[1,2,3], 1:[4,5,6]}
    SUCCESS = make_predicate(Success)(Group(1))
    FAILURE = make_predicate(Failure)(Group([2,3]))
    predicate = SUCCESS & FAILURE
    predicate.name = "SuccessAndFailure"

    # make task spec
    task_spec = [TaskSpec(predicate=predicate,
                          eval_fn=None,
                          eval_fn_kwargs={"success_target": 1,
                                          "test_item": Item.Hat})]
    tasks = make_task_from_spec(teams, task_spec)

    env = Env(ScriptedAgentTestConfig())
    env.reset(make_task_fn=lambda: tasks)
    env.step({})

    # check the task information
    self.assertEqual(env.tasks[0].spec_name,
                     "Task_SuccessAndFailure_(success_target:1_test_item:Hat)_reward_to:agent")

if __name__ == "__main__":
  unittest.main()
