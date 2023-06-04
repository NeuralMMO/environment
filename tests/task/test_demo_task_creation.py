# pylint: disable=invalid-name,unused-argument,unused-variable
import unittest
from tests.testhelpers import ScriptedAgentTestConfig

from nmmo.core.env import Env
from nmmo.lib.log import EventCode
from nmmo.systems import skill
from nmmo.task import predicate_api as p
from nmmo.task import task_api as t
from nmmo.task import base_predicates as bp
from nmmo.task.game_state import GameState
from nmmo.task.group import Group
from nmmo.task.task_api import make_team_tasks


def rollout(env, tasks, steps=5):
  env.reset(new_tasks=tasks)
  for _ in range(steps):
    env.step({})
  return env.step({})

class TestDemoTask(unittest.TestCase):

  def test_baseline_tasks(self):
    # Tasks from
    # https://github.com/NeuralMMO/baselines/
    # blob/4c1088d2bbe0f74a08dcf7d71b714cd30772557f/tasks.py
    class Tier:
      REWARD_SCALE = 15
      EASY         = 4 / REWARD_SCALE
      NORMAL       = 6 / REWARD_SCALE
      HARD         = 11 / REWARD_SCALE

    # Predicates defined below can be evaluated over one agent or several agents,
    #   which are sepcified separately
    # Reward multiplier is indendent from predicates and used by tasks.
    #   The multipliers are just shown to indicate the difficulty level of predicates

    # Usage of base predicates (see nmmo/task/base_predicates.py)
    player_kills = [ # (predicate, kwargs, reward_multiplier)
      (bp.CountEvent, {'event': 'PLAYER_KILL', 'N': 1}, Tier.EASY),
      (bp.CountEvent, {'event': 'PLAYER_KILL', 'N': 2}, Tier.NORMAL),
      (bp.CountEvent, {'event': 'PLAYER_KILL', 'N': 3}, Tier.HARD)]

    exploration = [ # (predicate, reward_multiplier)
      (bp.DistanceTraveled, {'dist': 16}, Tier.EASY),
      (bp.DistanceTraveled, {'dist': 32}, Tier.NORMAL),
      (bp.DistanceTraveled, {'dist': 64}, Tier.HARD)]

    # Demonstrates custom predicate - return float/boolean
    @p.define_predicate
    def EquipmentLevel(gs: GameState,
                       subject: Group,
                       number: int):
      equipped = subject.item.equipped > 0
      levels = subject.item.level[equipped]
      return levels.sum() >= number

    equipment = [ # (predicate, reward_multiplier)
      (EquipmentLevel, {'number': 1}, Tier.EASY),
      (EquipmentLevel, {'number': 5}, Tier.NORMAL),
      (EquipmentLevel, {'number': 10}, Tier.HARD)]

    @p.define_predicate
    def CombatSkill(gs, subject, lvl):
      # using predicate OR
      return p.OR(bp.AttainSkill(subject, skill.Melee, lvl, 1),
                  bp.AttainSkill(subject, skill.Range, lvl, 1),
                  bp.AttainSkill(subject, skill.Mage, lvl, 1))

    combat = [ # (predicate, reward_multiplier)
      (CombatSkill, {'lvl': 2}, Tier.EASY),
      (CombatSkill, {'lvl': 3}, Tier.NORMAL),
      (CombatSkill, {'lvl': 4}, Tier.HARD)]

    @p.define_predicate
    def ForageSkill(gs, subject, lvl):
      return p.OR(bp.AttainSkill(subject, skill.Fishing, lvl, 1),
                  bp.AttainSkill(subject, skill.Herbalism, lvl, 1),
                  bp.AttainSkill(subject, skill.Prospecting, lvl, 1),
                  bp.AttainSkill(subject, skill.Carving, lvl, 1),
                  bp.AttainSkill(subject, skill.Alchemy, lvl, 1))

    foraging = [ # (predicate, reward_multiplier)
      (ForageSkill, {'lvl': 2}, Tier.EASY),
      (ForageSkill, {'lvl': 3}, Tier.NORMAL),
      (ForageSkill, {'lvl': 4}, Tier.HARD)]

    # Test rollout
    config = ScriptedAgentTestConfig()
    env = Env(config)

    # Creating and testing "team" tasks
    # i.e., predicates are evalauated over all team members,
    #   and all team members get the same reward from each task

    # The team mapping can come from anywhere.
    # The below is an arbitrary example and even doesn't include all agents
    teams = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8]}

    # Making player_kills and exploration team tasks,
    team_tasks = []
    for pred, kwargs, weight in player_kills + exploration:
      for team in teams.values():
        team_tasks.append(
          pred(Group(team), **kwargs).create_task(reward_multiplier=weight))

    # Run the environment with these tasks
    #   check rewards and infos for the task info
    obs, rewards, dones, infos = rollout(env, team_tasks)

    # Creating and testing the same task for all agents
    # i.e, each agent gets evaluated and rewarded individually
    same_tasks = []
    for pred, kwargs, weight in exploration + equipment + combat + foraging:
      for agent_id in env.possible_agents:
        same_tasks.append(
          pred(Group([agent_id]), **kwargs).create_task(reward_multiplier=weight))

    # Run the environment with these tasks
    #   check rewards and infos for the task info
    obs, rewards, dones, infos = rollout(env, same_tasks)

    # DONE

  def test_player_kill_reward(self):
    # pylint: disable=no-value-for-parameter
    """ Design a predicate with a complex progress scheme
    """
    config = ScriptedAgentTestConfig()
    env = Env(config)

    # PARTICIPANT WRITES
    # ====================================
    @p.define_predicate
    def KillPredicate(gs: GameState,
                      subject: Group):
      """The progress, the max of which is 1, should
           * increase small for each player kill
           * increase big for the 1st and 3rd kills
           * reach 1 with 10 kills
      """
      num_kills = len(subject.event.PLAYER_KILL)
      progress = num_kills * 0.06
      if num_kills >= 1:
        progress += .1
      if num_kills >= 3:
        progress += .3
      return min(progress, 1.0)

    kill_tasks = [KillPredicate(Group(agent_id)).create_task()
                  for agent_id in env.possible_agents]

    # Test Reward
    env.reset(new_tasks=kill_tasks)
    players = env.realm.players
    code = EventCode.PLAYER_KILL
    env.realm.event_log.record(code, players[1], target=players[3])
    env.realm.event_log.record(code, players[2], target=players[4])
    env.realm.event_log.record(code, players[2], target=players[5])
    env.realm.event_log.record(EventCode.EAT_FOOD, players[2])

    # Award given as designed
    # Agent 1 kills 1 - reward .06 + .1
    # Agent 2 kills 2 - reward .12 + .1
    # Agent 3 kills 0 - reward 0
    _, rewards, _, _ = env.step({})
    self.assertEqual(rewards[1], 0.16)
    self.assertEqual(rewards[2], 0.22)
    self.assertEqual(rewards[3], 0)

    # No reward when no changes
    _, rewards, _, _ = env.step({})
    self.assertEqual(rewards[1], 0)
    self.assertEqual(rewards[2], 0)
    self.assertEqual(rewards[3], 0)

    # DONE

  def test_predicate_math(self):
    # pylint: disable=no-value-for-parameter
    config = ScriptedAgentTestConfig()
    env = Env(config)

    @p.define_predicate
    def PredicateMath(gs, subject):
      progress = 0.8 * bp.CountEvent(subject, event='PLAYER_KILL', N=7) + \
                 1.1 * bp.TickGE(subject, num_tick=3)
      # NOTE: the resulting progress will be bounded from [0, 1] afterwards
      return progress

    task_for_agent_1 = PredicateMath(Group(1)).create_task()

    # Test Reward
    env.reset(new_tasks=[task_for_agent_1])
    code = EventCode.PLAYER_KILL
    players = env.realm.players
    env.realm.event_log.record(code, players[1], target=players[2])
    env.realm.event_log.record(code, players[1], target=players[3])

    _, rewards, _, _ = env.step({})
    self.assertAlmostEqual(rewards[1], 0.8*2/7 + 1.1*1/3)

    for _ in range(2):
      _, _, _, infos = env.step({})

    # 0.8*2/7 + 1.1 > 1, but the progress is maxed at 1
    self.assertEqual(infos[1]['task'][env.tasks[0].name]['progress'], 1.0)
    self.assertTrue(env.tasks[0].completed) # because progress >= 1

    # DONE

  def test_make_team_tasks_inside_reset(self):
    # NOTE: len(teams) and len(task_spec) don't need to match
    teams = {0:[1,2,3], 1:[4,5], 2:[6,7], 3:[8,9], 4:[10,11]}

    # custom function to turn into predicate inside reset
    def custom_predicate_func(gs, subject, test):
      return True

    """ task_spec is a list of tuple (reward_to, predicate class, kwargs)

        each tuple in the task_spec will create tasks for a team in teams

        reward_to: must be in ['team', 'agent']
          * 'team' create a single team task, in which all team members get rewarded
          * 'agent' create a task for each agent, in which only the agent gets rewarded

        predicate class from the base predicates or custom predicates like above

        kwargs are the additional args that go into predicate. There are also special keys
          * 'target' must be ['left_team', 'right_team', 'left_team_leader', 'right_team_leader']
             these str will be translated into the actual agent ids
          * 'task_cls' is optional. If not provided, the standard Task is used. """
    task_spec = [ # (reward_to, predicate class, kwargs)
      ('team', bp.CountEvent, {'event': 'PLAYER_KILL', 'N': 1}), # one task
      ('agent', bp.CountEvent, {'event': 'PLAYER_KILL', 'N': 2}),
      ('agent', bp.AllDead, {'target': 'left_team'}),
      ('team', bp.CanSeeAgent, {'target': 'right_team_leader', 'task_cls': t.OngoingTask}),
      ('team', custom_predicate_func, {'test': 1})]

    config = ScriptedAgentTestConfig()
    env = Env(config)

    env.reset(make_task_fn=make_team_tasks,
              make_task_fn_kwargs={'task_spec':task_spec, 'teams':teams})

    self.assertEqual(len(env.tasks), 7) # 7 tasks were created
    self.assertEqual(env.tasks[0].name, # team 0 task assigned to agents 1,2,3
                     '(Task_eval_fn:(CountEvent_(1,2,3)_event:PLAYER_KILL_N:1)_assignee:(1,2,3))')
    self.assertEqual(env.tasks[1].name, # team 1, agent task assigned to agent 4
                     '(Task_eval_fn:(CountEvent_(4,)_event:PLAYER_KILL_N:2)_assignee:(4,))')
    self.assertEqual(env.tasks[2].name, # team 1, agent task assigned to agent 5
                     '(Task_eval_fn:(CountEvent_(5,)_event:PLAYER_KILL_N:2)_assignee:(5,))')
    self.assertEqual(env.tasks[3].name, # team 2, agent 6 task, left_team is team 3 (agents 8,9)
                     '(Task_eval_fn:(AllDead_(8,9))_assignee:(6,))')
    self.assertEqual(env.tasks[5].name, # team 3 task, right_team is team 2 (6,7), leader 6
                     '(OngoingTask_eval_fn:(CanSeeAgent_(8,9)_target:6)_assignee:(8,9))')
    self.assertEqual(env.tasks[6].name, # team 4 task, based on a predicate function
                     '(Task_eval_fn:(custom_predicate_func_(10,11)_test:1)_assignee:(10,11))')

    for _ in range(2):
      env.step({})

if __name__ == '__main__':
  unittest.main()
