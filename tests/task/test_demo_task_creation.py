import unittest
from tests.testhelpers import ScriptedAgentTestConfig

from nmmo.core.env import Env
from nmmo.lib.log import EventCode
from nmmo.systems import skill
from nmmo.task.predicate import predicate, OR
from nmmo.task import base_predicates as p
from nmmo.task import task_api as t
from nmmo.task.game_state import GameState
from nmmo.task.group import Group
from nmmo.task.team_helper import TeamHelper
from nmmo.task.scenario import Scenario

class TestDemoTask(unittest.TestCase):
  # pylint: disable=protected-access,invalid-name
  def test_example_user_task_definition(self):
    config = ScriptedAgentTestConfig()
    env = Env(config)
    team_helper = TeamHelper.generate_from_config(config)

    # Define Predicate Utilities
    @predicate
    def CustomPredicate(gs: GameState,
                        subject: Group,
                        target: Group):
        t1 = p.HoardGold(subject, 10) & p.AllDead(target)
        return t1

    # Define Task Utilities
    class CompletionChangeTask(t.PredicateTask):
       def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self._previous = 0

       def _rewards(self, gs: GameState):
          infos = {int(ent_id): {self._predicate.name: self.evaluate(gs)}
             for ent_id in self._assignee}
          change = self.evaluate(gs) - self._previous
          self._previous = self.evaluate(gs)
          rewards = {int(ent_id): change for ent_id in self._assignee}
          return rewards, infos

    # Creation of the actual tasks
    all_agents = team_helper.all_agents
    team_A = team_helper.own_team(1)
    team_B = team_helper.left_team(1)
    custom_predicate = CustomPredicate(subject=team_A, target=team_B)

    stay_alive_tasks = [t.Repeat(agent, p.StayAlive(agent)) for agent in all_agents]
    custom_task_2 = CompletionChangeTask(team_A, predicate=custom_predicate)

    tasks = [(custom_task_2, 5)] + stay_alive_tasks

    # Test rollout
    env.change_task(tasks)
    for _ in range(30):
       env.step({})

    # DONE

  def test_player_kill_reward(self):
    ''' Reward 0.1 per player defeated, 1 for first and 3rd kills
    Up to 100 kills
    '''
    # Setup 
    config = ScriptedAgentTestConfig()
    env = Env(config)
    team_helper = TeamHelper.generate_from_config(config)

    # Task Definition
      # Utility
    class KillTask(t.PredicateTask):
      def __init__(self, assignee: Group):
        # Use predicate to easily access game state
        kill_predicate = p.CountEvent(assignee, 'PLAYER_KILL', 100)
        super().__init__(assignee, kill_predicate)
        self._previous = 0
      def _rewards(self, gs: GameState):
          # Calculate data
          current_kills = int(self.evaluate(gs) * 100)
          previous_kills = self._previous
          change = current_kills - previous_kills
          self._previous = current_kills
          # Reward a bonus for 1st and 3rd kills
          bonus = 0
          if previous_kills < 1 <= current_kills:
             bonus += 0.9
          if previous_kills < 3 <= current_kills:
             bonus += 0.9
          rewards = {int(ent_id): bonus + change * 0.1
                     for ent_id in self._assignee}
          return rewards, {}
      # Assigning
    tasks = []
    all_agents = team_helper.all_agents
    for agent in all_agents:
      tasks.append(KillTask(agent))

    # Test Reward
    env.change_task(tasks)
    players = env.realm.players
    code = EventCode.PLAYER_KILL
    env.realm.event_log.record(code, players[1], target=players[3])
    env.realm.event_log.record(code, players[2], target=players[4])
    env.realm.event_log.record(code, players[2], target=players[5])
    env.realm.event_log.record(EventCode.EAT_FOOD, players[2])
      # Award given as designed
      # Agent 1 kills 1 - reward 1
      # Agent 2 kills 2 - reward 1 + 0.1
      # Agent 3 kills 0 - reward 0
    _, rewards, _, _ = env.step({})
    self.assertEqual(rewards[1],1)
    self.assertEqual(rewards[2],1.1)
    self.assertEqual(rewards[3],0)
      # No reward when no changes
    _, rewards, _, _ = env.step({})
    self.assertEqual(rewards[1],0)
    self.assertEqual(rewards[2],0)
    self.assertEqual(rewards[3],0)
      # Test task reset on env reset
    env.reset() 
    _, rewards, _, _ = env.step({})
    self.assertEqual(env.tasks[0][0]._previous,0)

    # Test Rollout
    env.change_task(tasks)
    for _ in range(10):
       env.step({})
  
    # DONE

  def test_baseline_tasks(self):
    # Tasks from
    # https://github.com/NeuralMMO/baselines/
    # blob/4c1088d2bbe0f74a08dcf7d71b714cd30772557f/tasks.py
    class Tier:
      REWARD_SCALE = 15
      EASY         = 4 / REWARD_SCALE
      NORMAL       = 6 / REWARD_SCALE
      HARD         = 11 / REWARD_SCALE

    # Usage of inbuilt predicate
    def player_kills(scenario: Scenario):
      scenario.add_tasks(p.CountEvent(event='PLAYER_KILL',N=1),reward_options={'reward':Tier.EASY})
      scenario.add_tasks(p.CountEvent(event='PLAYER_KILL',N=2),reward_options={'reward':Tier.NORMAL})
      scenario.add_tasks(p.CountEvent(event='PLAYER_KILL',N=3),reward_options={'reward':Tier.HARD})
      return scenario.tasks
    
    def exploration(scenario: Scenario):
      scenario.add_tasks(p.DistanceTraveled(dist=16),reward_options={'reward':Tier.EASY})
      scenario.add_tasks(p.DistanceTraveled(dist=32),reward_options={'reward':Tier.NORMAL})
      scenario.add_tasks(p.DistanceTraveled(dist=64),reward_options={'reward':Tier.HARD})
      return scenario.tasks

    # Demonstrates custom predicate - return float/boolean
    @predicate
    def EquipmentLevel(gs: GameState,
                       subject: Group,
                       number: int):
      equipped = (subject.item.equipped>0)
      levels = subject.item.level[equipped]
      return levels.sum() >= number

    def equipment(scenario: Scenario):
      scenario.add_tasks(EquipmentLevel(number=1 ), groups='agents', reward_options={'reward':Tier.EASY})
      scenario.add_tasks(EquipmentLevel(number=5 ), groups='agents', reward_options={'reward':Tier.NORMAL})
      scenario.add_tasks(EquipmentLevel(number=10), groups='agents', reward_options={'reward':Tier.HARD})
      return scenario.tasks
    
    @predicate
    def CombatSkill(gs, subject, lvl):
        return OR(p.AttainSkill(subject, skill.Melee, lvl, 1),
                  p.AttainSkill(subject, skill.Range, lvl, 1),
                  p.AttainSkill(subject, skill.Mage, lvl, 1))

    def combat(scenario: Scenario):
      scenario.add_tasks(CombatSkill(lvl=2), groups='agents', reward_options={'reward':Tier.EASY})
      scenario.add_tasks(CombatSkill(lvl=3), groups='agents', reward_options={'reward':Tier.NORMAL})
      scenario.add_tasks(CombatSkill(lvl=4), groups='agents', reward_options={'reward':Tier.HARD})
      return scenario.tasks

    @predicate
    def ForageSkill(gs, subject, lvl):
        return OR(p.AttainSkill(subject, skill.Fishing, lvl, 1),
                  p.AttainSkill(subject, skill.Herbalism, lvl, 1),
                  p.AttainSkill(subject, skill.Prospecting, lvl, 1),
                  p.AttainSkill(subject, skill.Carving, lvl, 1),
                  p.AttainSkill(subject, skill.Alchemy, lvl, 1))
  
    def foraging(scenario: Scenario):
      scenario.add_tasks(ForageSkill(lvl=2),reward_options={'reward':Tier.EASY})
      scenario.add_tasks(ForageSkill(lvl=3),reward_options={'reward':Tier.NORMAL})
      scenario.add_tasks(ForageSkill(lvl=4),reward_options={'reward':Tier.HARD})
      return scenario.tasks

    # Demonstrate task scenario definition API
    def all_tasks(scenario: Scenario):
      player_kills(scenario)
      exploration(scenario)
      equipment(scenario)
      combat(scenario)
      foraging(scenario)
      return scenario.tasks

    # Test rollout
    task_generators = [player_kills, exploration, equipment, combat, foraging, all_tasks]
    for tg in task_generators:
      config = ScriptedAgentTestConfig()
      env = Env(config)
      scenario = Scenario(config)
      tasks = tg(scenario)
      env.change_task(tasks)
      for _ in range(30):
        env.step({})

    # DONE

if __name__ == '__main__':
  unittest.main()
