import unittest
from tests.testhelpers import ScriptedAgentTestConfig

from nmmo.core.env import Env as TaskEnv
from nmmo.lib.log import EventCode
from nmmo.task.predicate import predicate, OR
from nmmo.task.base_predicates import HoardGold, AllDead, StayAlive, \
  CountEvent, DistanceTraveled, AttainSkill, nmmo_skill
from nmmo.task.game_state import GameState
from nmmo.task.group import Group
from nmmo.task.utils import TeamHelper
from nmmo.task.task_api import PredicateTask, Once, Repeat

class TestDemoTask(unittest.TestCase):
  # pylint: disable=protected-access,invalid-name
  def test_example_user_task_definition(self):
    config = ScriptedAgentTestConfig()
    env = TaskEnv(config)
    team_helper = TeamHelper.generate_from_config(config)

    # Define Predicate Utilities
    @predicate
    def CustomPredicate(gs: GameState,
                        subject: Group,
                        target: Group):
        t1 = HoardGold(subject, 10) & AllDead(target)
        return t1

    # Define Task Utilities
    class CompletionChangeTask(PredicateTask):
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
    all_agents = team_helper.all_agents()
    team_A = team_helper.own_team(1)
    team_B = team_helper.left_team(1)
    custom_predicate = CustomPredicate(subject=team_A, target=team_B)

    stay_alive_tasks = [Repeat(agent, StayAlive(agent)) for agent in all_agents]
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
    env = TaskEnv(config)
    team_helper = TeamHelper.generate_from_config(config)

    # Task Definition
      # Utility
    class KillTask(PredicateTask):
      def __init__(self, assignee: Group):
        # Use predicate to easily access game state
        kill_predicate = CountEvent(assignee, 'PLAYER_KILL', 100)
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
    all_agents = team_helper.all_agents()
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
    def player_kills(team_helper):
      all_agents = team_helper.all_agents()
      tasks = []
      for agent in all_agents:
         agent_tasks = [
            Once(agent, CountEvent(agent, 'PLAYER_KILL', 1), reward=Tier.EASY),
            Once(agent, CountEvent(agent, 'PLAYER_KILL', 2), reward=Tier.NORMAL),
            Once(agent, CountEvent(agent, 'PLAYER_KILL', 3), reward=Tier.HARD)
         ]
         tasks = tasks + agent_tasks
      return tasks
    
    def exploration(team_helper):
      all_agents = team_helper.all_agents()
      tasks = []
      for agent in all_agents:
         agent_tasks = [
            Once(agent, DistanceTraveled(agent, 16), reward=Tier.EASY),
            Once(agent, DistanceTraveled(agent, 32), reward=Tier.NORMAL),
            Once(agent, DistanceTraveled(agent, 64), reward=Tier.HARD)
         ]
         tasks = tasks + agent_tasks
      return tasks

    # Demonstrates custom predicate - return float/boolean
    def equipment(team_helper):
      @predicate
      def EquipmentLevel(gs: GameState,
                        subject: Group,
                        number: int):
        equipped = (subject.item.equipped>0)
        levels = subject.item.level[equipped]
        return levels.sum() >= number

      all_agents = team_helper.all_agents()
      tasks = []
      for agent in all_agents:
         agent_tasks = [
            Once(agent, EquipmentLevel(agent, 1), reward=Tier.EASY),
            Once(agent, EquipmentLevel(agent, 5), reward=Tier.NORMAL),
            Once(agent, EquipmentLevel(agent, 10), reward=Tier.HARD)
         ]
         tasks = tasks + agent_tasks
      return tasks

    # Demonstrates custom predicate - return predicate
    def combat(team_helper):
      @predicate
      def CombatSkill(gs, agent, lvl):
         return OR(AttainSkill(agent, nmmo_skill.Melee, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Range, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Mage, lvl, 1))
      all_agents = team_helper.all_agents()
      tasks = []
      for agent in all_agents:
         agent_tasks = [
            Once(agent, CombatSkill(agent, 2), reward=Tier.EASY),
            Once(agent, CombatSkill(agent, 3), reward=Tier.NORMAL),
            Once(agent, CombatSkill(agent, 4), reward=Tier.HARD)
         ]
         tasks = tasks + agent_tasks
      return tasks

    def foraging(team_helper):
      @predicate
      def ForageSkill(gs, agent, lvl):
         return OR(AttainSkill(agent, nmmo_skill.Fishing, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Herbalism, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Prospecting, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Carving, lvl, 1),
                   AttainSkill(agent, nmmo_skill.Alchemy, lvl, 1))

      all_agents = team_helper.all_agents()
      tasks = []
      for agent in all_agents:
         agent_tasks = [
            Once(agent, ForageSkill(agent, 2), reward=Tier.EASY),
            Once(agent, ForageSkill(agent, 3), reward=Tier.NORMAL),
            Once(agent, ForageSkill(agent, 4), reward=Tier.HARD)
         ]
         tasks = tasks + agent_tasks
      return tasks

    # Demonstrate task scenario definition API
    def all_tasks(team_helper):
       return exploration(team_helper) + \
        equipment(team_helper) + \
        combat(team_helper) + \
        foraging(team_helper)

    # Test rollout
    scenarios = [player_kills, exploration, equipment, combat, foraging, all_tasks]
    for scenario in scenarios:
      config = ScriptedAgentTestConfig()
      env = TaskEnv(config)
      team_helper = TeamHelper.generate_from_config(config)
      tasks = scenario(team_helper)
      env.change_task(tasks)
      for _ in range(30):
        env.step({})

    # DONE

if __name__ == '__main__':
  unittest.main()
