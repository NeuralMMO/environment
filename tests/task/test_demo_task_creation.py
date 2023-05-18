import unittest
from tests.testhelpers import ScriptedAgentTestConfig

from nmmo.core.env import Env
from nmmo.lib.log import EventCode
from nmmo.systems import skill
from nmmo.task import base_predicates as p
from nmmo.task import task_api as t
from nmmo.task.game_state import GameState
from nmmo.task.group import Group
from nmmo.task.scenario import Scenario

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

    # Usage of inbuilt predicate
    def player_kills(scenario: Scenario):
      scenario.add_tasks(p.CountEvent(event='PLAYER_KILL',N=1)*Tier.EASY)
      scenario.add_tasks(p.CountEvent(event='PLAYER_KILL',N=2)*Tier.NORMAL)
      scenario.add_tasks(p.CountEvent(event='PLAYER_KILL',N=3)*Tier.HARD)
      return scenario.tasks

    def exploration(scenario: Scenario):
      scenario.add_tasks(p.DistanceTraveled(dist=16)*Tier.EASY)
      scenario.add_tasks(p.DistanceTraveled(dist=32)*Tier.NORMAL)
      scenario.add_tasks(p.DistanceTraveled(dist=64)*Tier.HARD)
      return scenario.tasks

    # Demonstrates custom predicate - return float/boolean
    @t.define_predicate
    def EquipmentLevel(gs: GameState,
                       subject: Group,
                       number: int):
      equipped = (subject.item.equipped>0)
      levels = subject.item.level[equipped]
      return levels.sum() >= number

    def equipment(scenario: Scenario):
      scenario.add_tasks(EquipmentLevel(number=1 )*Tier.EASY, groups='agents')
      scenario.add_tasks(EquipmentLevel(number=5 )*Tier.NORMAL, groups='agents')
      scenario.add_tasks(EquipmentLevel(number=10)*Tier.HARD, groups='agents')
      return scenario.tasks

    @t.define_predicate
    def CombatSkill(gs, subject, lvl):
        return t.OR(p.AttainSkill(subject, skill.Melee, lvl, 1),
                  p.AttainSkill(subject, skill.Range, lvl, 1),
                  p.AttainSkill(subject, skill.Mage, lvl, 1))

    def combat(scenario: Scenario):
      scenario.add_tasks(CombatSkill(lvl=2)*Tier.EASY, groups='agents')
      scenario.add_tasks(CombatSkill(lvl=3)*Tier.NORMAL, groups='agents')
      scenario.add_tasks(CombatSkill(lvl=4)*Tier.HARD, groups='agents')
      return scenario.tasks

    @t.define_predicate
    def ForageSkill(gs, subject, lvl):
        return t.OR(p.AttainSkill(subject, skill.Fishing, lvl, 1),
                  p.AttainSkill(subject, skill.Herbalism, lvl, 1),
                  p.AttainSkill(subject, skill.Prospecting, lvl, 1),
                  p.AttainSkill(subject, skill.Carving, lvl, 1),
                  p.AttainSkill(subject, skill.Alchemy, lvl, 1))

    def foraging(scenario: Scenario):
      scenario.add_tasks(ForageSkill(lvl=2)*Tier.EASY)
      scenario.add_tasks(ForageSkill(lvl=3)*Tier.NORMAL)
      scenario.add_tasks(ForageSkill(lvl=4)*Tier.HARD)
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
      for _ in range(10):
        env.step({})

    # DONE

  def test_player_kill_reward(self):
    """ Reward 0.1 per player defeated, 1 for first and 3rd kills
    """
    config = ScriptedAgentTestConfig()
    env = Env(config)
    scenario = Scenario(config)

    # PARTICIPANT WRITES
    # ====================================
    @t.define_task
    def KillTask(gs: GameState,
                 subject: Group):
      """ Reward 0.1 per player defeated, with a bonus for the 1st and 3rd kills.
      """
      num_kills = len(subject.event.PLAYER_KILL)
      score = num_kills * 0.1
      if num_kills >= 1:
        score += 1
      if num_kills >= 3:
        score += 1
      return score

    scenario.add_tasks(lambda agent: KillTask(agent), groups='agents')
    # ====================================

    # Test Reward
    env.change_task(scenario.tasks)
    players = env.realm.players
    code = EventCode.PLAYER_KILL
    env.realm.event_log.record(code, players[1], target=players[3])
    env.realm.event_log.record(code, players[2], target=players[4])
    env.realm.event_log.record(code, players[2], target=players[5])
    env.realm.event_log.record(EventCode.EAT_FOOD, players[2])
      # Award given as designed
      # Agent 1 kills 1 - reward 1 + 0.1
      # Agent 2 kills 2 - reward 1 + 0.2
      # Agent 3 kills 0 - reward 0
    _, rewards, _, _ = env.step({})
    self.assertEqual(rewards[1],1.1)
    self.assertEqual(rewards[2],1.2)
    self.assertEqual(rewards[3],0)
      # No reward when no changes
    _, rewards, _, _ = env.step({})
    self.assertEqual(rewards[1],0)
    self.assertEqual(rewards[2],0)
    self.assertEqual(rewards[3],0)
      # Test task reset on env reset
    env.reset() 
    _, rewards, _, _ = env.step({})
    self.assertEqual(env.tasks[0][0]._score,0)

    # Test Rollout
    env.change_task(scenario.tasks)
    for _ in range(10):
       env.step({})

    # DONE

  def test_combination_task_reward(self):
    config = ScriptedAgentTestConfig()
    env = Env(config)
    scenario = Scenario(config)

    task = t.OR(p.CountEvent(event='PLAYER_KILL',N=5),p.TickGE(num_tick=5))
    task = task * 5
    scenario.add_tasks(task)

    # Test Reward
    env.change_task(scenario.tasks)
    code = EventCode.PLAYER_KILL
    players = env.realm.players
    env.realm.event_log.record(code, players[1], target=players[2])
    env.realm.event_log.record(code, players[1], target=players[3])

    _, rewards, _, _ = env.step({})
    self.assertEqual(rewards[1],2)

    for _ in range(4):
      _, _, _, infos = env.step({})
    
    self.assertEqual(list(infos[1]['task'].values())[0],5.0)

    # DONE

if __name__ == '__main__':
  unittest.main()
