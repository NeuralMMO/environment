import unittest

from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv

from nmmo.lib.event_log import EventState, EventCode, EventLogger
from nmmo.systems.item import Scrap, Ration
from nmmo.systems import skill as Skill


class TestEventLog(unittest.TestCase):

  def test_event_logging(self):
    config =  ScriptedAgentTestConfig()
    env = ScriptedAgentTestEnv(config)
    env.reset()

    # EventCode.SCORE_KILL: set level for agent 5 (target)
    env.realm.players[5].skills.range.level.update(5)

    # initialize Event datastore
    env.realm.datastore.register_object_type("Event", EventState.State.num_attributes)

    event_log = EventLogger(env.realm)

    """logging events to test/count"""

    # tick = 1
    env.step({})
    event_log.record(EventCode.EAT_FOOD, env.realm.players[1])
    event_log.record(EventCode.DRINK_WATER, env.realm.players[2])
    event_log.record(EventCode.SCORE_HIT, env.realm.players[2],
                     combat_style=Skill.Melee, damage=50)
    event_log.record(EventCode.SCORE_KILL, env.realm.players[3],
                     target=env.realm.players[5])

    # tick = 2
    env.step({})
    event_log.record(EventCode.CONSUME_ITEM, env.realm.players[4],
                     item=Ration(env.realm, 8))
    event_log.record(EventCode.GIVE_ITEM, env.realm.players[4])
    event_log.record(EventCode.DESTROY_ITEM, env.realm.players[5])
    event_log.record(EventCode.PRODUCE_ITEM, env.realm.players[6],
                     item=Scrap(env.realm, 3))

    # tick = 3
    env.step({})
    event_log.record(EventCode.GIVE_GOLD, env.realm.players[7])
    event_log.record(EventCode.LIST_ITEM, env.realm.players[8],
                     item=Ration(env.realm, 5), price=11)
    event_log.record(EventCode.EARN_GOLD, env.realm.players[9], amount=15)
    event_log.record(EventCode.BUY_ITEM, env.realm.players[10],
                     item=Scrap(env.realm, 7), price=21)
    event_log.record(EventCode.SPEND_GOLD, env.realm.players[11], amount=25)

    log_data = [list(row) for row in event_log.get_data()]

    self.assertListEqual(log_data, [
      [ 1,  1, 1, EventCode.EAT_FOOD, 0, 0, 0, 0, 0],
      [ 2,  2, 1, EventCode.DRINK_WATER, 0, 0, 0, 0, 0],
      [ 3,  2, 1, EventCode.SCORE_HIT, 1, 0, 50, 0, 0],
      [ 4,  3, 1, EventCode.SCORE_KILL, 0, 5, 0, 0, 5],
      [ 5,  4, 2, EventCode.CONSUME_ITEM, 16, 8, 1, 0, 0],
      [ 6,  4, 2, EventCode.GIVE_ITEM, 0, 0, 0, 0, 0],
      [ 7,  5, 2, EventCode.DESTROY_ITEM, 0, 0, 0, 0, 0],
      [ 8,  6, 2, EventCode.PRODUCE_ITEM, 13, 3, 1, 0, 0],
      [ 9,  7, 3, EventCode.GIVE_GOLD, 0, 0, 0, 0, 0],
      [10,  8, 3, EventCode.LIST_ITEM, 16, 5, 1, 11, 0],
      [11,  9, 3, EventCode.EARN_GOLD, 0, 0, 0, 15, 0],
      [12, 10, 3, EventCode.BUY_ITEM, 13, 7, 1, 21, 0],
      [13, 11, 3, EventCode.SPEND_GOLD, 0, 0, 0, 25, 0]])


if __name__ == '__main__':
  unittest.main()
