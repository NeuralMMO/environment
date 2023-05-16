import unittest

import nmmo
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.lib.event_log import EventState, EventLogger
from nmmo.lib.log import EventCode
from nmmo.entity.entity import Entity
from nmmo.systems.item import ItemState
from nmmo.systems.item import Scrap, Ration, Hat
from nmmo.systems import skill as Skill


class MockRealm:
  def __init__(self):
    self.config = nmmo.config.Default()
    self.datastore = NumpyDatastore()
    self.items = {}
    self.datastore.register_object_type("Event", EventState.State.num_attributes)
    self.datastore.register_object_type("Item", ItemState.State.num_attributes)
    self.tick = 0


class MockEntity(Entity):
  # pylint: disable=super-init-not-called
  def __init__(self, ent_id, **kwargs):
    self.id = ent_id
    self.level = kwargs.pop('attack_level', 0)

  @property
  def ent_id(self):
    return self.id

  @property
  def attack_level(self):
    return self.level


class TestEventLog(unittest.TestCase):

  def test_event_logging(self):
    mock_realm = MockRealm()
    event_log = EventLogger(mock_realm)

    mock_realm.tick = 0 # tick increase to 1 after all actions are processed
    event_log.record(EventCode.EAT_FOOD, MockEntity(1))
    event_log.record(EventCode.DRINK_WATER, MockEntity(2))
    event_log.record(EventCode.SCORE_HIT, MockEntity(2),
                     combat_style=Skill.Melee, damage=50)
    event_log.record(EventCode.PLAYER_KILL, MockEntity(3),
                     target=MockEntity(5, attack_level=5))

    mock_realm.tick = 1
    event_log.record(EventCode.CONSUME_ITEM, MockEntity(4),
                     item=Ration(mock_realm, 8))
    event_log.record(EventCode.GIVE_ITEM, MockEntity(4))
    event_log.record(EventCode.DESTROY_ITEM, MockEntity(5))
    event_log.record(EventCode.HARVEST_ITEM, MockEntity(6),
                     item=Scrap(mock_realm, 3))

    mock_realm.tick = 2
    event_log.record(EventCode.GIVE_GOLD, MockEntity(7))
    event_log.record(EventCode.LIST_ITEM, MockEntity(8),
                     item=Ration(mock_realm, 5), price=11)
    event_log.record(EventCode.EARN_GOLD, MockEntity(9), amount=15)
    event_log.record(EventCode.BUY_ITEM, MockEntity(10),
                     item=Scrap(mock_realm, 7), price=21)
    #event_log.record(EventCode.SPEND_GOLD, env.realm.players[11], amount=25)

    mock_realm.tick = 3
    event_log.record(EventCode.LEVEL_UP, MockEntity(12),
                     skill=Skill.Fishing, level=3)

    mock_realm.tick = 4
    event_log.record(EventCode.GO_FARTHEST, MockEntity(12), distance=6)
    event_log.record(EventCode.EQUIP_ITEM, MockEntity(12),
                     item=Hat(mock_realm, 4))

    log_data = [list(row) for row in event_log.get_data()]

    self.assertListEqual(log_data, [
      [ 1,  1, 1, EventCode.EAT_FOOD, 0, 0, 0, 0, 0],
      [ 2,  2, 1, EventCode.DRINK_WATER, 0, 0, 0, 0, 0],
      [ 3,  2, 1, EventCode.SCORE_HIT, 1, 0, 50, 0, 0],
      [ 4,  3, 1, EventCode.PLAYER_KILL, 0, 5, 0, 0, 5],
      [ 5,  4, 2, EventCode.CONSUME_ITEM, 16, 8, 1, 0, 0],
      [ 6,  4, 2, EventCode.GIVE_ITEM, 0, 0, 0, 0, 0],
      [ 7,  5, 2, EventCode.DESTROY_ITEM, 0, 0, 0, 0, 0],
      [ 8,  6, 2, EventCode.HARVEST_ITEM, 13, 3, 1, 0, 0],
      [ 9,  7, 3, EventCode.GIVE_GOLD, 0, 0, 0, 0, 0],
      [10,  8, 3, EventCode.LIST_ITEM, 16, 5, 1, 11, 0],
      [11,  9, 3, EventCode.EARN_GOLD, 0, 0, 0, 15, 0],
      [12, 10, 3, EventCode.BUY_ITEM, 13, 7, 1, 21, 0],
      [13, 12, 4, EventCode.LEVEL_UP, 4, 3, 0, 0, 0],
      [14, 12, 5, EventCode.GO_FARTHEST, 0, 0, 6, 0, 0],
      [15, 12, 5, EventCode.EQUIP_ITEM, 2, 4, 1, 0, 0]])

if __name__ == '__main__':
  unittest.main()

  """
  TEST_HORIZON = 50
  RANDOM_SEED = 338

  from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv

  config = ScriptedAgentTestConfig()
  env = ScriptedAgentTestEnv(config)

  env.reset(seed=RANDOM_SEED)

  from tqdm import tqdm
  for tick in tqdm(range(TEST_HORIZON)):
    env.step({})

    # events to check
    log = env.realm.event_log.get_data()    
    idx = (log[:,2] == tick+1) & (log[:,3] == EventCode.EQUIP_ITEM)
    if sum(idx):
      print(log[idx])
      print()

  print('done')
  """
