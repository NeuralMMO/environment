import unittest
from typing import List, Tuple
import random

from tests.testhelpers import ScriptedAgentTestConfig, provide_item
from tests.testhelpers import change_spawn_pos as change_agent_pos

from scripted.baselines import Sleeper

from nmmo.entity.entity import EntityState
from nmmo.systems import item as Item
from nmmo.systems import skill as Skill
from nmmo.lib import material as Material
from nmmo.lib.log import EventCode

# pylint: disable=import-error
from nmmo.core.env import Env as TaskEnv
from nmmo.task.task_api import Repeat, MultiTask
from nmmo.task.predicate import Predicate
from nmmo.task.group import Group
import nmmo.task.predicate.base_predicate as bp
import nmmo.task.predicate.item_predicate as ip
import nmmo.task.predicate.gold_predicate as gp

# use the constant reward of 1 for testing predicates
REWARD = 1
NUM_AGENT = 6
ALL_AGENT = Group(list(range(1, NUM_AGENT+1)), 'All')

class TestBasePredicate(unittest.TestCase):
  # pylint: disable=protected-access,invalid-name

  def _get_taskenv(self,
                   test_tasks: List[Tuple[Predicate, int]],
                   grass_map=False):

    config = ScriptedAgentTestConfig()
    config.PLAYERS = [Sleeper]
    config.PLAYER_N = NUM_AGENT
    config.IMMORTAL = True

    tasks = MultiTask(
      *(Repeat(team, tsk, REWARD) for tsk, team in test_tasks)
    )

    env = TaskEnv(config)
    env.change_task(tasks)

    if grass_map:
      MS = env.config.MAP_SIZE
      # Change entire map to grass to become habitable
      for i in range(MS):
        for j in range(MS):
          tile = env.realm.map.tiles[i,j]
          tile.material = Material.Grass
          tile.material_id.update(Material.Grass.index)
          tile.state = Material.Grass(env.config)

    return env

  def _check_result(self, env, test_tasks, infos, true_task):
    for tid, (task, assignee) in enumerate(test_tasks):
      # result is cached when at least one assignee is alive so that the task is evaled
      if set(assignee).intersection(infos):
        self.assertEqual(env.game_state.cache_result[task.name], tid in true_task)
      for ent_id in infos:
        if ent_id in assignee:
          # the agents that are assigned the task get evaluated for reward
          self.assertEqual(int(infos[ent_id]['task'][task.name]), int(tid in true_task))
        else:
          # the agents that are not assigned the task are not evaluated
          self.assertTrue(task.name not in infos[ent_id]['task'])

  def _check_progress(self, task, infos, value):
    """ Some predicates return a float in the range 0-1 indicating completion progress.
    """
    predicate, assignee = task[0], task[1]
    for ent_id in infos:
      if ent_id in assignee:
        self.assertAlmostEqual(infos[ent_id]['task'][predicate.name],value)

  def test_tickge_stay_alive_rip(self):
    tick_true = 5
    death_note = [1, 2, 3]
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (bp.TickGE(tick_true), ALL_AGENT),
      (bp.StayAlive(Group([1, 3])), ALL_AGENT),
      (bp.StayAlive(Group([3, 4])), Group([1, 2])),
      (bp.StayAlive(Group([4])), Group([5, 6])),
      (bp.AllDead(Group([1, 3])), ALL_AGENT),
      (bp.AllDead(Group([3, 4])), Group([1, 2])),
      (bp.AllDead(Group([4])), Group([5, 6]))]

    env = self._get_taskenv(test_tasks)

    for _ in range(tick_true-1):
      _, _, _, infos = env.step({})

    # TickGE_5 is false. All agents are alive,
    # so all StayAlive (ti in [1,2,3]) tasks are true
    # and all AllDead tasks (ti in [4, 5, 6]) are false

    true_task = [1, 2, 3]
    self._check_result(env, test_tasks, infos, true_task)
    self._check_progress(test_tasks[0], infos, (tick_true-1) / tick_true)

    # kill agents 1-3
    for ent_id in death_note:
      env.realm.players[ent_id].resources.health.update(0)
    env.obs = env._compute_observations()

    # 6th tick
    _, _, _, infos = env.step({})

    # those who have survived
    entities = EntityState.Query.table(env.realm.datastore)
    entities = list(entities[:, EntityState.State.attr_name_to_col['id']]) # ent_ids

    # make sure the dead agents are not in the realm & datastore
    for ent_id in env.realm.players.spawned:
      if ent_id in death_note:
        # make sure that dead players not in the realm nor the datastore
        self.assertTrue(ent_id not in env.realm.players)
        self.assertTrue(ent_id not in entities)
        # CHECK ME: dead agents are also not in infos
        self.assertTrue(ent_id not in infos)

    # TickGE_5 is true. Agents 1-3 are dead, so
    # StayAlive(1,3) and StayAlive(3,4) are false, StayAlive(4) is true
    # AllDead(1,3) is true, AllDead(3,4) and AllDead(4) are false
    true_task = [0, 3, 4]
    self._check_result(env, test_tasks, infos, true_task)

    # 3 is dead but 4 is alive. Half of agents killed, 50% completion.
    self._check_progress(test_tasks[5], infos, 0.5)

    # DONE

  def test_can_see_tile(self):
    a1_target = Material.Forest
    a2_target = Material.Water
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (bp.CanSeeTile(Group([1]), a1_target), ALL_AGENT), # True
      (bp.CanSeeTile(Group([1,3,5]), a2_target), ALL_AGENT), # False
      (bp.CanSeeTile(Group([2]), a2_target), Group([1,2,3])), # True
      (bp.CanSeeTile(Group([2,5,6]), a1_target), ALL_AGENT), # False
      (bp.CanSeeTile(ALL_AGENT, a2_target), Group([2,3,4]))] # True

    # setup env with all grass map
    env = self._get_taskenv(test_tasks, grass_map=True)

    # Two corners to the target materials
    MS = env.config.MAP_SIZE
    tile = env.realm.map.tiles[0,MS-2]
    tile.material = Material.Forest
    tile.material_id.update(Material.Forest.index)

    tile = env.realm.map.tiles[MS-1,0]
    tile.material = Material.Water
    tile.material_id.update(Material.Water.index)

    # All agents to one corner
    for ent_id in env.realm.players:
      change_agent_pos(env.realm,ent_id,(0,0))

    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    # no target tiles are found, so all are false
    true_task = []
    self._check_result(env, test_tasks, infos, true_task)

    # Team one to forest, team two to water
    change_agent_pos(env.realm,1,(0,MS-2)) # agent 1, team 0, forest
    change_agent_pos(env.realm,2,(MS-2,0)) # agent 2, team 1, water
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # t0, t2, t4 are true
    true_task = [0, 2, 4]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_can_see_agent(self):
    search_target = 1
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (bp.CanSeeAgent(Group([1]), search_target), ALL_AGENT), # Always True
      (bp.CanSeeAgent(Group([2]), search_target), Group([2,3,4])), # False -> True -> True
      (bp.CanSeeAgent(Group([3,4,5]), search_target), Group([1,2,3])), # False -> False -> True
      (bp.CanSeeGroup(Group([1]), Group([3,4])), ALL_AGENT)] # False -> False -> True

    env = self._get_taskenv(test_tasks, grass_map=True)

    # All agents to one corner
    for ent_id in env.realm.players:
      change_agent_pos(env.realm,ent_id,(0,0))

    # Teleport agent 1 to the opposite corner
    MS = env.config.MAP_SIZE
    change_agent_pos(env.realm,1,(MS-2,MS-2))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # Only CanSeeAgent(Group([1]), search_target) is true, others are false
    true_task = [0]
    self._check_result(env, test_tasks, infos, true_task)

    # Teleport agent 2 to agent 1's pos
    change_agent_pos(env.realm,2,(MS-2,MS-2))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # SearchAgent(Team([2]), search_target) is also true
    true_task = [0,1]
    self._check_result(env, test_tasks, infos, true_task)

    # Teleport agent 3 to agent 1s position
    change_agent_pos(env.realm,3,(MS-2,MS-2))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})
    true_task = [0,1,2,3]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_occupy_tile(self):
    target_tile = (30, 30)
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (bp.OccupyTile(Group([1]), *target_tile), ALL_AGENT), # False -> True
      (bp.OccupyTile(Group([1,2,3]), *target_tile), Group([4,5,6])), # False -> True
      (bp.OccupyTile(Group([2]), *target_tile), Group([2,3,4])), # False
      (bp.OccupyTile(Group([3,4,5]), *target_tile), Group([1,2,3]))] # False

    # make all tiles habitable
    env = self._get_taskenv(test_tasks, grass_map=True)

    # All agents to one corner
    for ent_id in env.realm.players:
      change_agent_pos(env.realm,ent_id,(0,0))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # all tasks must be false
    true_task = []
    self._check_result(env, test_tasks, infos, true_task)

    # teleport agent 1 to the target tile, agent 2 to the adjacent tile
    change_agent_pos(env.realm,1,target_tile)
    change_agent_pos(env.realm,2,(target_tile[0],target_tile[1]-1))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # tid 0 and 1 should be true: OccupyTile(Group([1]), *target_tile)
    #  & OccupyTile(Group([1,2,3]), *target_tile)
    true_task = [0, 1]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_distance_traveled(self):
    agent_dist = 6
    team_dist = 10
    # NOTE: when evaluating predicates, to whom tasks are assigned are irrelevant
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (bp.DistanceTraveled(Group([1]), agent_dist), ALL_AGENT), # False -> True
      (bp.DistanceTraveled(Group([2, 5]), agent_dist), ALL_AGENT), # False
      (bp.DistanceTraveled(Group([3, 4]), agent_dist), ALL_AGENT), # False
      (bp.DistanceTraveled(Group([1, 2, 3]), team_dist), ALL_AGENT), # False -> True
      (bp.DistanceTraveled(Group([6]), agent_dist), ALL_AGENT)] # False

    # make all tiles habitable
    env = self._get_taskenv(test_tasks, grass_map=True)

    _, _, _, infos = env.step({})

    # one cannot accomplish these goals in the first tick, so all false
    true_task = []
    self._check_result(env, test_tasks, infos, true_task)

    # all are sleeper, so they all stay in the spawn pos
    spawn_pos = { ent_id: ent.pos for ent_id, ent in env.realm.players.items() }
    ent_id = 1 # move 6 tiles, to reach the goal
    change_agent_pos(env.realm, ent_id, (spawn_pos[ent_id][0]+6, spawn_pos[ent_id][1]))
    ent_id = 2 # move 2, fail to reach agent_dist, but reach team_dist if add all
    change_agent_pos(env.realm, ent_id, (spawn_pos[ent_id][0]+2, spawn_pos[ent_id][1]))
    ent_id = 3 # move 3, fail to reach agent_dist, but reach team_dist if add all
    change_agent_pos(env.realm, ent_id, (spawn_pos[ent_id][0], spawn_pos[ent_id][1]+3))
    env.obs = env._compute_observations()

    _,_,_, infos = env.step({})

    true_task = [0, 3]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_all_members_within_range(self):
    dist_123 = 1
    dist_135 = 5
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (bp.AllMembersWithinRange(Group([1]), dist_123), ALL_AGENT), # Always true for group of 1
      (bp.AllMembersWithinRange(Group([1,2]), dist_123), ALL_AGENT), # True
      (bp.AllMembersWithinRange(Group([1,3]), dist_123), ALL_AGENT), # True
      (bp.AllMembersWithinRange(Group([2,3]), dist_123), ALL_AGENT), # False
      (bp.AllMembersWithinRange(Group([1,3,5]), dist_123), ALL_AGENT), # False
      (bp.AllMembersWithinRange(Group([1,3,5]), dist_135), ALL_AGENT), # True
      (bp.AllMembersWithinRange(Group([2,4,6]), dist_135), ALL_AGENT)] # False

    # make all tiles habitable
    env = self._get_taskenv(test_tasks, grass_map=True)

    MS = env.config.MAP_SIZE

    # team 0: staying within goal_dist
    change_agent_pos(env.realm, 1, (MS//2, MS//2))
    change_agent_pos(env.realm, 3, (MS//2-1, MS//2)) # also StayCloseTo a1 = True
    change_agent_pos(env.realm, 5, (MS//2-5, MS//2))

    # team 1: staying goal_dist+1 apart
    change_agent_pos(env.realm, 2, (MS//2+1, MS//2)) # also StayCloseTo a1 = True
    change_agent_pos(env.realm, 4, (MS//2+5, MS//2))
    change_agent_pos(env.realm, 6, (MS//2+8, MS//2))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [0, 1, 2, 5]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_attain_skill(self):
    goal_level = 5
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (bp.AttainSkill(Group([1]), Skill.Melee, goal_level, 1), ALL_AGENT), # False
      (bp.AttainSkill(Group([2]), Skill.Melee, goal_level, 1), ALL_AGENT), # False
      (bp.AttainSkill(Group([1]), Skill.Range, goal_level, 1), ALL_AGENT), # True
      (bp.AttainSkill(Group([1,3]), Skill.Fishing, goal_level, 1), ALL_AGENT), # True
      (bp.AttainSkill(Group([1,2,3]), Skill.Carving, goal_level, 3), ALL_AGENT), # False
      (bp.AttainSkill(Group([2,4]), Skill.Carving, goal_level, 2), ALL_AGENT)] # True

    env = self._get_taskenv(test_tasks)

    # AttainSkill(Group([1]), Skill.Melee, goal_level, 1) is false
    # AttainSkill(Group([2]), Skill.Melee, goal_level, 1) is false
    env.realm.players[1].skills.melee.level.update(goal_level-1)
    # AttainSkill(Group([1]), Skill.Range, goal_level, 1) is true
    env.realm.players[1].skills.range.level.update(goal_level)
    # AttainSkill(Group([1,3]), Skill.Fishing, goal_level, 1) is true
    env.realm.players[1].skills.fishing.level.update(goal_level)
    # AttainSkill(Group([1,2,3]), Skill.Carving, goal_level, 3) is false
    env.realm.players[1].skills.carving.level.update(goal_level)
    env.realm.players[2].skills.carving.level.update(goal_level)
    # AttainSkill(Group([2,4]), Skill.Carving, goal_level, 2) is true
    env.realm.players[4].skills.carving.level.update(goal_level+2)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [2, 3, 5]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_inventory_space_ge_not(self):
    # also test NOT InventorySpaceGE
    target_space = 3
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (ip.InventorySpaceGE(Group([1]), target_space), ALL_AGENT), # True -> False
      (ip.InventorySpaceGE(Group([2,3]), target_space), ALL_AGENT), # True
      (ip.InventorySpaceGE(Group([1,2,3]), target_space), ALL_AGENT), # True -> False
      (ip.InventorySpaceGE(Group([1,2,3,4]), target_space+1), ALL_AGENT), # False
      (~ip.InventorySpaceGE(Group([1]), target_space+1), ALL_AGENT), # True
      (~ip.InventorySpaceGE(Group([1,2,3]), target_space), ALL_AGENT), # False -> True
      (~ip.InventorySpaceGE(Group([1,2,3,4]), target_space+1), ALL_AGENT)] # True

    env = self._get_taskenv(test_tasks)

    # add one items to agent 1 within the limit
    capacity = env.realm.players[1].inventory.capacity
    provide_item(env.realm, 1, Item.Ration, level=1, quantity=capacity-target_space)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    self.assertTrue(env.realm.players[1].inventory.space >= target_space)
    true_task = [0, 1, 2, 4, 6]
    self._check_result(env, test_tasks, infos, true_task)

    # add one more item to agent 1
    provide_item(env.realm, 1, Item.Ration, level=1, quantity=1)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    self.assertTrue(env.realm.players[1].inventory.space < target_space)
    true_task = [1, 4, 5, 6]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_own_equip_item(self):
    # ration, level 2, quantity 3 (non-stackable)
    # ammo level 2, quantity 3 (stackable, equipable)
    goal_level = 2
    goal_quantity = 3
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (ip.OwnItem(Group([1]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # False
      (ip.OwnItem(Group([2]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # False
      (ip.OwnItem(Group([1,2]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # True
      (ip.OwnItem(Group([3]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # True
      (ip.OwnItem(Group([4,5,6]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # False
      (ip.EquipItem(Group([4]), Item.Scrap, goal_level, 1), ALL_AGENT), # False
      (ip.EquipItem(Group([4,5]), Item.Scrap, goal_level, 1), ALL_AGENT), # True
      (ip.EquipItem(Group([4,5,6]), Item.Scrap, goal_level, 2), ALL_AGENT)] # True

    env = self._get_taskenv(test_tasks)

    # set the level, so that agents 4-6 can equip the scrap
    equip_scrap = [4, 5, 6]
    for ent_id in equip_scrap:
      env.realm.players[ent_id].skills.melee.level.update(6) # melee skill level=6

    # provide items
    ent_id = 1 # OwnItem(Group([1]), Item.Ration, goal_level, goal_quantity) is false
    provide_item(env.realm, ent_id, Item.Ration, level=1, quantity=4)
    provide_item(env.realm, ent_id, Item.Ration, level=2, quantity=2)
    # OwnItem(Group([2]), Item.Ration, goal_level, goal_quantity) is false
    ent_id = 2 # OwnItem(Group([1,2]), Item.Ration, goal_level, goal_quantity) is true
    provide_item(env.realm, ent_id, Item.Ration, level=4, quantity=1)
    ent_id = 3 # OwnItem(Group([3]), Item.Ration, goal_level, goal_quantity) is true
    provide_item(env.realm, ent_id, Item.Ration, level=3, quantity=3)
    # OwnItem(Group([4,5,6]), Item.Ration, goal_level, goal_quantity) is false

    # provide and equip items
    ent_id = 4 # EquipItem(Group([4]), Item.Scrap, goal_level, 1) is false
    provide_item(env.realm, ent_id, Item.Scrap, level=1, quantity=4)
    ent_id = 5 # EquipItem(Group([4,5]), Item.Scrap, goal_level, 1) is true
    provide_item(env.realm, ent_id, Item.Scrap, level=4, quantity=1)
    ent_id = 6 # EquipItem(Group([4,5,6]), Item.Scrap, goal_level, 2) is true
    provide_item(env.realm, ent_id, Item.Scrap, level=2, quantity=4)
    for ent_id in [4, 5, 6]:
      scrap = env.realm.players[ent_id].inventory.items[0]
      scrap.use(env.realm.players[ent_id])
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [2, 3, 6, 7]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_fully_armed(self):
    goal_level = 5
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (ip.FullyArmed(Group([1,2,3]), Skill.Range, goal_level, 1), ALL_AGENT), # False
      (ip.FullyArmed(Group([3,4]), Skill.Range, goal_level, 1), ALL_AGENT), # True
      (ip.FullyArmed(Group([4]), Skill.Melee, goal_level, 1), ALL_AGENT), # False
      (ip.FullyArmed(Group([4,5,6]), Skill.Range, goal_level, 3), ALL_AGENT), # True
      (ip.FullyArmed(Group([4,5,6]), Skill.Range, goal_level+3, 1), ALL_AGENT), # False
      (ip.FullyArmed(Group([4,5,6]), Skill.Range, goal_level, 4), ALL_AGENT)] # False

    env = self._get_taskenv(test_tasks)

    # fully equip agents 4-6
    fully_equip = [4, 5, 6]
    for ent_id in fully_equip:
      env.realm.players[ent_id].skills.range.level.update(goal_level+2)
      # prepare the items
      item_list = [ itm(env.realm, goal_level) for itm in [
        Item.Hat, Item.Top, Item.Bottom, Item.Bow, Item.Shaving]]
      for itm in item_list:
        env.realm.players[ent_id].inventory.receive(itm)
        itm.use(env.realm.players[ent_id])
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [1, 3]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_hoard_gold_and_team(self): # HoardGold, TeamHoardGold
    agent_gold_goal = 10
    team_gold_goal = 30
    test_tasks = [ # (Predicate, Team), the reward is 1 by default
      (gp.HoardGold(Group([1]), agent_gold_goal), ALL_AGENT), # True
      (gp.HoardGold(Group([4,5,6]), agent_gold_goal), ALL_AGENT), # False
      (gp.HoardGold(Group([1,3,5]), team_gold_goal), ALL_AGENT), # True
      (gp.HoardGold(Group([2,4,6]), team_gold_goal), ALL_AGENT)] # False

    env = self._get_taskenv(test_tasks)

    # give gold to agents 1-3
    gold_struck = [1, 2, 3]
    for ent_id in gold_struck:
      env.realm.players[ent_id].gold.update(ent_id * 10)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [0, 2]
    self._check_result(env, test_tasks, infos, true_task)
    g = sum(env.realm.players[eid].gold.val for eid in Group([2,4,6]).agents)
    self._check_progress(test_tasks[3], infos, g / team_gold_goal)

    # DONE

  def test_exchange_gold_predicates(self): # Earn Gold, Spend Gold, Make Profit
    gold_goal = 10
    test_tasks = [
      (gp.EarnGold(Group([1,2]), gold_goal), ALL_AGENT), # True
      (gp.EarnGold(Group([2,4]), gold_goal), ALL_AGENT), # False
      (gp.SpendGold(Group([1]), 5), ALL_AGENT), # False -> True
      (gp.SpendGold(Group([1]), 6), ALL_AGENT), # False,
      (gp.MakeProfit(Group([1,2]), 5), ALL_AGENT), # True,
      (gp.MakeProfit(Group([1]), 5), ALL_AGENT) # True -> False
    ]

    env = self._get_taskenv(test_tasks)
    players = env.realm.players

    # 8 gold earned for agent 1
    # 2 for agent 2
    env.realm.event_log.record(EventCode.EARN_GOLD, players[1], amount = 5)
    env.realm.event_log.record(EventCode.EARN_GOLD, players[1], amount = 3)
    env.realm.event_log.record(EventCode.EARN_GOLD, players[2], amount = 2)

    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [0,4,5]
    self._check_result(env, test_tasks, infos, true_task)
    self._check_progress(test_tasks[1], infos, 2 / gold_goal)

    env.realm.event_log.record(EventCode.BUY_ITEM, players[1],
                               item=Item.Ration(env.realm,1),
                               price=5)
    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [0,2,4]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_count_event(self): # CountEvent
    test_tasks = [
      (bp.CountEvent(Group([1]),"EAT_FOOD",1), ALL_AGENT), # True
      (bp.CountEvent(Group([1]),"EAT_FOOD",2), ALL_AGENT), # False
      (bp.CountEvent(Group([1]),"DRINK_WATER",1), ALL_AGENT), # False
      (bp.CountEvent(Group([1,2]),"GIVE_GOLD",1), ALL_AGENT) # True
    ]

    # 1 Drinks water once
    # 2 Gives gold once
    env = self._get_taskenv(test_tasks)
    players = env.realm.players
    env.realm.event_log.record(EventCode.EAT_FOOD, players[1])
    env.realm.event_log.record(EventCode.GIVE_GOLD, players[2])
    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [0,3]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE

  def test_score_hit(self): # ScoreHit
    test_tasks = [
      (bp.ScoreHit(Group([1]), Skill.Mage, 2), ALL_AGENT), # False -> True
      (bp.ScoreHit(Group([1]), Skill.Melee, 1), ALL_AGENT) # True
    ]
    env = self._get_taskenv(test_tasks)
    players = env.realm.players

    env.realm.event_log.record(EventCode.SCORE_HIT,
                               players[1],
                               combat_style = Skill.Mage,
                               damage=1)
    env.realm.event_log.record(EventCode.SCORE_HIT,
                               players[1],
                               combat_style = Skill.Melee,
                               damage=1)

    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [1]
    self._check_result(env, test_tasks, infos, true_task)
    self._check_progress(test_tasks[0], infos, 0.5)

    env.realm.event_log.record(EventCode.SCORE_HIT,
                               players[1],
                               combat_style = Skill.Mage,
                               damage=1)
    env.realm.event_log.record(EventCode.SCORE_HIT,
                               players[1],
                               combat_style = Skill.Melee,
                               damage=1)

    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [0,1]
    self._check_result(env, test_tasks, infos, true_task)

    # DONE
  
  def test_item_event_predicates(self): # Consume, Harvest, List, Buy
    for predicate, event_type in [(ip.ConsumeItem, 'CONSUME_ITEM'),
                                  (ip.HarvestItem, 'HARVEST_ITEM'),
                                  (ip.ListItem, 'LIST_ITEM'),
                                  (ip.BuyItem, 'BUY_ITEM')]:
      id_ = getattr(EventCode, event_type)
      lvl = random.randint(5,10)
      quantity = random.randint(5,10)
      true_item = Item.Ration
      false_item = Item.Poultice
      test_tasks = [
        (predicate(Group([1,3,5]), true_item, lvl, quantity), ALL_AGENT), # True
        (predicate(Group([2]), true_item, lvl, quantity), ALL_AGENT), # False
        (predicate(Group([4]), true_item, lvl, quantity), ALL_AGENT), # False
        (predicate(Group([6]), true_item, lvl, quantity), ALL_AGENT) # False
      ]

      env = self._get_taskenv(test_tasks)
      players = env.realm.players
      # True case: split the required items between 3 and 5
      for player in (1,3):
        for _ in range(quantity // 2 + 1):
          env.realm.event_log.record(id_,
                                players[player],
                                price=1,
                                item=true_item(env.realm,
                                               lvl+random.randint(0,3)))

      # False case 1: Quantity
      for _ in range(quantity-1):
          env.realm.event_log.record(id_,
                                players[2],
                                price=1,
                                item=true_item(env.realm, lvl))

      # False case 2: Type
      for _ in range(quantity+1):
          env.realm.event_log.record(id_,
                                players[4],
                                price=1,
                                item=false_item(env.realm, lvl))
      # False case 3: Level
      for _ in range(quantity+1):
          env.realm.event_log.record(id_,
                                players[4],
                                price=1,
                                item=true_item(env.realm,
                                               random.randint(0,lvl-1)))
      env.obs = env._compute_observations()
      _, _, _, infos = env.step({})
      true_task = [0]
      self._check_result(env, test_tasks, infos, true_task)

    # DONE
if __name__ == '__main__':
  unittest.main()
