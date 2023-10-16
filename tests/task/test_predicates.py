import unittest
from typing import List, Tuple, Union, Iterable
import random

from tests.testhelpers import ScriptedAgentTestConfig, provide_item
from tests.testhelpers import change_spawn_pos as change_agent_pos

from scripted.baselines import Sleeper

from nmmo.entity.entity import EntityState
from nmmo.systems import item as Item
from nmmo.systems import skill as Skill
from nmmo.lib import material as Material
from nmmo.lib.event_code import EventCode

# pylint: disable=import-error
from nmmo.core.env import Env
from nmmo.task.predicate_api import Predicate, make_predicate
from nmmo.task.task_api import OngoingTask
from nmmo.task.group import Group
import nmmo.task.base_predicates as bp

# use the constant reward of 1 for testing predicates
NUM_AGENT = 6
ALL_AGENT = list(range(1, NUM_AGENT+1))


class TestBasePredicate(unittest.TestCase):
  # pylint: disable=protected-access,invalid-name,no-member

  def _get_taskenv(self,
                   test_preds: List[Tuple[Predicate, Union[Iterable[int], int]]],
                   grass_map=False):

    config = ScriptedAgentTestConfig()
    config.PLAYERS = [Sleeper]
    config.PLAYER_N = NUM_AGENT
    config.IMMORTAL = True
    config.ALLOW_MULTI_TASKS_PER_AGENT = True

    # OngoingTask keeps evaluating and returns progress as the reward
    #   vs. Task stops evaluating once the task is completed, returns reward = delta(progress)
    test_tasks = [OngoingTask(pred, assignee) for pred, assignee in test_preds]

    env = Env(config)
    env.reset(make_task_fn=lambda: test_tasks)

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

  def _check_result(self, env, test_preds, infos, true_task):
    for tid, (predicate, assignee) in enumerate(test_preds):
      # result is cached when at least one assignee is alive so that the task is evaled
      if len(set(assignee) & set(infos)) > 0:
        self.assertEqual(int(env.game_state.cache_result[predicate.name]),
                         int(tid in true_task))

      for ent_id in infos:
        if ent_id in assignee:
          # the agents that are assigned the task get evaluated for reward
          self.assertEqual(int(infos[ent_id]['task'][env.tasks[tid].name]['reward']),
                           int(tid in true_task))
        else:
          # the agents that are not assigned the task are not evaluated
          self.assertTrue(env.tasks[tid].name not in infos[ent_id]['task'])

  def _check_progress(self, task, infos, value):
    """ Tasks return a float in the range 0-1 indicating completion progress.
    """
    for ent_id in infos:
      if ent_id in task.assignee:
        self.assertAlmostEqual(infos[ent_id]['task'][task.name]['progress'],value)

  def test_tickge_stay_alive_rip(self):
    tickge_pred_cls = make_predicate(bp.TickGE)
    stay_alive_pred_cls = make_predicate(bp.StayAlive)
    all_dead_pred_cls = make_predicate(bp.AllDead)

    tick_true = 5
    death_note = [1, 2, 3]
    test_preds = [ # (instantiated predicate, task assignee)
      (tickge_pred_cls(Group([1]), tick_true), ALL_AGENT),
      (stay_alive_pred_cls(Group([1,3])), ALL_AGENT),
      (stay_alive_pred_cls(Group([3,4])), [1,2]),
      (stay_alive_pred_cls(Group([4])), [5,6]),
      (all_dead_pred_cls(Group([1,3])), ALL_AGENT),
      (all_dead_pred_cls(Group([3,4])), [1,2]),
      (all_dead_pred_cls(Group([4])), [5,6])]

    env = self._get_taskenv(test_preds)

    for _ in range(tick_true-1):
      _, _, _, infos = env.step({})

    # TickGE_5 is false. All agents are alive,
    # so all StayAlive (ti in [1,2,3]) tasks are true
    # and all AllDead tasks (ti in [4, 5, 6]) are false

    true_task = [1, 2, 3]
    self._check_result(env, test_preds, infos, true_task)
    self._check_progress(env.tasks[0], infos, (tick_true-1) / tick_true)

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

    # TickGE_5 is true. Agents 1-3 are dead, so
    # StayAlive(1,3) and StayAlive(3,4) are false, StayAlive(4) is true
    # AllDead(1,3) is true, AllDead(3,4) and AllDead(4) are false
    true_task = [0, 3, 4]
    self._check_result(env, test_preds, infos, true_task)

    # 3 is dead but 4 is alive. Half of agents killed, 50% completion.
    self._check_progress(env.tasks[5], infos, 0.5)

    # DONE

  def test_can_see_tile(self):
    canseetile_pred_cls = make_predicate(bp.CanSeeTile)

    a1_target = Material.Foilage
    a2_target = Material.Water
    test_preds = [ # (instantiated predicate, task assignee)
      (canseetile_pred_cls(Group([1]), a1_target), ALL_AGENT), # True
      (canseetile_pred_cls(Group([1,3,5]), a2_target), ALL_AGENT), # False
      (canseetile_pred_cls(Group([2]), a2_target), [1,2,3]), # True
      (canseetile_pred_cls(Group([2,5,6]), a1_target), ALL_AGENT), # False
      (canseetile_pred_cls(Group(ALL_AGENT), a2_target), [2,3,4])] # True

    # setup env with all grass map
    env = self._get_taskenv(test_preds, grass_map=True)

    # Two corners to the target materials
    BORDER = env.config.MAP_BORDER
    MS = env.config.MAP_CENTER + BORDER
    tile = env.realm.map.tiles[BORDER,MS-2]
    tile.material = Material.Foilage
    tile.material_id.update(Material.Foilage.index)

    tile = env.realm.map.tiles[MS-1,BORDER]
    tile.material = Material.Water
    tile.material_id.update(Material.Water.index)

    # All agents to one corner
    for ent_id in env.realm.players:
      change_agent_pos(env.realm,ent_id,(BORDER,BORDER))

    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    # no target tiles are found, so all are false
    true_task = []
    self._check_result(env, test_preds, infos, true_task)

    # Team one to foilage, team two to water
    change_agent_pos(env.realm,1,(BORDER,MS-2)) # agent 1, team 0, foilage
    change_agent_pos(env.realm,2,(MS-2,BORDER)) # agent 2, team 1, water
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # t0, t2, t4 are true
    true_task = [0, 2, 4]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_can_see_agent(self):
    cansee_agent_pred_cls = make_predicate(bp.CanSeeAgent)
    cansee_group_pred_cls = make_predicate(bp.CanSeeGroup)

    search_target = 1
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (cansee_agent_pred_cls(Group([1]), search_target), ALL_AGENT), # Always True
      (cansee_agent_pred_cls(Group([2]), search_target), [2,3,4]), # False -> True -> True
      (cansee_agent_pred_cls(Group([3,4,5]), search_target), [1,2,3]), # False -> False -> True
      (cansee_group_pred_cls(Group([1]), [3,4]), ALL_AGENT)] # False -> False -> True

    env = self._get_taskenv(test_preds, grass_map=True)

    # All agents to one corner
    BORDER = env.config.MAP_BORDER
    MS = env.config.MAP_CENTER + BORDER
    for ent_id in env.realm.players:
      change_agent_pos(env.realm,ent_id,(BORDER,BORDER)) # the map border

    # Teleport agent 1 to the opposite corner
    change_agent_pos(env.realm,1,(MS-2,MS-2))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # Only CanSeeAgent(Group([1]), search_target) is true, others are false
    true_task = [0]
    self._check_result(env, test_preds, infos, true_task)

    # Teleport agent 2 to agent 1's pos
    change_agent_pos(env.realm,2,(MS-2,MS-2))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # SearchAgent(Team([2]), search_target) is also true
    true_task = [0,1]
    self._check_result(env, test_preds, infos, true_task)

    # Teleport agent 3 to agent 1s position
    change_agent_pos(env.realm,3,(MS-2,MS-2))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})
    true_task = [0,1,2,3]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_occupy_tile(self):
    occupy_tile_pred_cls = make_predicate(bp.OccupyTile)

    target_tile = (30, 30)
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (occupy_tile_pred_cls(Group([1]), *target_tile), ALL_AGENT), # False -> True
      (occupy_tile_pred_cls(Group([1,2,3]), *target_tile), [4,5,6]), # False -> True
      (occupy_tile_pred_cls(Group([2]), *target_tile), [2,3,4]), # False
      (occupy_tile_pred_cls(Group([3,4,5]), *target_tile), [1,2,3])] # False

    # make all tiles habitable
    env = self._get_taskenv(test_preds, grass_map=True)

    # All agents to one corner
    BORDER = env.config.MAP_BORDER
    for ent_id in env.realm.players:
      change_agent_pos(env.realm,ent_id,(BORDER,BORDER))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # all tasks must be false
    true_task = []
    self._check_result(env, test_preds, infos, true_task)

    # teleport agent 1 to the target tile, agent 2 to the adjacent tile
    change_agent_pos(env.realm,1,target_tile)
    change_agent_pos(env.realm,2,(target_tile[0],target_tile[1]-1))
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    # tid 0 and 1 should be true: OccupyTile(Group([1]), *target_tile)
    #  & OccupyTile(Group([1,2,3]), *target_tile)
    true_task = [0, 1]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_distance_traveled(self):
    distance_traveled_pred_cls = make_predicate(bp.DistanceTraveled)

    agent_dist = 6
    team_dist = 10
    # NOTE: when evaluating predicates, to whom tasks are assigned are irrelevant
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (distance_traveled_pred_cls(Group([1]), agent_dist), ALL_AGENT), # False -> True
      (distance_traveled_pred_cls(Group([2, 5]), agent_dist), ALL_AGENT), # False
      (distance_traveled_pred_cls(Group([3, 4]), agent_dist), ALL_AGENT), # False
      (distance_traveled_pred_cls(Group([1, 2, 3]), team_dist), ALL_AGENT), # False -> True
      (distance_traveled_pred_cls(Group([6]), agent_dist), ALL_AGENT)] # False

    # make all tiles habitable
    env = self._get_taskenv(test_preds, grass_map=True)

    _, _, _, infos = env.step({})

    # one cannot accomplish these goals in the first tick, so all false
    true_task = []
    self._check_result(env, test_preds, infos, true_task)

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
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_all_members_within_range(self):
    within_range_pred_cls = make_predicate(bp.AllMembersWithinRange)

    dist_123 = 1
    dist_135 = 5
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (within_range_pred_cls(Group([1]), dist_123), ALL_AGENT), # Always true for group of 1
      (within_range_pred_cls(Group([1,2]), dist_123), ALL_AGENT), # True
      (within_range_pred_cls(Group([1,3]), dist_123), ALL_AGENT), # True
      (within_range_pred_cls(Group([2,3]), dist_123), ALL_AGENT), # False
      (within_range_pred_cls(Group([1,3,5]), dist_123), ALL_AGENT), # False
      (within_range_pred_cls(Group([1,3,5]), dist_135), ALL_AGENT), # True
      (within_range_pred_cls(Group([2,4,6]), dist_135), ALL_AGENT)] # False

    # make all tiles habitable
    env = self._get_taskenv(test_preds, grass_map=True)

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
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_attain_skill(self):
    attain_skill_pred_cls = make_predicate(bp.AttainSkill)

    goal_level = 5
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (attain_skill_pred_cls(Group([1]), Skill.Melee, goal_level, 1), ALL_AGENT), # False
      (attain_skill_pred_cls(Group([2]), Skill.Melee, goal_level, 1), ALL_AGENT), # False
      (attain_skill_pred_cls(Group([1]), Skill.Range, goal_level, 1), ALL_AGENT), # True
      (attain_skill_pred_cls(Group([1,3]), Skill.Fishing, goal_level, 1), ALL_AGENT), # True
      (attain_skill_pred_cls(Group([1,2,3]), Skill.Carving, goal_level, 3), ALL_AGENT), # False
      (attain_skill_pred_cls(Group([2,4]), Skill.Carving, goal_level, 2), ALL_AGENT)] # True

    env = self._get_taskenv(test_preds)

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
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_gain_experience(self):
    attain_gain_exp_cls = make_predicate(bp.GainExperience)

    goal_exp = 5
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (attain_gain_exp_cls(Group([1]), Skill.Melee, goal_exp, 1), ALL_AGENT), # False
      (attain_gain_exp_cls(Group([2]), Skill.Melee, goal_exp, 1), ALL_AGENT), # False
      (attain_gain_exp_cls(Group([1]), Skill.Range, goal_exp, 1), ALL_AGENT), # True
      (attain_gain_exp_cls(Group([1,3]), Skill.Fishing, goal_exp, 1), ALL_AGENT), # True
      (attain_gain_exp_cls(Group([1,2,3]), Skill.Carving, goal_exp, 3), ALL_AGENT), # False
      (attain_gain_exp_cls(Group([2,4]), Skill.Carving, goal_exp, 2), ALL_AGENT)] # True

    env = self._get_taskenv(test_preds)

    # AttainSkill(Group([1]), Skill.Melee, goal_level, 1) is false
    # AttainSkill(Group([2]), Skill.Melee, goal_level, 1) is false
    env.realm.players[1].skills.melee.exp.update(goal_exp-1)
    # AttainSkill(Group([1]), Skill.Range, goal_level, 1) is true
    env.realm.players[1].skills.range.exp.update(goal_exp)
    # AttainSkill(Group([1,3]), Skill.Fishing, goal_level, 1) is true
    env.realm.players[1].skills.fishing.exp.update(goal_exp)
    # AttainSkill(Group([1,2,3]), Skill.Carving, goal_level, 3) is false
    env.realm.players[1].skills.carving.exp.update(goal_exp)
    env.realm.players[2].skills.carving.exp.update(goal_exp)
    # AttainSkill(Group([2,4]), Skill.Carving, goal_level, 2) is true
    env.realm.players[4].skills.carving.exp.update(goal_exp+2)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [2, 3, 5]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_inventory_space_ge_not(self):
    inv_space_ge_pred_cls = make_predicate(bp.InventorySpaceGE)

    # also test NOT InventorySpaceGE
    target_space = 3
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (inv_space_ge_pred_cls(Group([1]), target_space), ALL_AGENT), # True -> False
      (inv_space_ge_pred_cls(Group([2,3]), target_space), ALL_AGENT), # True
      (inv_space_ge_pred_cls(Group([1,2,3]), target_space), ALL_AGENT), # True -> False
      (inv_space_ge_pred_cls(Group([1,2,3,4]), target_space+1), ALL_AGENT), # False
      (~inv_space_ge_pred_cls(Group([1]), target_space+1), ALL_AGENT), # True
      (~inv_space_ge_pred_cls(Group([1,2,3]), target_space), ALL_AGENT), # False -> True
      (~inv_space_ge_pred_cls(Group([1,2,3,4]), target_space+1), ALL_AGENT)] # True

    env = self._get_taskenv(test_preds)

    # add one items to agent 1 within the limit
    capacity = env.realm.players[1].inventory.capacity
    provide_item(env.realm, 1, Item.Ration, level=1, quantity=capacity-target_space)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    self.assertTrue(env.realm.players[1].inventory.space >= target_space)
    true_task = [0, 1, 2, 4, 6]
    self._check_result(env, test_preds, infos, true_task)

    # add one more item to agent 1
    provide_item(env.realm, 1, Item.Ration, level=1, quantity=1)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    self.assertTrue(env.realm.players[1].inventory.space < target_space)
    true_task = [1, 4, 5, 6]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_own_equip_item(self):
    own_item_pred_cls = make_predicate(bp.OwnItem)
    equip_item_pred_cls = make_predicate(bp.EquipItem)

    # ration, level 2, quantity 3 (non-stackable)
    # ammo level 2, quantity 3 (stackable, equipable)
    goal_level = 2
    goal_quantity = 3
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (own_item_pred_cls(Group([1]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # False
      (own_item_pred_cls(Group([2]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # False
      (own_item_pred_cls(Group([1,2]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # True
      (own_item_pred_cls(Group([3]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # True
      (own_item_pred_cls(Group([4,5,6]), Item.Ration, goal_level, goal_quantity), ALL_AGENT), # F
      (equip_item_pred_cls(Group([4]), Item.Whetstone, goal_level, 1), ALL_AGENT), # False
      (equip_item_pred_cls(Group([4,5]), Item.Whetstone, goal_level, 1), ALL_AGENT), # True
      (equip_item_pred_cls(Group([4,5,6]), Item.Whetstone, goal_level, 2), ALL_AGENT)] # True

    env = self._get_taskenv(test_preds)

    # set the level, so that agents 4-6 can equip the Whetstone
    equip_stone = [4, 5, 6]
    for ent_id in equip_stone:
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
    ent_id = 4 # EquipItem(Group([4]), Item.Whetstone, goal_level, 1) is false
    provide_item(env.realm, ent_id, Item.Whetstone, level=1, quantity=4)
    ent_id = 5 # EquipItem(Group([4,5]), Item.Whetstone, goal_level, 1) is true
    provide_item(env.realm, ent_id, Item.Whetstone, level=4, quantity=1)
    ent_id = 6 # EquipItem(Group([4,5,6]), Item.Whetstone, goal_level, 2) is true
    provide_item(env.realm, ent_id, Item.Whetstone, level=2, quantity=4)
    for ent_id in [4, 5, 6]:
      whetstone = env.realm.players[ent_id].inventory.items[0]
      whetstone.use(env.realm.players[ent_id])
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [2, 3, 6, 7]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_fully_armed(self):
    fully_armed_pred_cls = make_predicate(bp.FullyArmed)

    goal_level = 5
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (fully_armed_pred_cls(Group([1,2,3]), Skill.Range, goal_level, 1), ALL_AGENT), # False
      (fully_armed_pred_cls(Group([3,4]), Skill.Range, goal_level, 1), ALL_AGENT), # True
      (fully_armed_pred_cls(Group([4]), Skill.Melee, goal_level, 1), ALL_AGENT), # False
      (fully_armed_pred_cls(Group([4,5,6]), Skill.Range, goal_level, 3), ALL_AGENT), # True
      (fully_armed_pred_cls(Group([4,5,6]), Skill.Range, goal_level+3, 1), ALL_AGENT), # False
      (fully_armed_pred_cls(Group([4,5,6]), Skill.Range, goal_level, 4), ALL_AGENT)] # False

    env = self._get_taskenv(test_preds)

    # fully equip agents 4-6
    fully_equip = [4, 5, 6]
    for ent_id in fully_equip:
      env.realm.players[ent_id].skills.range.level.update(goal_level+2)
      # prepare the items
      item_list = [ itm(env.realm, goal_level) for itm in [
        Item.Hat, Item.Top, Item.Bottom, Item.Bow, Item.Arrow]]
      for itm in item_list:
        env.realm.players[ent_id].inventory.receive(itm)
        itm.use(env.realm.players[ent_id])
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [1, 3]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_hoard_gold_and_team(self): # HoardGold, TeamHoardGold
    hoard_gold_pred_cls = make_predicate(bp.HoardGold)

    agent_gold_goal = 10
    team_gold_goal = 30
    test_preds = [ # (Predicate, Team), the reward is 1 by default
      (hoard_gold_pred_cls(Group([1]), agent_gold_goal), ALL_AGENT), # True
      (hoard_gold_pred_cls(Group([4,5,6]), agent_gold_goal), ALL_AGENT), # False
      (hoard_gold_pred_cls(Group([1,3,5]), team_gold_goal), ALL_AGENT), # True
      (hoard_gold_pred_cls(Group([2,4,6]), team_gold_goal), ALL_AGENT)] # False

    env = self._get_taskenv(test_preds)

    # give gold to agents 1-3
    gold_struck = [1, 2, 3]
    for ent_id in gold_struck:
      env.realm.players[ent_id].gold.update(ent_id * 10)
    env.obs = env._compute_observations()

    _, _, _, infos = env.step({})

    true_task = [0, 2]
    self._check_result(env, test_preds, infos, true_task)
    g = sum(env.realm.players[eid].gold.val for eid in Group([2,4,6]).agents)
    self._check_progress(env.tasks[3], infos, g / team_gold_goal)

    # DONE

  def test_exchange_gold_predicates(self): # Earn Gold, Spend Gold, Make Profit
    earn_gold_pred_cls = make_predicate(bp.EarnGold)
    spend_gold_pred_cls = make_predicate(bp.SpendGold)
    make_profit_pred_cls = make_predicate(bp.MakeProfit)

    gold_goal = 10
    test_preds = [
      (earn_gold_pred_cls(Group([1,2]), gold_goal), ALL_AGENT), # True
      (earn_gold_pred_cls(Group([2,4]), gold_goal), ALL_AGENT), # False
      (spend_gold_pred_cls(Group([1]), 5), ALL_AGENT), # False -> True
      (spend_gold_pred_cls(Group([1]), 6), ALL_AGENT), # False,
      (make_profit_pred_cls(Group([1,2]), 5), ALL_AGENT), # True,
      (make_profit_pred_cls(Group([1]), 5), ALL_AGENT) # True -> False
    ]

    env = self._get_taskenv(test_preds)
    players = env.realm.players

    # 8 gold earned for agent 1
    # 2 for agent 2
    env.realm.event_log.record(EventCode.EARN_GOLD, players[1], amount = 5)
    env.realm.event_log.record(EventCode.EARN_GOLD, players[1], amount = 3)
    env.realm.event_log.record(EventCode.EARN_GOLD, players[2], amount = 2)

    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [0,4,5]
    self._check_result(env, test_preds, infos, true_task)
    self._check_progress(env.tasks[1], infos, 2 / gold_goal)

    env.realm.event_log.record(EventCode.BUY_ITEM, players[1],
                               item=Item.Ration(env.realm,1),
                               price=5)
    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [0,2,4]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_count_event(self): # CountEvent
    count_event_pred_cls = make_predicate(bp.CountEvent)

    test_preds = [
      (count_event_pred_cls(Group([1]),"EAT_FOOD",1), ALL_AGENT), # True
      (count_event_pred_cls(Group([1]),"EAT_FOOD",2), ALL_AGENT), # False
      (count_event_pred_cls(Group([1]),"DRINK_WATER",1), ALL_AGENT), # False
      (count_event_pred_cls(Group([1,2]),"GIVE_GOLD",1), ALL_AGENT) # True
    ]

    # 1 Drinks water once
    # 2 Gives gold once
    env = self._get_taskenv(test_preds)
    players = env.realm.players
    env.realm.event_log.record(EventCode.EAT_FOOD, players[1])
    env.realm.event_log.record(EventCode.GIVE_GOLD, players[2])
    env.obs = env._compute_observations()
    _, _, _, infos = env.step({})

    true_task = [0,3]
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_score_hit(self): # ScoreHit
    score_hit_pred_cls = make_predicate(bp.ScoreHit)

    test_preds = [
      (score_hit_pred_cls(Group([1]), Skill.Mage, 2), ALL_AGENT), # False -> True
      (score_hit_pred_cls(Group([1]), Skill.Melee, 1), ALL_AGENT) # True
    ]
    env = self._get_taskenv(test_preds)
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
    self._check_result(env, test_preds, infos, true_task)
    self._check_progress(env.tasks[0], infos, 0.5)

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
    self._check_result(env, test_preds, infos, true_task)

    # DONE

  def test_defeat_entity(self): # PlayerKill
    defeat_pred_cls = make_predicate(bp.DefeatEntity)

    test_preds = [
      (defeat_pred_cls(Group([1]), 'npc', level=1, num_agent=1), ALL_AGENT),
      (defeat_pred_cls(Group([1]), 'player', level=2, num_agent=2), ALL_AGENT)]
    env = self._get_taskenv(test_preds)
    players = env.realm.players
    npcs = env.realm.npcs

    # set levels
    npcs[-1].skills.melee.level.update(1)
    npcs[-1].skills.range.level.update(1)
    npcs[-1].skills.mage.level.update(1)
    self.assertEqual(npcs[-1].attack_level, 1)
    self.assertEqual(players[2].attack_level, 1)
    players[3].skills.melee.level.update(3)
    players[4].skills.melee.level.update(2)

    # killing player 2 does not progress the both tasks
    env.realm.event_log.record(EventCode.PLAYER_KILL, players[1],
                               target=players[2]) # level 1 player
    _, _, _, infos = env.step({})

    true_task = [] # all false
    self._check_result(env, test_preds, infos, true_task)
    for task in env.tasks:
      self._check_progress(task, infos, 0)

    # killing npc -1 completes the first task
    env.realm.event_log.record(EventCode.PLAYER_KILL, players[1],
                               target=npcs[-1]) # level 1 npc
    _, _, _, infos = env.step({})

    true_task = [0]
    self._check_result(env, test_preds, infos, true_task)
    self._check_progress(env.tasks[0], infos, 1)

    # killing player 3 makes half progress on the second task
    env.realm.event_log.record(EventCode.PLAYER_KILL, players[1],
                               target=players[3]) # level 3 player
    _, _, _, infos = env.step({})
    self._check_progress(env.tasks[1], infos, .5)

    # killing player 4 completes the second task
    env.realm.event_log.record(EventCode.PLAYER_KILL, players[1],
                               target=players[4]) # level 2 player
    _, _, _, infos = env.step({})

    true_task = [0,1]
    self._check_result(env, test_preds, infos, true_task)
    self._check_progress(env.tasks[1], infos, 1)

    # DONE

  def test_item_event_predicates(self): # Consume, Harvest, List, Buy
    for pred_fn, event_type in [(bp.ConsumeItem, 'CONSUME_ITEM'),
                                  (bp.HarvestItem, 'HARVEST_ITEM'),
                                  (bp.ListItem, 'LIST_ITEM'),
                                  (bp.BuyItem, 'BUY_ITEM')]:
      predicate = make_predicate(pred_fn)
      id_ = getattr(EventCode, event_type)
      lvl = random.randint(5,10)
      quantity = random.randint(5,10)
      true_item = Item.Ration
      false_item = Item.Potion
      test_preds = [
        (predicate(Group([1,3,5]), true_item, lvl, quantity), ALL_AGENT), # True
        (predicate(Group([2]), true_item, lvl, quantity), ALL_AGENT), # False
        (predicate(Group([4]), true_item, lvl, quantity), ALL_AGENT), # False
        (predicate(Group([6]), true_item, lvl, quantity), ALL_AGENT) # False
      ]

      env = self._get_taskenv(test_preds)
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
      self._check_result(env, test_preds, infos, true_task)

    # DONE

if __name__ == '__main__':
  unittest.main()
