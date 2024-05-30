import unittest
import logging

from tests.testhelpers import ScriptedTestTemplate, change_spawn_pos, provide_item

from nmmo.core import action
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState
from scripted import baselines

RANDOM_SEED = 985

LOGFILE = None  # 'tests/action/test_destroy_give_gold.log'

class TestDestroyGiveGold(ScriptedTestTemplate):
  # pylint: disable=protected-access,multiple-statements,no-member

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # config specific to the tests here
    cls.config.set("PLAYERS", [baselines.Melee, baselines.Range])
    cls.config.set("PLAYER_N", 6)

    cls.policy = { 1:'Melee', 2:'Range', 3:'Melee', 4:'Range', 5:'Melee', 6:'Range' }
    cls.spawn_locs = { 1:(17,17), 2:(21,21), 3:(17,17), 4:(21,21), 5:(21,21), 6:(17,17) }
    cls.ammo = { 1:Item.Whetstone, 2:Item.Arrow, 3:Item.Whetstone,
                 4:Item.Arrow, 5:Item.Whetstone, 6:Item.Arrow }

    if LOGFILE:  # for debugging
      logging.basicConfig(filename=LOGFILE, level=logging.INFO)

  def test_destroy(self):
    env = self._setup_env(random_seed=RANDOM_SEED)

    # check if level-0 and level-3 ammo are in the correct place
    for ent_id in self.policy:
      for idx, lvl in enumerate(self.item_level):
        assert self.item_sig[ent_id][idx] == (self.ammo[ent_id], lvl)

    # equipped items cannot be destroyed, i.e. that action will be ignored
    # this should be marked in the mask too

    """ First tick """ # First tick actions: USE (equip) level-0 ammo
    env.step({ ent_id: { action.Use: { action.InventoryItem:
        env.obs[ent_id].inventory.sig(*self.item_sig[ent_id][0]) } # level-0 ammo
      } for ent_id in self.policy })

    # check if the agents have equipped the ammo
    for ent_id in self.policy:
      ent_obs = env.obs[ent_id]
      inv_idx = ent_obs.inventory.sig(*self.item_sig[ent_id][0]) # level-0 ammo
      self.assertEqual(1, # True
        ItemState.parse_array(ent_obs.inventory.values[inv_idx]).equipped)

      # check Destroy InventoryItem mask -- one cannot destroy equipped item
      for item_sig in self.item_sig[ent_id]:
        if item_sig == (self.ammo[ent_id], 0): # level-0 ammo
          self.assertFalse(self._check_inv_mask(ent_obs, action.Destroy, item_sig))
        else:
          # other items can be destroyed
          self.assertTrue(self._check_inv_mask(ent_obs, action.Destroy, item_sig))

    """ Second tick """ # Second tick actions: DESTROY ammo
    actions = {}

    for ent_id in self.policy:
      if ent_id in [1, 2]:
        # agent 1 & 2, destroy the level-3 ammos, which are valid
        actions[ent_id] = { action.Destroy:
          { action.InventoryItem: env.obs[ent_id].inventory.sig(*self.item_sig[ent_id][1]) } }
      else:
        # other agents: destroy the equipped level-0 ammos, which are invalid
        actions[ent_id] = { action.Destroy:
          { action.InventoryItem: env.obs[ent_id].inventory.sig(*self.item_sig[ent_id][0]) } }
    env.step(actions)

    # check if the ammos were destroyed
    for ent_id in self.policy:
      if ent_id in [1, 2]:
        inv_idx = env.obs[ent_id].inventory.sig(*self.item_sig[ent_id][1])
        self.assertTrue(inv_idx is None) # valid actions, thus destroyed
      else:
        inv_idx = env.obs[ent_id].inventory.sig(*self.item_sig[ent_id][0])
        self.assertTrue(inv_idx is not None) # invalid actions, thus not destroyed

    # DONE

  def test_give_tile_npc(self):
    # cannot give to self (should be masked)
    # cannot give if not on the same tile (should be masked)
    # cannot give to the other team member (should be masked)
    # cannot give to npc (should be masked)
    env = self._setup_env(random_seed=RANDOM_SEED)

    # teleport the npc -1 to agent 5's location
    change_spawn_pos(env.realm, -1, self.spawn_locs[5])
    env._compute_observations()

    """ First tick actions """
    actions = {}
    test_cond = {}

    # agent 1: give ammo to agent 3 (valid: the same team, same tile)
    test_cond[1] = { 'tgt_id': 3, 'item_sig': self.item_sig[1][0],
                     'ent_mask': True, 'inv_mask': True, 'valid': True }
    # agent 2: give ammo to agent 2 (invalid: cannot give to self)
    test_cond[2] = { 'tgt_id': 2, 'item_sig': self.item_sig[2][0],
                     'ent_mask': False, 'inv_mask': True, 'valid': False }
    # agent 5: give ammo to npc -1 (invalid, should be masked)
    test_cond[5] = { 'tgt_id': -1, 'item_sig': self.item_sig[5][0],
                     'ent_mask': False, 'inv_mask': True, 'valid': False }

    actions = self._check_assert_make_action(env, action.Give, test_cond)
    env.step(actions)

    # check the results
    for ent_id, cond in test_cond.items():
      self.assertEqual( cond['valid'],
        env.obs[ent_id].inventory.sig(*cond['item_sig']) is None)

      if ent_id == 1: # agent 1 gave ammo stack to agent 3
        tgt_inv = env.obs[cond['tgt_id']].inventory
        inv_idx = tgt_inv.sig(*cond['item_sig'])
        self.assertEqual(2 * self.ammo_quantity,
          ItemState.parse_array(tgt_inv.values[inv_idx]).quantity)

    # DONE

  def test_give_equipped_listed(self):
    # cannot give equipped items (should be masked)
    # cannot give listed items (should be masked)
    env = self._setup_env(random_seed=RANDOM_SEED)

    """ First tick actions """
    actions = {}

    # agent 1: equip the ammo
    ent_id = 1; item_sig = self.item_sig[ent_id][0]
    self.assertTrue(
      self._check_inv_mask(env.obs[ent_id], action.Use, item_sig))
    actions[ent_id] = { action.Use: { action.InventoryItem:
        env.obs[ent_id].inventory.sig(*item_sig) } }

    # agent 2: list the ammo for sale
    ent_id = 2; price = 5; item_sig = self.item_sig[ent_id][0]
    self.assertTrue(
      self._check_inv_mask(env.obs[ent_id], action.Sell, item_sig))
    actions[ent_id] = { action.Sell: {
        action.InventoryItem: env.obs[ent_id].inventory.sig(*item_sig),
        action.Price: action.Price.index(price) } }

    env.step(actions)

    # Check the first tick actions
    # agent 1: equip the ammo
    ent_id = 1; item_sig = self.item_sig[ent_id][0]
    inv_idx = env.obs[ent_id].inventory.sig(*item_sig)
    self.assertEqual(1,
      ItemState.parse_array(env.obs[ent_id].inventory.values[inv_idx]).equipped)

    # agent 2: list the ammo for sale
    ent_id = 2; price = 5; item_sig = self.item_sig[ent_id][0]
    inv_idx = env.obs[ent_id].inventory.sig(*item_sig)
    self.assertEqual(price,
      ItemState.parse_array(env.obs[ent_id].inventory.values[inv_idx]).listed_price)
    self.assertTrue(env.obs[ent_id].inventory.id(inv_idx) in env.obs[ent_id].market.ids)

    """ Second tick actions """
    actions = {}
    test_cond = {}

    # agent 1: give equipped ammo to agent 3 (invalid: should be masked)
    test_cond[1] = { 'tgt_id': 3, 'item_sig': self.item_sig[1][0],
                     'ent_mask': True, 'inv_mask': False, 'valid': False }
    # agent 2: give listed ammo to agent 4 (invalid: should be masked)
    test_cond[2] = { 'tgt_id': 4, 'item_sig': self.item_sig[2][0],
                     'ent_mask': True, 'inv_mask': False, 'valid': False }

    actions = self._check_assert_make_action(env, action.Give, test_cond)
    env.step(actions)

    # Check the second tick actions
    # check the results
    for ent_id, cond in test_cond.items():
      self.assertEqual( cond['valid'],
        env.obs[ent_id].inventory.sig(*cond['item_sig']) is None)

    # DONE

  def test_give_full_inventory(self):
    # cannot give to an agent with the full inventory,
    #   but it's possible if the agent has the same ammo stack
    env = self._setup_env(random_seed=RANDOM_SEED)

    # make the inventory full for agents 1, 2
    extra_items = { (Item.Bottom, 0), (Item.Bottom, 3) }
    for ent_id in [1, 2]:
      for item_sig in extra_items:
        self.item_sig[ent_id].append(item_sig)
        provide_item(env.realm, ent_id, item_sig[0], item_sig[1], 1)
    env._compute_observations()

    # check if the inventory is full
    for ent_id in [1, 2]:
      self.assertEqual(env.obs[ent_id].inventory.len, env.config.ITEM_INVENTORY_CAPACITY)
      self.assertTrue(env.realm.players[ent_id].inventory.space == 0)

    """ First tick actions """
    actions = {}
    test_cond = {}

    # agent 3: give ammo to agent 1 (the same ammo stack, so valid)
    test_cond[3] = { 'tgt_id': 1, 'item_sig': self.item_sig[3][0],
                     'ent_mask': True, 'inv_mask': True, 'valid': True }
    # agent 4: give gloves to agent 2 (not the stack, so invalid)
    test_cond[4] = { 'tgt_id': 2, 'item_sig': self.item_sig[4][4],
                     'ent_mask': True, 'inv_mask': True, 'valid': False }

    actions = self._check_assert_make_action(env, action.Give, test_cond)
    env.step(actions)

    # Check the first tick actions
    # check the results
    for ent_id, cond in test_cond.items():
      self.assertEqual( cond['valid'],
        env.obs[ent_id].inventory.sig(*cond['item_sig']) is None)

      if ent_id == 3: # successfully gave the ammo stack to agent 1
        tgt_inv = env.obs[cond['tgt_id']].inventory
        inv_idx = tgt_inv.sig(*cond['item_sig'])
        self.assertEqual(2 * self.ammo_quantity,
          ItemState.parse_array(tgt_inv.values[inv_idx]).quantity)

    # DONE

  def test_give_gold(self):
    # cannot give to an npc (should be masked)
    # cannot give to self (should be masked)
    # cannot give if not on the same tile (should be masked)
    env = self._setup_env(random_seed=RANDOM_SEED)

    # teleport the npc -1 to agent 3's location
    change_spawn_pos(env.realm, -1, self.spawn_locs[3])
    env._compute_observations()

    test_cond = {}

    # NOTE: the below tests rely on the static execution order from 1 to N
    # agent 1: give gold to agent 3 (valid: same tile)
    test_cond[1] = { 'tgt_id': 3, 'gold': 1, 'ent_mask': True,
                     'ent_gold': self.init_gold-1, 'tgt_gold': self.init_gold+1 }
    # agent 2: give gold to agent 4 (valid: same tile)
    test_cond[2] = { 'tgt_id': 4, 'gold': self.init_gold, 'ent_mask': True,
                     'ent_gold': 0, 'tgt_gold': 2*self.init_gold }
    # agent 3: give gold to npc -1 (invalid: cannot give to npc)
    #  ent_gold is self.init_gold+1 because (3) got 1 gold from (1)
    test_cond[3] = { 'tgt_id': -1, 'gold': 1, 'ent_mask': False,
                     'ent_gold': self.init_gold+1, 'tgt_gold': self.init_gold }
    # agent 4: give -1 gold to 2 (invalid: cannot give minus gold)
    #  ent_gold is 2*self.init_gold because (4) got 5 gold from (2)
    #  tgt_gold is 0 because (2) gave all gold to (4)
    test_cond[4] = { 'tgt_id': 2, 'gold': -1, 'ent_mask': True,
                     'ent_gold': 2*self.init_gold, 'tgt_gold': 0 }

    actions = self._check_assert_make_action(env, action.GiveGold, test_cond)
    env.step(actions)

    # check the results
    for ent_id, cond in test_cond.items():
      self.assertEqual(cond['ent_gold'], env.realm.players[ent_id].gold.val)
      if cond['tgt_id'] > 0:
        self.assertEqual(cond['tgt_gold'], env.realm.players[cond['tgt_id']].gold.val)

    # DONE


if __name__ == '__main__':
  unittest.main()
