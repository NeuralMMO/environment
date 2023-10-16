import unittest
import logging

from tests.testhelpers import ScriptedTestTemplate, provide_item

from nmmo.core import action
from nmmo.systems import item as Item
from nmmo.systems.item import ItemState
from scripted import baselines

RANDOM_SEED = 985

LOGFILE = None  # 'tests/action/test_sell_buy.log'

class TestSellBuy(ScriptedTestTemplate):
  # pylint: disable=protected-access,multiple-statements,unsubscriptable-object,no-member

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    # config specific to the tests here
    cls.config.PLAYERS = [baselines.Melee, baselines.Range]
    cls.config.PLAYER_N = 6

    cls.policy = { 1:'Melee', 2:'Range', 3:'Melee', 4:'Range', 5:'Melee', 6:'Range' }
    cls.ammo = { 1:Item.Whetstone, 2:Item.Arrow, 3:Item.Whetstone,
                 4:Item.Arrow, 5:Item.Whetstone, 6:Item.Arrow }

    if LOGFILE:  # for debugging
      logging.basicConfig(filename=LOGFILE, level=logging.INFO)

  def test_sell_buy(self):
    # cannot list an item with 0 price --> impossible to do this
    # cannot list an equipped item for sale (should be masked)
    # cannot buy an item with the full inventory,
    #   but it's possible if the agent has the same ammo stack
    # cannot buy its own item (should be masked)
    # cannot buy an item if gold is not enough (should be masked)
    # cannot list an already listed item for sale (should be masked)
    env = self._setup_env(random_seed=RANDOM_SEED)

    # make the inventory full for agents 1, 2
    extra_items = { (Item.Bottom, 0), (Item.Bottom, 3) }
    for ent_id in [1, 2]:
      for item_sig in extra_items:
        self.item_sig[ent_id].append(item_sig)
        provide_item(env.realm, ent_id, item_sig[0], item_sig[1], 1)

    env.obs = env._compute_observations()

    # check if the inventory is full
    for ent_id in [1, 2]:
      self.assertEqual(env.obs[ent_id].inventory.len, env.config.ITEM_INVENTORY_CAPACITY)
      self.assertTrue(env.realm.players[ent_id].inventory.space == 0)

    """ First tick actions """
    # cannot list an item with 0 price
    actions = {}

    # agent 1-2: equip the ammo
    for ent_id in [1, 2]:
      item_sig = self.item_sig[ent_id][0]
      self.assertTrue(
        self._check_inv_mask(env.obs[ent_id], action.Use, item_sig))
      actions[ent_id] = { action.Use: { action.InventoryItem:
          env.obs[ent_id].inventory.sig(*item_sig) } }

    # agent 4: list the ammo for sale with price 0
    #   the zero in action.Price is deserialized into Discrete_1, so it's valid
    ent_id = 4; price = 0; item_sig = self.item_sig[ent_id][0]
    actions[ent_id] = { action.Sell: {
        action.InventoryItem: env.obs[ent_id].inventory.sig(*item_sig),
        action.Price: action.Price.edges[price] } }

    env.step(actions)

    # Check the first tick actions
    # agent 1-2: the ammo equipped, thus should be masked for sale
    for ent_id in [1, 2]:
      item_sig = self.item_sig[ent_id][0]
      inv_idx = env.obs[ent_id].inventory.sig(*item_sig)
      self.assertEqual(1, # equipped = true
        ItemState.parse_array(env.obs[ent_id].inventory.values[inv_idx]).equipped)
      self.assertFalse( # not allowed to list
        self._check_inv_mask(env.obs[ent_id], action.Sell, item_sig))

    """ Second tick actions """
    # listing the level-0 ammo with different prices
    # cannot list an equipped item for sale (should be masked)

    listing_price = { 1:1, 2:5, 3:15, 5:2 } # gold
    for ent_id, price in listing_price.items():
      item_sig = self.item_sig[ent_id][0]
      actions[ent_id] = { action.Sell: {
          action.InventoryItem: env.obs[ent_id].inventory.sig(*item_sig),
          action.Price: action.Price.edges[price-1] } }

    env.step(actions)

    # Check the second tick actions
    # agent 1-2: the ammo equipped, thus not listed for sale
    # agent 3-5's ammos listed for sale
    for ent_id, price in listing_price.items():
      item_id = env.obs[ent_id].inventory.id(0)

      if ent_id in [1, 2]: # failed to list for sale
        self.assertFalse(item_id in env.obs[ent_id].market.ids) # not listed
        self.assertEqual(0,
          ItemState.parse_array(env.obs[ent_id].inventory.values[0]).listed_price)

      else: # should succeed to list for sale
        self.assertTrue(item_id in env.obs[ent_id].market.ids) # listed
        self.assertEqual(price, # sale price set
          ItemState.parse_array(env.obs[ent_id].inventory.values[0]).listed_price)

        # should not buy mine
        self.assertFalse( self._check_mkt_mask(env.obs[ent_id], item_id))

        # should not list the same item twice
        self.assertFalse(
          self._check_inv_mask(env.obs[ent_id], action.Sell, self.item_sig[ent_id][0]))

    """ Third tick actions """
    # cannot buy an item with the full inventory,
    #   but it's possible if the agent has the same ammo stack
    # cannot buy its own item (should be masked)
    # cannot buy an item if gold is not enough (should be masked)
    # cannot list an already listed item for sale (should be masked)

    test_cond = {}

    # agent 1: buy agent 5's ammo (valid: 1 has the same ammo stack)
    #   although 1's inventory is full, this action is valid
    agent5_ammo = env.obs[5].inventory.id(0)
    test_cond[1] = { 'item_id': agent5_ammo, 'mkt_mask': True }

    # agent 2: buy agent 5's ammo (invalid: full space and no same stack)
    test_cond[2] = { 'item_id': agent5_ammo, 'mkt_mask': False }

    # agent 4: cannot buy its own item (invalid)
    test_cond[4] = { 'item_id': env.obs[4].inventory.id(0), 'mkt_mask': False }

    # agent 5: cannot buy agent 3's ammo (invalid: not enought gold)
    test_cond[5] = { 'item_id': env.obs[3].inventory.id(0), 'mkt_mask': False }

    actions = self._check_assert_make_action(env, action.Buy, test_cond)

    # agent 3: list an already listed item for sale (try different price)
    ent_id = 3; item_sig = self.item_sig[ent_id][0]
    actions[ent_id] = { action.Sell: {
        action.InventoryItem: env.obs[ent_id].inventory.sig(*item_sig),
        action.Price: action.Price.edges[7] } } # try to set different price

    env.step(actions)

    # Check the third tick actions
    # agent 1: buy agent 5's ammo (valid: 1 has the same ammo stack)
    #   agent 5's ammo should be gone
    seller_id = 5; buyer_id = 1
    self.assertFalse( agent5_ammo in env.obs[seller_id].inventory.ids)
    self.assertEqual( env.realm.players[seller_id].gold.val, # gold transfer
                      self.init_gold + listing_price[seller_id])
    self.assertEqual(2 * self.ammo_quantity, # ammo transfer
          ItemState.parse_array(env.obs[buyer_id].inventory.values[0]).quantity)
    self.assertEqual( env.realm.players[buyer_id].gold.val, # gold transfer
                      self.init_gold - listing_price[seller_id])

    # agent 2-4: invalid buy, no exchange, thus the same money
    for ent_id in [2, 3, 4]:
      self.assertEqual( env.realm.players[ent_id].gold.val, self.init_gold)

    # DONE


if __name__ == '__main__':
  unittest.main()
