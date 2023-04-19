from functools import lru_cache

import numpy as np

from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState
import nmmo.systems.item as item_system
from nmmo.io import action
from nmmo.lib import material, utils


class BasicObs:
  def __init__(self, values, id_col):
    self.values = values
    self.ids = values[:, id_col]

  @property
  def len(self):
    return len(self.ids)

  def id(self, i):
    return self.ids[i] if i < self.len else None

  def index(self, val):
    return np.nonzero(self.ids == val)[0][0] if val in self.ids else None


class InventoryObs(BasicObs):
  def __init__(self, values, id_col):
    super().__init__(values, id_col)
    self.inv_type = self.values[:,ItemState.State.attr_name_to_col["type_id"]]
    self.inv_level = self.values[:,ItemState.State.attr_name_to_col["level"]]

  def sig(self, item: item_system.Item, level: int):
    idx = np.nonzero((self.inv_type == item.ITEM_TYPE_ID) & (self.inv_level == level))[0]
    return idx[0] if len(idx) else None


class Observation:
  def __init__(self,
    config,
    agent_id: int,
    tiles,
    entities,
    inventory,
    market) -> None:

    self.config = config
    self.agent_id = agent_id

    self.tiles = tiles[0:config.MAP_N_OBS]
    self.entities = BasicObs(entities[0:config.PLAYER_N_OBS],
                              EntityState.State.attr_name_to_col["id"])

    if config.ITEM_SYSTEM_ENABLED:
      self.inventory = InventoryObs(inventory[0:config.INVENTORY_N_OBS],
                                    ItemState.State.attr_name_to_col["id"])
    else:
      assert inventory.size == 0

    if config.EXCHANGE_SYSTEM_ENABLED:
      self.market = BasicObs(market[0:config.MARKET_N_OBS],
                             ItemState.State.attr_name_to_col["id"])
    else:
      assert market.size == 0

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def tile(self, r_delta, c_delta):
    '''Return the array object corresponding to a nearby tile

    Args:
        r_delta: row offset from current agent
        c_delta: col offset from current agent

    Returns:
        Vector corresponding to the specified tile
    '''
    agent = self.agent()
    if (0 <= agent.row + r_delta < self.config.MAP_SIZE) & \
       (0 <= agent.col + c_delta < self.config.MAP_SIZE):
      r_cond = (self.tiles[:,TileState.State.attr_name_to_col["row"]] == agent.row + r_delta)
      c_cond = (self.tiles[:,TileState.State.attr_name_to_col["col"]] == agent.col + c_delta)
      return TileState.parse_array(self.tiles[r_cond & c_cond][0])

    # return a dummy lava tile at (inf, inf)
    return TileState.parse_array([np.inf, np.inf, material.Lava.index])

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def entity(self, entity_id):
    rows = self.entities.values[self.entities.ids == entity_id]
    if rows.size == 0:
      return None
    return EntityState.parse_array(rows[0])

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def agent(self):
    return self.entity(self.agent_id)

  def to_gym(self):
    '''Convert the observation to a format that can be used by OpenAI Gym'''

    gym_obs = {
      "Tile": np.vstack([
        self.tiles,
        np.zeros((self.config.MAP_N_OBS - self.tiles.shape[0], self.tiles.shape[1]))
      ]),
      "Entity": np.vstack([
        self.entities.values, np.zeros((
          self.config.PLAYER_N_OBS - self.entities.values.shape[0],
          self.entities.values.shape[1]))
      ]),
    }

    if self.config.ITEM_SYSTEM_ENABLED:
      gym_obs["Inventory"] = np.vstack([
        self.inventory.values, np.zeros((
          self.config.INVENTORY_N_OBS - self.inventory.values.shape[0],
          self.inventory.values.shape[1]))
      ])

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      gym_obs["Market"] = np.vstack([
        self.market.values, np.zeros((
          self.config.MARKET_N_OBS - self.market.values.shape[0],
          self.market.values.shape[1]))
      ])

    if self.config.PROVIDE_ACTION_TARGETS:
      gym_obs["ActionTargets"] = self._make_action_targets()

    return gym_obs

  def _make_action_targets(self):
    # TODO(kywch): return all-0 masks for buy/sell/give during combat

    masks = {}
    masks[action.Move] = {
      action.Direction: self._make_move_mask()
    }

    if self.config.COMBAT_SYSTEM_ENABLED:
      masks[action.Attack] = {
        action.Style: np.ones(len(action.Style.edges), dtype=np.int8),
        action.Target: self._make_attack_mask()
      }

    if self.config.ITEM_SYSTEM_ENABLED:
      masks[action.Use] = {
        action.InventoryItem: self._make_use_mask()
      }
      masks[action.Give] = {
        action.InventoryItem: self._make_sell_mask(),
        action.Target: self._make_give_target_mask()
      }
      masks[action.Destroy] = {
        action.InventoryItem: self._make_destroy_item_mask()
      }

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      masks[action.Sell] = {
        action.InventoryItem: self._make_sell_mask(),
        action.Price: np.ones(len(action.Price.edges), dtype=np.int8)
      }
      masks[action.Buy] = {
        action.MarketItem: self._make_buy_mask()
      }
      masks[action.GiveGold] = {
        action.Target: self._make_give_target_mask(),
        action.Price: self._make_give_gold_mask() # reusing Price
      }

    if self.config.COMMUNICATION_SYSTEM_ENABLED:
      masks[action.Comm] = {
        action.Token: np.ones(len(action.Token.edges), dtype=np.int8)
      }

    return masks

  def _make_move_mask(self):
    # pylint: disable=not-an-iterable
    return np.array(
      [self.tile(*d.delta).material_id in material.Habitable
       for d in action.Direction.edges], dtype=np.int8)

  def _make_attack_mask(self):
    # TODO: Currently, all attacks have the same range
    #   if we choose to make ranges different, the masks
    #   should be differently generated by attack styles
    assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_RANGE_REACH
    assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_MAGE_REACH
    assert self.config.COMBAT_RANGE_REACH == self.config.COMBAT_MAGE_REACH

    attack_range = self.config.COMBAT_MELEE_REACH

    agent = self.agent()
    entities_pos = self.entities.values[:, [EntityState.State.attr_name_to_col["row"],
                                            EntityState.State.attr_name_to_col["col"]]]
    within_range = utils.linf(entities_pos, (agent.row, agent.col)) <= attack_range

    immunity = self.config.COMBAT_SPAWN_IMMUNITY
    if 0 < immunity < agent.time_alive:
      # ids > 0 equals entity.is_player
      spawn_immunity = (self.entities.ids > 0) & \
        (self.entities.values[:,EntityState.State.attr_name_to_col["time_alive"]] < immunity)
    else:
      spawn_immunity = np.ones(self.entities.len, dtype=np.int8)

    # allow friendly fire but no self shooting
    not_me = np.ones(self.entities.len, dtype=np.int8)
    not_me[self.entities.index(agent.id)] = 0 # mask self

    return np.concatenate([within_range & not_me & spawn_immunity,
      np.zeros(self.config.PLAYER_N_OBS - self.entities.len, dtype=np.int8)])

  def _make_use_mask(self):
    # empty inventory -- nothing to use
    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0):
      return np.zeros(self.config.INVENTORY_N_OBS, dtype=np.int8)

    item_skill = self._item_skill()

    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0
    item_type = self.inventory.values[:,ItemState.State.attr_name_to_col["type_id"]]
    item_level = self.inventory.values[:,ItemState.State.attr_name_to_col["level"]]

    # level limits are differently applied depending on item types
    type_flt = np.tile ( np.array(list(item_skill.keys())), (self.inventory.len,1) )
    level_flt = np.tile ( np.array(list(item_skill.values())), (self.inventory.len,1) )
    item_type = np.tile( np.transpose(np.atleast_2d(item_type)), (1, len(item_skill)))
    item_level = np.tile( np.transpose(np.atleast_2d(item_level)), (1, len(item_skill)))
    level_satisfied = np.any((item_type == type_flt) & (item_level <= level_flt), axis=1)

    return np.concatenate([not_listed & level_satisfied,
      np.zeros(self.config.INVENTORY_N_OBS - self.inventory.len, dtype=np.int8)])

  def _item_skill(self):
    agent = self.agent()

    # the minimum agent level is 1
    level = max(1, agent.melee_level, agent.range_level, agent.mage_level,
                agent.fishing_level, agent.herbalism_level, agent.prospecting_level,
                agent.carving_level, agent.alchemy_level)
    return {
      item_system.Hat.ITEM_TYPE_ID: level,
      item_system.Top.ITEM_TYPE_ID: level,
      item_system.Bottom.ITEM_TYPE_ID: level,
      item_system.Sword.ITEM_TYPE_ID: agent.melee_level,
      item_system.Bow.ITEM_TYPE_ID: agent.range_level,
      item_system.Wand.ITEM_TYPE_ID: agent.mage_level,
      item_system.Rod.ITEM_TYPE_ID: agent.fishing_level,
      item_system.Gloves.ITEM_TYPE_ID: agent.herbalism_level,
      item_system.Pickaxe.ITEM_TYPE_ID: agent.prospecting_level,
      item_system.Chisel.ITEM_TYPE_ID: agent.carving_level,
      item_system.Arcane.ITEM_TYPE_ID: agent.alchemy_level,
      item_system.Scrap.ITEM_TYPE_ID: agent.melee_level,
      item_system.Shaving.ITEM_TYPE_ID: agent.range_level,
      item_system.Shard.ITEM_TYPE_ID: agent.mage_level,
      item_system.Ration.ITEM_TYPE_ID: level,
      item_system.Poultice.ITEM_TYPE_ID: level
    }

  def _make_destroy_item_mask(self):
    # empty inventory -- nothing to destroy
    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0):
      return np.zeros(self.config.INVENTORY_N_OBS, dtype=np.int8)

    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0

    # not equipped items in the inventory can be destroyed
    return np.concatenate([not_equipped,
      np.zeros(self.config.INVENTORY_N_OBS - self.inventory.len, dtype=np.int8)])

  def _make_give_target_mask(self):
    # empty inventory -- nothing to give
    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0):
      return np.zeros(self.config.PLAYER_N_OBS, dtype=np.int8)

    agent = self.agent()
    entities_pos = self.entities.values[:, [EntityState.State.attr_name_to_col["row"],
                                            EntityState.State.attr_name_to_col["col"]]]
    same_tile = utils.linf(entities_pos, (agent.row, agent.col)) == 0
    not_me = (self.entities.ids != self.agent_id)
    player = (self.entities.values[:,EntityState.State.attr_name_to_col["npc_type"]] == 0)

    return np.concatenate([
      (same_tile & player & not_me),
      np.zeros(self.config.PLAYER_N_OBS - self.entities.len, dtype=np.int8)])

  def _make_give_gold_mask(self):
    gold = int(self.agent().gold)
    mask = np.zeros(self.config.PRICE_N_OBS, dtype=np.int8)

    if gold:
      mask[:gold] = 1 # NOTE that action.Price starts from Discrete_1

    return mask

  def _make_sell_mask(self):
    # empty inventory -- nothing to sell
    if not (self.config.EXCHANGE_SYSTEM_ENABLED and self.inventory.len > 0):
      return np.zeros(self.config.INVENTORY_N_OBS, dtype=np.int8)

    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0
    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0

    return np.concatenate([not_equipped & not_listed,
      np.zeros(self.config.INVENTORY_N_OBS - self.inventory.len, dtype=np.int8)])

  def _make_buy_mask(self):
    if not self.config.EXCHANGE_SYSTEM_ENABLED:
      return np.zeros(self.config.MARKET_N_OBS, dtype=np.int8)

    market_flt = np.ones(self.market.len, dtype=np.int8)
    full_inventory = self.inventory.len >= self.config.ITEM_INVENTORY_CAPACITY

    # if the inventory is full, one can only buy existing ammo stack
    #   otherwise, one can buy anything owned by other, having enough money
    if full_inventory:
      exist_ammo_listings = self._existing_ammo_listings()
      if not np.any(exist_ammo_listings):
        return np.zeros(self.config.MARKET_N_OBS, dtype=np.int8)
      market_flt = exist_ammo_listings

    agent = self.agent()
    market_items = self.market.values
    enough_gold = market_items[:,ItemState.State.attr_name_to_col["listed_price"]] <= agent.gold
    not_mine = market_items[:,ItemState.State.attr_name_to_col["owner_id"]] != self.agent_id

    return np.concatenate([market_flt & enough_gold & not_mine,
      np.zeros(self.config.MARKET_N_OBS - self.market.len, dtype=np.int8)])

  def _existing_ammo_listings(self):
    sig_col = (ItemState.State.attr_name_to_col["type_id"],
               ItemState.State.attr_name_to_col["level"])
    ammo_id = [ammo.ITEM_TYPE_ID for ammo in
              [item_system.Scrap, item_system.Shaving, item_system.Shard]]

    # search ammo stack from the inventory
    type_flt = np.tile( np.array(ammo_id), (self.inventory.len,1))
    item_type = np.tile(
      np.transpose(np.atleast_2d(self.inventory.values[:,sig_col[0]])),
      (1, len(ammo_id)))
    exist_ammo = self.inventory.values[np.any(item_type == type_flt, axis=1)]

    # self does not have ammo
    if exist_ammo.shape[0] == 0:
      return np.zeros(self.market.len, dtype=np.int8)

    # search the existing ammo stack from the market that's not mine
    type_flt = np.tile( np.array(exist_ammo[:,sig_col[0]]), (self.market.len,1))
    level_flt = np.tile( np.array(exist_ammo[:,sig_col[1]]), (self.market.len,1))
    item_type = np.tile( np.transpose(np.atleast_2d(self.market.values[:,sig_col[0]])),
      (1, exist_ammo.shape[0]))
    item_level = np.tile( np.transpose(np.atleast_2d(self.market.values[:,sig_col[1]])),
      (1, exist_ammo.shape[0]))
    exist_ammo_listings = np.any((item_type == type_flt) & (item_level == level_flt), axis=1)

    not_mine = self.market.values[:,ItemState.State.attr_name_to_col["owner_id"]] != self.agent_id

    return exist_ammo_listings & not_mine
