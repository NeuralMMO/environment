# pylint: disable=no-member,c-extension-no-member
from functools import lru_cache
from copy import deepcopy
from types import MappingProxyType

import numpy as np

from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState
import nmmo.systems.item as item_system
from nmmo.core import action
from nmmo.lib import material, utils
import nmmo.lib.cython_helper as chp

ROW_DELTA = np.array([-1, 1, 0, 0], dtype=np.int64)
COL_DELTA = np.array([0, 0, 1, -1], dtype=np.int64)
EMPTY_TILE = TileState.parse_array(
  np.array([0, 0, material.Void.index], dtype=np.int16))


class BasicObs:
  def __init__(self, id_col, obs_dim):
    self.values = None
    self.ids = None
    self.id_col = id_col
    self.obs_dim = obs_dim

  def reset(self):
    self.values = None
    self.ids = None

  def update(self, values):
    self.values = values[:self.obs_dim]
    self.ids = values[:, self.id_col]

  @property
  def len(self):
    return len(self.ids)

  def id(self, i):
    return self.ids[i] if i < self.len else None

  def index(self, val):
    return np.nonzero(self.ids == val)[0][0] if val in self.ids else None

class InventoryObs(BasicObs):
  def __init__(self, id_col, obs_dim):
    super().__init__(id_col, obs_dim)
    self.inv_type = None
    self.inv_level = None

  def update(self, values):
    super().update(values)
    self.inv_type = self.values[:,ItemState.State.attr_name_to_col["type_id"]]
    self.inv_level = self.values[:,ItemState.State.attr_name_to_col["level"]]

  def sig(self, item: item_system.Item, level: int):
    idx = np.nonzero((self.inv_type == item.ITEM_TYPE_ID) & (self.inv_level == level))[0]
    return idx[0] if len(idx) else None

class GymObs:
  keys_to_clear = ["Tile", "Entity", "Inventory", "Market", "Communication"]

  def __init__(self, config, agent_id):
    self.config = config
    self.agent_id = agent_id
    self.values = self._make_empty_obs()

  def reset(self, task_embedding=None):
    self.clear()
    self.values["Task"][:] = 0 if task_embedding is None else task_embedding

  def clear(self, tick=None):
    self.values["CurrentTick"] = tick or 0
    for key in self.keys_to_clear:
      if key in self.values:
        if key == "Inventory" and not self.config.ITEM_SYSTEM_ENABLED:
          continue
        if key == "Market" and not self.config.EXCHANGE_SYSTEM_ENABLED:
          continue
        if key == "Communication" and not self.config.COMMUNICATION_SYSTEM_ENABLED:
          continue
        self.values[key][:] = 0

  def _make_empty_obs(self):
    num_tile_attributes = TileState.State.num_attributes
    num_tile_attributes += 1 if self.config.original["PROVIDE_DEATH_FOG_OBS"] else 0
    gym_obs = {
      "CurrentTick": 0,
      "AgentId": self.agent_id,
      "Task": np.zeros(self.config.TASK_EMBED_DIM, dtype=np.float16),
      "Tile": np.zeros((self.config.MAP_N_OBS, num_tile_attributes), dtype=np.int16),
      "Entity": np.zeros((self.config.PLAYER_N_OBS,
                          EntityState.State.num_attributes), dtype=np.int16)}
    if self.config.original["ITEM_SYSTEM_ENABLED"]:
      gym_obs["Inventory"] = np.zeros((self.config.INVENTORY_N_OBS,
                                       ItemState.State.num_attributes), dtype=np.int16)
    if self.config.original["EXCHANGE_SYSTEM_ENABLED"]:
      gym_obs["Market"] = np.zeros((self.config.MARKET_N_OBS,
                                    ItemState.State.num_attributes), dtype=np.int16)
    if self.config.original["COMMUNICATION_SYSTEM_ENABLED"]:
      gym_obs["Communication"] = np.zeros((self.config.COMMUNICATION_N_OBS,
                                           len(EntityState.State.comm_attr_map)),
                                           dtype=np.int16)
    return gym_obs

  def set_arr_values(self, key, values):
    obs_shape = self.values[key].shape
    self.values[key][:values.shape[0], :] = values[:, :obs_shape[1]]

class ActionTargets:
  no_op_keys = ["Direction", "Target", "InventoryItem", "MarketItem"]
  all_ones = ["Style", "Price", "Token"]

  def __init__(self, config):
    self.config = config
    if not self.config.original["PROVIDE_ACTION_TARGETS"]:
      return

    self._no_op = 1 if config.original["PROVIDE_NOOP_ACTION_TARGET"] else 0
    self.values = self._make_empty_targets()
    self.keys_to_clear = None
    self.clear(reset=True)  # to set the no-op option to 1, if needed

  def _get_keys_to_clear(self):
    keys = []
    if self.config.COMBAT_SYSTEM_ENABLED:
      keys.append("Attack")
    if self.config.ITEM_SYSTEM_ENABLED:
      keys.extend(["Use", "Give", "Destroy"])
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      keys.extend(["Sell", "Buy", "GiveGold"])
    if self.config.COMMUNICATION_SYSTEM_ENABLED:
      keys.append("Comm")
    return keys

  def reset(self):
    if not self.config.original["PROVIDE_ACTION_TARGETS"]:
      return
    self.keys_to_clear = self._get_keys_to_clear()
    self.clear(reset=True)

  def clear(self, reset=False):
    if not self.config.original["PROVIDE_ACTION_TARGETS"]:
      return
    for key, mask in self.values.items():
      if reset is True or key in self.keys_to_clear:
        for sub_key in mask:
          mask[sub_key][:] = 1 if sub_key in self.all_ones else 0
          if self._no_op > 0 and sub_key in self.no_op_keys:
            mask[sub_key][-1] = 1  # set the no-op option to 1

  def _make_empty_targets(self):
    masks = {}
    masks["Move"] = {"Direction": np.zeros(len(action.Direction.edges), dtype=np.int8)}
    if self.config.original["COMBAT_SYSTEM_ENABLED"]:
      masks["Attack"] = {
        "Style": np.ones(len(action.Style.edges), dtype=np.int8),
        "Target": np.zeros(self.config.PLAYER_N_OBS + self._no_op, dtype=np.int8)}
    if self.config.original["ITEM_SYSTEM_ENABLED"]:
      masks["Use"] = {
        "InventoryItem": np.zeros(self.config.INVENTORY_N_OBS + self._no_op, dtype=np.int8)}
      masks["Give"] = {
        "InventoryItem": np.zeros(self.config.INVENTORY_N_OBS + self._no_op, dtype=np.int8),
        "Target": np.zeros(self.config.PLAYER_N_OBS + self._no_op, dtype=np.int8)}
      masks["Destroy"] = {
        "InventoryItem": np.zeros(self.config.INVENTORY_N_OBS + self._no_op, dtype=np.int8)}
    if self.config.original["EXCHANGE_SYSTEM_ENABLED"]:
      masks["Sell"] = {
        "InventoryItem": np.zeros(self.config.INVENTORY_N_OBS + self._no_op, dtype=np.int8),
        "Price": np.ones(self.config.PRICE_N_OBS, dtype=np.int8)}
      masks["Buy"] = {
        "MarketItem": np.zeros(self.config.MARKET_N_OBS + self._no_op, dtype=np.int8)}
      masks["GiveGold"] = {
        "Price": np.ones(self.config.PRICE_N_OBS, dtype=np.int8),
        "Target": np.zeros(self.config.PLAYER_N_OBS + self._no_op, dtype=np.int8)}
    if self.config.original["COMMUNICATION_SYSTEM_ENABLED"]:
      masks["Comm"] = {"Token": np.ones(self.config.COMMUNICATION_NUM_TOKENS, dtype=np.int8)}
    return masks

class Observation:
  def __init__(self, config, agent_id: int) -> None:
    self.config = config
    self.agent_id = agent_id
    self.agent = None

    self.current_tick = None
    self._is_agent_dead = None
    self.habitable_tiles = None
    self.agent_in_combat = None
    self.gym_obs = GymObs(config, agent_id)
    self.empty_obs = deepcopy(self.gym_obs.values)
    self.action_targets = ActionTargets(config)
    if self.config.original["PROVIDE_ACTION_TARGETS"]:
      self.empty_obs["ActionTargets"] = deepcopy(self.action_targets.values)

    self.vision_radius = self.config.PLAYER_VISION_RADIUS
    self.vision_diameter = self.config.PLAYER_VISION_DIAMETER
    self._noop_action = 1 if config.original["PROVIDE_NOOP_ACTION_TARGET"] else 0
    self.tiles = None
    self.entities = BasicObs(EntityState.State.attr_name_to_col["id"],
                             config.PLAYER_N_OBS)
    self.inventory = InventoryObs(ItemState.State.attr_name_to_col["id"],
                                  config.INVENTORY_N_OBS) \
      if config.original["ITEM_SYSTEM_ENABLED"] else None
    self.market = BasicObs(ItemState.State.attr_name_to_col["id"],
                           config.MARKET_N_OBS) \
      if config.original["EXCHANGE_SYSTEM_ENABLED"] else None
    self.comm = BasicObs(EntityState.State.attr_name_to_col["id"],
                         config.COMMUNICATION_N_OBS) \
      if config.original["COMMUNICATION_SYSTEM_ENABLED"] else None

  def reset(self, habitable_tiles, task_embedding=None):
    self.gym_obs.reset(task_embedding)
    self.action_targets.reset()
    self.habitable_tiles = habitable_tiles
    self._is_agent_dead = False
    self.agent_in_combat = None

    self.current_tick = 0
    self.tiles = None
    self.entities.reset()
    if self.config.ITEM_SYSTEM_ENABLED:
      self.inventory.reset()
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      self.market.reset()
    if self.config.COMMUNICATION_SYSTEM_ENABLED:
      self.comm.reset()
    return self

  @property
  def return_dummy_obs(self):
    return self._is_agent_dead

  def set_agent_dead(self):
    self._is_agent_dead = True

  def update(self, tick, visible_tiles, visible_entities,
             inventory=None, market=None, comm=None):
    if self._is_agent_dead:
      return

    # cache has previous tick's data, so clear it
    self.clear_cache()

    # update the obs
    self.current_tick = tick
    self.tiles = visible_tiles  # assert len(visible_tiles) == self.config.MAP_N_OBS
    self.entities.update(visible_entities)
    if self.config.ITEM_SYSTEM_ENABLED:
      assert inventory is not None, "Inventory must be provided if ITEM_SYSTEM_ENABLED"
      self.inventory.update(inventory)
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      assert market is not None, "Market must be provided if EXCHANGE_SYSTEM_ENABLED"
      self.market.update(market)
    if self.config.COMMUNICATION_SYSTEM_ENABLED:
      assert comm is not None, "Comm must be provided if COMMUNICATION_SYSTEM_ENABLED"
      self.comm.update(comm)

    # update helper vars
    self.agent = self.entity(self.agent_id)
    if self.config.COMBAT_SYSTEM_ENABLED:
      latest_combat_tick = self.agent.latest_combat_tick
      self.agent_in_combat = False if latest_combat_tick == 0 else \
        (tick - latest_combat_tick) < self.config.COMBAT_STATUS_DURATION
    else:
      self.agent_in_combat = False

  @lru_cache
  def tile(self, r_delta, c_delta):
    '''Return the array object corresponding to a nearby tile

    Args:
        r_delta: row offset from current agent
        c_delta: col offset from current agent

    Returns:
        Vector corresponding to the specified tile
    '''
    idx_1d = (self.vision_radius+r_delta)*self.vision_diameter + self.vision_radius+c_delta
    try:
      return TileState.parse_array(self.tiles[idx_1d])
    except IndexError:
      return EMPTY_TILE

  @lru_cache
  def entity(self, entity_id):
    rows = self.entities.values[self.entities.ids == entity_id]
    if rows.shape[0] == 0:
      return None
    return EntityState.parse_array(rows[0])

  def clear_cache(self):
    # clear the outdated cache
    self.entity.cache_clear()
    self.tile.cache_clear()

  def to_gym(self):
    '''Convert the observation to a format that can be used by OpenAI Gym'''
    if self.return_dummy_obs:
      return MappingProxyType(self.empty_obs)  # NOTE: read-only
    self.gym_obs.clear(self.current_tick)
    # NOTE: assume that all len(self.tiles) == self.config.MAP_N_OBS
    self.gym_obs.set_arr_values('Tile', self.tiles)
    self.gym_obs.set_arr_values('Entity', self.entities.values)
    if self.config.ITEM_SYSTEM_ENABLED:
      self.gym_obs.set_arr_values('Inventory', self.inventory.values)
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      self.gym_obs.set_arr_values('Market', self.market.values)
    if self.config.COMMUNICATION_SYSTEM_ENABLED:
      self.gym_obs.set_arr_values('Communication', self.comm.values)
    if self.config.PROVIDE_ACTION_TARGETS:
      self.gym_obs.values["ActionTargets"] = self._make_action_targets()

    # Making the gym obs read-only
    return MappingProxyType(self.gym_obs.values)

  def _make_action_targets(self):
    self.action_targets.clear()
    masks = self.action_targets.values
    self._make_move_mask(masks["Move"])
    if self.config.COMBAT_SYSTEM_ENABLED:
      # Test below. see tests/core/test_observation_tile.py, test_action_target_consts()
      # assert len(action.Style.edges) == 3
      self._make_attack_mask(masks["Attack"])
    if self.config.ITEM_SYSTEM_ENABLED:
      self._make_use_mask(masks["Use"])
      self._make_destroy_item_mask(masks["Destroy"])
      self._make_give_mask(masks["Give"])
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      self._make_sell_mask(masks["Sell"])
      self._make_give_gold_mask(masks["GiveGold"])
      self._make_buy_mask(masks["Buy"])
    return masks

  def _make_move_mask(self, move_mask, use_cython=None):
    use_cython = use_cython or self.config.USE_CYTHON
    if use_cython:
      chp.make_move_mask(move_mask["Direction"], self.habitable_tiles,
                         self.agent.row, self.agent.col, ROW_DELTA, COL_DELTA)
      return
    move_mask["Direction"][:4] = self.habitable_tiles[self.agent.row+ROW_DELTA,
                                                      self.agent.col+COL_DELTA]

  def _make_attack_mask(self, attack_mask, use_cython=None):
    if self.config.COMBAT_ALLOW_FLEXIBLE_STYLE:
      # NOTE: if the style is flexible, then the reach of all styles should be the same
      assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_RANGE_REACH
      assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_MAGE_REACH
      assert self.config.COMBAT_RANGE_REACH == self.config.COMBAT_MAGE_REACH

    if not self.config.COMBAT_SYSTEM_ENABLED or self.return_dummy_obs:
      return

    use_cython = use_cython or self.config.USE_CYTHON
    if use_cython:
      chp.make_attack_mask(
        attack_mask["Target"], self.entities.values, EntityState.State.attr_name_to_col,
        {"agent_id": self.agent_id, "row": self.agent.row, "col": self.agent.col,
         "immunity": self.config.COMBAT_SPAWN_IMMUNITY,
         "attack_range": self.config.COMBAT_RANGE_REACH})
      return

    # allow friendly fire but no self shooting
    targetable = self.entities.ids != self.agent.id

    # NOTE: this is a hack. Only target "normal" agents, which has npc_type of 0, 1, 2, 3
    # For example, immortal "scout" agents has npc_type of -1
    targetable &= self.entities.values[:,EntityState.State.attr_name_to_col["npc_type"]] >= 0

    immunity = self.config.COMBAT_SPAWN_IMMUNITY
    if self.agent.time_alive < immunity:
      # NOTE: CANNOT attack players during immunity, thus mask should set to 0
      targetable &= ~(self.entities.ids > 0)  # ids > 0 equals entity.is_player

    within_range = np.maximum( # calculating the l-inf dist
        np.abs(self.entities.values[:,EntityState.State.attr_name_to_col["row"]] - self.agent.row),
        np.abs(self.entities.values[:,EntityState.State.attr_name_to_col["col"]] - self.agent.col)
      ) <= self.config.COMBAT_MELEE_REACH

    attack_mask["Target"][:self.entities.len] = targetable & within_range
    if sum(attack_mask["Target"][:self.entities.len]) > 0:
      # Mask the no-op option, since there should be at least one allowed move
      # NOTE: this will make agents always attack if there is a valid target
      attack_mask["Target"][-1] = 0

  def _make_use_mask(self, use_mask):
    # empty inventory -- nothing to use
    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0)\
        or self.return_dummy_obs or self.agent_in_combat:
      return

    item_skill = self._item_skill()
    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0
    item_type = self.inventory.values[:,ItemState.State.attr_name_to_col["type_id"]]
    item_level = self.inventory.values[:,ItemState.State.attr_name_to_col["level"]]

    # level limits are differently applied depending on item types
    type_flt = np.tile(np.array(list(item_skill.keys())), (self.inventory.len,1))
    level_flt = np.tile(np.array(list(item_skill.values())), (self.inventory.len,1))
    item_type = np.tile(np.transpose(np.atleast_2d(item_type)), (1,len(item_skill)))
    item_level = np.tile(np.transpose(np.atleast_2d(item_level)), (1,len(item_skill)))
    level_satisfied = np.any((item_type==type_flt) & (item_level<=level_flt), axis=1)
    use_mask["InventoryItem"][:self.inventory.len] = not_listed & level_satisfied

  def _item_skill(self):
    agent = self.agent

    # the minimum agent level is 1
    level = max(1, agent.melee_level, agent.range_level, agent.mage_level,
                agent.fishing_level, agent.herbalism_level, agent.prospecting_level,
                agent.carving_level, agent.alchemy_level)
    return {
      item_system.Hat.ITEM_TYPE_ID: level,
      item_system.Top.ITEM_TYPE_ID: level,
      item_system.Bottom.ITEM_TYPE_ID: level,
      item_system.Spear.ITEM_TYPE_ID: agent.melee_level,
      item_system.Bow.ITEM_TYPE_ID: agent.range_level,
      item_system.Wand.ITEM_TYPE_ID: agent.mage_level,
      item_system.Rod.ITEM_TYPE_ID: agent.fishing_level,
      item_system.Gloves.ITEM_TYPE_ID: agent.herbalism_level,
      item_system.Pickaxe.ITEM_TYPE_ID: agent.prospecting_level,
      item_system.Axe.ITEM_TYPE_ID: agent.carving_level,
      item_system.Chisel.ITEM_TYPE_ID: agent.alchemy_level,
      item_system.Whetstone.ITEM_TYPE_ID: agent.melee_level,
      item_system.Arrow.ITEM_TYPE_ID: agent.range_level,
      item_system.Runes.ITEM_TYPE_ID: agent.mage_level,
      item_system.Ration.ITEM_TYPE_ID: level,
      item_system.Potion.ITEM_TYPE_ID: level
    }

  def _make_destroy_item_mask(self, destroy_mask):
    # empty inventory -- nothing to destroy
    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0)\
        or self.return_dummy_obs or self.agent_in_combat:
      return
    # not equipped items in the inventory can be destroyed
    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0
    destroy_mask["InventoryItem"][:self.inventory.len] = not_equipped

  def _make_give_mask(self, give_mask):
    if not self.config.ITEM_SYSTEM_ENABLED or self.return_dummy_obs or self.agent_in_combat\
       or self.inventory.len == 0:
      return

    # InventoryItem
    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0
    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0
    give_mask["InventoryItem"][:self.inventory.len] = not_equipped & not_listed

    # Give Target
    entities_pos = self.entities.values[:,[EntityState.State.attr_name_to_col["row"],
                                           EntityState.State.attr_name_to_col["col"]]]
    same_tile = utils.linf(entities_pos, (self.agent.row, self.agent.col)) == 0
    not_me = self.entities.ids != self.agent_id
    player = (self.entities.values[:,EntityState.State.attr_name_to_col["npc_type"]] == 0)
    give_mask["Target"][:self.entities.len] = same_tile & player & not_me

  def _make_sell_mask(self, sell_mask):
    # empty inventory -- nothing to sell
    if not (self.config.EXCHANGE_SYSTEM_ENABLED and self.inventory.len > 0) \
      or self.return_dummy_obs or self.agent_in_combat:
      return

    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0
    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0
    sell_mask["InventoryItem"][:self.inventory.len] = not_equipped & not_listed

  def _make_give_gold_mask(self, give_mask):
    if not self.config.EXCHANGE_SYSTEM_ENABLED or self.return_dummy_obs or self.agent_in_combat\
       or int(self.agent.gold) <= 2:  # NOTE: this is a hack to reduce mask computation
      return

    # GiveGold Target
    entities_pos = self.entities.values[:,[EntityState.State.attr_name_to_col["row"],
                                           EntityState.State.attr_name_to_col["col"]]]
    same_tile = utils.linf(entities_pos, (self.agent.row, self.agent.col)) == 0
    not_me = self.entities.ids != self.agent_id
    player = (self.entities.values[:,EntityState.State.attr_name_to_col["npc_type"]] == 0)
    give_mask["Target"][:self.entities.len] = same_tile & player & not_me

    # GiveGold Amount (Price)
    gold = int(self.agent.gold)
    give_mask["Price"][gold:] = 0 # NOTE: Price masks starts with all ones

  def _make_buy_mask(self, buy_mask):
    if not self.config.EXCHANGE_SYSTEM_ENABLED or self.return_dummy_obs or self.agent_in_combat \
       or self.market.len == 0:
      return

    market_items = self.market.values
    not_mine = market_items[:,ItemState.State.attr_name_to_col["owner_id"]] != self.agent_id
    # if the inventory is full, one can only buy existing ammo stack
    #   otherwise, one can buy anything owned by other, having enough money
    if self.inventory.len >= self.config.ITEM_INVENTORY_CAPACITY:
      exist_ammo_listings = self._existing_ammo_listings()
      if not np.any(exist_ammo_listings):
        return
      not_mine &= exist_ammo_listings

    enough_gold = market_items[:,ItemState.State.attr_name_to_col["listed_price"]] \
                    <= self.agent.gold
    buy_mask["MarketItem"][:self.market.len] = not_mine & enough_gold

  def _existing_ammo_listings(self):
    sig_col = (ItemState.State.attr_name_to_col["type_id"],
               ItemState.State.attr_name_to_col["level"])
    ammo_id = [ammo.ITEM_TYPE_ID for ammo in
              [item_system.Whetstone, item_system.Arrow, item_system.Runes]]

    # search ammo stack from the inventory
    type_flt = np.tile(np.array(ammo_id), (self.inventory.len,1))
    item_type = np.tile(
      np.transpose(np.atleast_2d(self.inventory.values[:,sig_col[0]])),
      (1, len(ammo_id)))
    exist_ammo = self.inventory.values[np.any(item_type == type_flt, axis=1)]

    # self does not have ammo
    if exist_ammo.shape[0] == 0:
      return np.zeros(self.market.len, dtype=bool)

    # search the existing ammo stack from the market that's not mine
    type_flt = np.tile(np.array(exist_ammo[:,sig_col[0]]), (self.market.len,1))
    level_flt = np.tile(np.array(exist_ammo[:,sig_col[1]]), (self.market.len,1))
    item_type = np.tile(np.transpose(np.atleast_2d(self.market.values[:,sig_col[0]])),
                        (1, exist_ammo.shape[0]))
    item_level = np.tile(np.transpose(np.atleast_2d(self.market.values[:,sig_col[1]])),
                         (1, exist_ammo.shape[0]))
    exist_ammo_listings = np.any((item_type==type_flt) & (item_level==level_flt), axis=1)

    not_mine = self.market.values[:,ItemState.State.attr_name_to_col["owner_id"]] != self.agent_id

    return exist_ammo_listings & not_mine
