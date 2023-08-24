from functools import lru_cache, cached_property

import numpy as np

from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState
import nmmo.systems.item as item_system
from nmmo.core import action
from nmmo.lib import material, utils


class BasicObs:
  def __init__(self, values, id_col):
    self.values = values
    self.ids = values[:, id_col]

  @cached_property
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
    current_tick: int,
    agent_id: int,
    task_embedding,
    tiles,
    entities,
    inventory,
    market) -> None:

    self.config = config
    self.current_tick = current_tick
    self.agent_id = agent_id
    self.task_embedding = task_embedding

    self.tiles = tiles[0:config.MAP_N_OBS]
    self.entities = BasicObs(entities[0:config.PLAYER_N_OBS],
                              EntityState.State.attr_name_to_col["id"])

    self.dummy_obs = self.agent() is None
    if config.COMBAT_SYSTEM_ENABLED and not self.dummy_obs:
      latest_combat_tick = self.agent().latest_combat_tick
      self.agent_in_combat = False if latest_combat_tick == 0 else \
        (current_tick - latest_combat_tick) < config.COMBAT_STATUS_DURATION
    else:
      self.agent_in_combat = False

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

    self._noop_action = 1 if config.PROVIDE_NOOP_ACTION_TARGET else 0

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
    center = self.config.PLAYER_VISION_RADIUS
    tile_dim = self.config.PLAYER_VISION_DIAMETER
    mat_map = self.tiles[:,2].reshape(tile_dim,tile_dim)
    new_row = agent.row + r_delta
    new_col = agent.col + c_delta
    if (0 <= new_row < self.config.MAP_SIZE) & \
       (0 <= new_col < self.config.MAP_SIZE):
      return TileState.parse_array([new_row, new_col, mat_map[center+r_delta,center+c_delta]])

    # return a dummy void tile at (inf, inf)
    return TileState.parse_array([np.inf, np.inf, material.Void.index])

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def entity(self, entity_id):
    rows = self.entities.values[self.entities.ids == entity_id]
    if rows.shape[0] == 0:
      return None
    return EntityState.parse_array(rows[0])

  # pylint: disable=method-cache-max-size-none
  @lru_cache(maxsize=None)
  def agent(self):
    return self.entity(self.agent_id)

  def clear_cache(self):
    # clear the cache, so that this object can be garbage collected
    self.agent.cache_clear()
    self.entity.cache_clear()
    self.tile.cache_clear()

  def get_empty_obs(self):
    gym_obs = {
      "CurrentTick": self.current_tick,
      "AgentId": self.agent_id,
      "Task": self.task_embedding,
      "Tile": None, # np.zeros((self.config.MAP_N_OBS, self.tiles.shape[1])),
      "Entity": np.zeros((self.config.PLAYER_N_OBS,
                          self.entities.values.shape[1]), dtype=np.int16)}
    if self.config.ITEM_SYSTEM_ENABLED:
      gym_obs["Inventory"] = np.zeros((self.config.INVENTORY_N_OBS,
                                       self.inventory.values.shape[1]), dtype=np.int16)
    if self.config.EXCHANGE_SYSTEM_ENABLED:
      gym_obs["Market"] = np.zeros((self.config.MARKET_N_OBS,
                                    self.market.values.shape[1]), dtype=np.int16)
    return gym_obs

  def to_gym(self):
    '''Convert the observation to a format that can be used by OpenAI Gym'''
    gym_obs = self.get_empty_obs()
    if self.dummy_obs:
      # return empty obs for the dead agents
      gym_obs['Tile'] = np.zeros((self.config.MAP_N_OBS, self.tiles.shape[1]), dtype=np.int16)
      if self.config.PROVIDE_ACTION_TARGETS:
        gym_obs["ActionTargets"] = self._make_action_targets()
      return gym_obs

    # NOTE: assume that all len(self.tiles) == self.config.MAP_N_OBS
    gym_obs['Tile'] = self.tiles
    gym_obs['Entity'][:self.entities.values.shape[0],:] = self.entities.values

    if self.config.ITEM_SYSTEM_ENABLED:
      gym_obs["Inventory"][:self.inventory.values.shape[0],:] = self.inventory.values

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      gym_obs["Market"][:self.market.values.shape[0],:] = self.market.values

    if self.config.PROVIDE_ACTION_TARGETS:
      gym_obs["ActionTargets"] = self._make_action_targets()

    return gym_obs

  def _make_action_targets(self):
    masks = {}
    masks["Move"] = {
      "Direction": self._make_move_mask()
    }

    if self.config.COMBAT_SYSTEM_ENABLED:
      # Test below. see tests/core/test_observation_tile.py, test_action_target_consts()
      # assert len(action.Style.edges) == 3
      masks["Attack"] = {
        "Style": np.zeros(3, dtype=np.int8) if self.dummy_obs\
                  else np.ones(3, dtype=np.int8),
        "Target": self._make_attack_mask()
      }

    if self.config.ITEM_SYSTEM_ENABLED:
      masks["Use"] = {
        "InventoryItem": self._make_use_mask()
      }
      masks["Give"] = {
        "InventoryItem": self._make_sell_mask(),
        "Target": self._make_give_target_mask()
      }
      masks["Destroy"] = {
        "InventoryItem": self._make_destroy_item_mask()
      }

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      masks["Sell"] = {
        "InventoryItem": self._make_sell_mask(),
        "Price": np.zeros(self.config.PRICE_N_OBS, dtype=np.int8) if self.dummy_obs\
                  else np.ones(self.config.PRICE_N_OBS, dtype=np.int8)
      }
      masks["Buy"] = {
        "MarketItem": self._make_buy_mask()
      }
      masks["GiveGold"] = {
        "Target": self._make_give_target_mask(),
        "Price": self._make_give_gold_mask() # reusing Price
      }

    if self.config.COMMUNICATION_SYSTEM_ENABLED:
      masks["Comm"] = {
        "Token":\
          np.zeros(self.config.COMMUNICATION_NUM_TOKENS, dtype=np.int8) if self.dummy_obs\
            else np.ones(self.config.COMMUNICATION_NUM_TOKENS, dtype=np.int8)
      }

    return masks

  def _make_move_mask(self):
    if self.dummy_obs:
      return np.zeros(len(action.Direction.edges), dtype=np.int8)
    # pylint: disable=not-an-iterable
    return np.array([self.tile(*d.delta).material_id in material.Habitable.indices
                     for d in action.Direction.edges], dtype=np.int8)

  def _make_attack_mask(self):
    # NOTE: Currently, all attacks have the same range
    #   if we choose to make ranges different, the masks
    #   should be differently generated by attack styles
    assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_RANGE_REACH
    assert self.config.COMBAT_MELEE_REACH == self.config.COMBAT_MAGE_REACH
    assert self.config.COMBAT_RANGE_REACH == self.config.COMBAT_MAGE_REACH

    attack_mask = np.zeros(self.config.PLAYER_N_OBS + self._noop_action, dtype=np.int8)
    if self.config.PROVIDE_NOOP_ACTION_TARGET:
      attack_mask[-1] = 1

    if self.dummy_obs:
      return attack_mask

    agent = self.agent()
    within_range = np.maximum( # calculating the l-inf dist
        np.abs(self.entities.values[:,EntityState.State.attr_name_to_col["row"]] - agent.row),
        np.abs(self.entities.values[:,EntityState.State.attr_name_to_col["col"]] - agent.col)
      ) <= self.config.COMBAT_MELEE_REACH

    immunity = self.config.COMBAT_SPAWN_IMMUNITY
    if agent.time_alive < immunity:
      # NOTE: CANNOT attack players during immunity, thus mask should set to 0
      no_spawn_immunity = ~(self.entities.ids > 0)  # ids > 0 equals entity.is_player
    else:
      no_spawn_immunity = np.ones(self.entities.len, dtype=bool)

    # allow friendly fire but no self shooting
    not_me = self.entities.ids != agent.id

    attack_mask[:self.entities.len] = within_range & not_me & no_spawn_immunity
    return attack_mask

  def _make_use_mask(self):
    # empty inventory -- nothing to use
    use_mask = np.zeros(self.config.INVENTORY_N_OBS + self._noop_action, dtype=np.int8)
    if self.config.PROVIDE_NOOP_ACTION_TARGET:
      use_mask[-1] = 1

    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0)\
        or self.dummy_obs or self.agent_in_combat:
      return use_mask

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

    use_mask[:self.inventory.len] = not_listed & level_satisfied
    return use_mask

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

  def _make_destroy_item_mask(self):
    destroy_mask = np.zeros(self.config.INVENTORY_N_OBS + self._noop_action, dtype=np.int8)
    if self.config.PROVIDE_NOOP_ACTION_TARGET:
      destroy_mask[-1] = 1

    # empty inventory -- nothing to destroy
    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0)\
        or self.dummy_obs or self.agent_in_combat:
      return destroy_mask

    # not equipped items in the inventory can be destroyed
    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0

    destroy_mask[:self.inventory.len] = not_equipped
    return destroy_mask

  def _make_give_target_mask(self):
    give_mask = np.zeros(self.config.PLAYER_N_OBS + self._noop_action, dtype=np.int8)
    if self.config.PROVIDE_NOOP_ACTION_TARGET:
      give_mask[-1] = 1
    # empty inventory -- nothing to give
    if not (self.config.ITEM_SYSTEM_ENABLED and self.inventory.len > 0)\
        or self.dummy_obs or self.agent_in_combat:
      return give_mask

    agent = self.agent()
    entities_pos = self.entities.values[:,[EntityState.State.attr_name_to_col["row"],
                                           EntityState.State.attr_name_to_col["col"]]]
    same_tile = utils.linf(entities_pos, (agent.row, agent.col)) == 0
    not_me = self.entities.ids != self.agent_id
    player = (self.entities.values[:,EntityState.State.attr_name_to_col["npc_type"]] == 0)

    give_mask[:self.entities.len] = same_tile & player & not_me
    return give_mask

  def _make_give_gold_mask(self):
    mask = np.zeros(self.config.PRICE_N_OBS, dtype=np.int8)
    if self.dummy_obs:
      return mask

    gold = int(self.agent().gold)
    if gold and not self.agent_in_combat:
      mask[:gold] = 1 # NOTE that action.Price starts from Discrete_1

    return mask

  def _make_sell_mask(self):
    sell_mask = np.zeros(self.config.INVENTORY_N_OBS + self._noop_action, dtype=np.int8)
    if self.config.PROVIDE_NOOP_ACTION_TARGET:
      sell_mask[-1] = 1

    # empty inventory -- nothing to sell
    if not (self.config.EXCHANGE_SYSTEM_ENABLED and self.inventory.len > 0) \
      or self.dummy_obs or self.agent_in_combat:
      return sell_mask

    not_equipped = self.inventory.values[:,ItemState.State.attr_name_to_col["equipped"]] == 0
    not_listed = self.inventory.values[:,ItemState.State.attr_name_to_col["listed_price"]] == 0

    sell_mask[:self.inventory.len] = not_equipped & not_listed
    return sell_mask

  def _make_buy_mask(self):
    buy_mask = np.zeros(self.config.MARKET_N_OBS + self._noop_action, dtype=np.int8)
    if self.config.PROVIDE_NOOP_ACTION_TARGET:
      buy_mask[-1] = 1

    if not self.config.EXCHANGE_SYSTEM_ENABLED or self.dummy_obs or self.agent_in_combat:
      return buy_mask

    agent = self.agent()
    market_items = self.market.values
    not_mine = market_items[:,ItemState.State.attr_name_to_col["owner_id"]] != self.agent_id

    # if the inventory is full, one can only buy existing ammo stack
    #   otherwise, one can buy anything owned by other, having enough money
    if self.inventory.len >= self.config.ITEM_INVENTORY_CAPACITY:
      exist_ammo_listings = self._existing_ammo_listings()
      if not np.any(exist_ammo_listings):
        return buy_mask
      not_mine &= exist_ammo_listings

    enough_gold = market_items[:,ItemState.State.attr_name_to_col["listed_price"]] <= agent.gold
    buy_mask[:self.market.len] = not_mine & enough_gold
    return buy_mask

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
