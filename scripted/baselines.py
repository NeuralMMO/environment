# pylint: disable=invalid-name, attribute-defined-outside-init, no-member
from typing import Dict
from collections import defaultdict

import nmmo
from nmmo import material
from nmmo.systems import skill
import nmmo.systems.item as item_system
from nmmo.lib import colors
from nmmo.core import action
from nmmo.core.observation import Observation

from scripted import attack, move


class Scripted(nmmo.Scripted):
  '''Template class for baseline scripted models.

  You may either subclass directly or mirror the __call__ function'''
  scripted = True
  color    = colors.Neon.SKY
  def __init__(self, config, idx):
    '''
    Args:
        config : A forge.blade.core.Config object or subclass object
    '''
    super().__init__(config, idx)
    self.health_max = config.PLAYER_BASE_HEALTH

    if config.RESOURCE_SYSTEM_ENABLED:
      self.food_max   = config.RESOURCE_BASE
      self.water_max  = config.RESOURCE_BASE

    self.spawnR    = None
    self.spawnC    = None

  @property
  def policy(self):
    return self.__class__.__name__

  @property
  def forage_criterion(self) -> bool:
    '''Return true if low on food or water'''
    min_level = 7 * self.config.RESOURCE_DEPLETION_RATE
    return self.me.food <= min_level or self.me.water <= min_level

  def forage(self):
    '''Min/max food and water using Dijkstra's algorithm'''
    # TODO: do not access realm._np_random directly. ALSO see below for all other uses
    move.forageDijkstra(self.config, self.ob, self.actions,
                        self.food_max, self.water_max, self._np_random)

  def gather(self, resource):
    '''BFS search for a particular resource'''
    return move.gatherBFS(self.config, self.ob, self.actions, resource, self._np_random)

  def explore(self):
    '''Route away from spawn'''
    move.explore(self.config, self.ob, self.actions, self.me.row, self.me.col, self._np_random)

  @property
  def downtime(self):
    '''Return true if agent is not occupied with a high-priority action'''
    return not self.forage_criterion and self.attacker is None

  def evade(self):
    '''Target and path away from an attacker'''
    move.evade(self.config, self.ob, self.actions, self.attacker, self._np_random)
    self.target     = self.attacker
    self.targetID   = self.attackerID
    self.targetDist = self.attackerDist

  def attack(self):
    '''Attack the current target'''
    if self.target is not None:
      assert self.targetID is not None
      style = self._np_random.choice(self.style)
      attack.target(self.config, self.actions, style, self.targetID)

  def target_weak(self): # pylint: disable=inconsistent-return-statements
    '''Target the nearest agent if it is weak'''
    if self.closest is None:
      return False

    selfLevel  = self.me.level
    targLevel  = max(self.closest.melee_level, self.closest.range_level, self.closest.mage_level)

    if self.closest.npc_type == 1 or \
       targLevel <= selfLevel <= 5 or \
       selfLevel >= targLevel + 3:
      self.target     = self.closest
      self.targetID   = self.closestID
      self.targetDist = self.closestDist

  def scan_agents(self):
    '''Scan the nearby area for agents'''
    self.closest, self.closestDist   = attack.closestTarget(self.config, self.ob)
    self.attacker, self.attackerDist = attack.attacker(self.config, self.ob)

    self.closestID = None
    if self.closest is not None:
      self.closestID = self.ob.entities.index(self.closest.id)

    self.attackerID = None
    if self.attacker is not None:
      self.attackerID = self.ob.entities.index(self.attacker.id)

    self.target     = None
    self.targetID   = None
    self.targetDist = None

  def adaptive_control_and_targeting(self, explore=True):
    '''Balanced foraging, evasion, and exploration'''
    self.scan_agents()

    if self.attacker is not None:
      self.evade()
      return

    if self.fog_criterion:
      self.explore()
    elif self.forage_criterion or not explore:
      self.forage()
    else:
      self.explore()

    self.target_weak()

  def process_inventory(self):
    if not self.config.ITEM_SYSTEM_ENABLED:
      return

    self.inventory   = {}
    self.best_items: Dict   = {}
    self.item_counts = defaultdict(int)

    self.item_levels = {
      item_system.Hat.ITEM_TYPE_ID: self.level,
      item_system.Top.ITEM_TYPE_ID: self.level,
      item_system.Bottom.ITEM_TYPE_ID: self.level,
      item_system.Spear.ITEM_TYPE_ID: self.me.melee_level,
      item_system.Bow.ITEM_TYPE_ID: self.me.range_level,
      item_system.Wand.ITEM_TYPE_ID: self.me.mage_level,
      item_system.Rod.ITEM_TYPE_ID: self.me.fishing_level,
      item_system.Gloves.ITEM_TYPE_ID: self.me.herbalism_level,
      item_system.Pickaxe.ITEM_TYPE_ID: self.me.prospecting_level,
      item_system.Axe.ITEM_TYPE_ID: self.me.carving_level,
      item_system.Chisel.ITEM_TYPE_ID: self.me.alchemy_level,
      item_system.Whetstone.ITEM_TYPE_ID: self.me.melee_level,
      item_system.Arrow.ITEM_TYPE_ID: self.me.range_level,
      item_system.Runes.ITEM_TYPE_ID: self.me.mage_level,
      item_system.Ration.ITEM_TYPE_ID: self.level,
      item_system.Potion.ITEM_TYPE_ID: self.level
    }

    for item_ary in self.ob.inventory.values:
      itm = item_system.ItemState.parse_array(item_ary)
      assert itm.quantity != 0

      # Too high level to equip or use
      if itm.type_id in self.item_levels and itm.level > self.item_levels[itm.type_id]:
        continue

      # cannot use listed item
      if itm.listed_price:
        continue

      self.item_counts[itm.type_id] += itm.quantity
      self.inventory[itm.id] = itm

      # Best by default
      if itm.type_id not in self.best_items:
        self.best_items[itm.type_id] = itm

      best_itm = self.best_items[itm.type_id]

      if itm.level > best_itm.level:
        self.best_items[itm.type_id] = itm

  def upgrade_heuristic(self, current_level, upgrade_level, price):
    return (upgrade_level - current_level) / max(price, 1)

  def process_market(self):
    if not self.config.EXCHANGE_SYSTEM_ENABLED:
      return

    self.market         = {}
    self.best_heuristic = {}

    for item_ary in self.ob.market.values:
      itm = item_system.ItemState.parse_array(item_ary)

      self.market[itm.id] = itm

      # Prune Unaffordable
      if itm.listed_price > self.me.gold:
        continue

      # Too high level to equip
      if itm.type_id in self.item_levels and itm.level > self.item_levels[itm.type_id] :
        continue

      #Current best item level
      current_level = 0
      if itm.type_id in self.best_items:
        current_level = self.best_items[itm.type_id].level

      itm.heuristic = self.upgrade_heuristic(current_level, itm.level, itm.listed_price)

      #Always count first item
      if itm.type_id not in self.best_heuristic:
        self.best_heuristic[itm.type_id] = itm
        continue

      #Better heuristic value
      if itm.heuristic > self.best_heuristic[itm.type_id].heuristic:
        self.best_heuristic[itm.type_id] = itm

  def equip(self, items: set):
    for type_id, itm in self.best_items.items():
      if type_id not in items:
        continue

      if itm.equipped or itm.listed_price:
        continue

      # InventoryItem needs where the item is (index) in the inventory
      self.actions[action.Use] = {
        action.InventoryItem: self.ob.inventory.index(itm.id)}

    return True

  def consume(self):
    if self.me.health <= self.health_max // 2 \
        and item_system.Potion.ITEM_TYPE_ID in self.best_items:
      itm = self.best_items[item_system.Potion.ITEM_TYPE_ID]
    elif (self.me.food == 0 or self.me.water == 0) \
        and item_system.Ration.ITEM_TYPE_ID in self.best_items:
      itm = self.best_items[item_system.Ration.ITEM_TYPE_ID]
    else:
      return

    if itm.listed_price:
      return

    # InventoryItem needs where the item is (index) in the inventory
    self.actions[action.Use] = {
      action.InventoryItem: self.ob.inventory.index(itm.id)}

  def sell(self, keep_k: dict, keep_best: set):
    for itm in self.inventory.values():
      price = int(max(itm.level, 1))
      assert itm.quantity > 0

      if itm.equipped or itm.listed_price:
        continue

      if itm.type_id in keep_k:
        owned = self.item_counts[itm.type_id]
        k     = keep_k[itm.type_id]
        if owned <= k:
          continue

      #Exists an equippable of the current class, best needs to be kept, and this is the best item
      if itm.type_id in self.best_items and \
        itm.type_id in keep_best and \
        itm.id == self.best_items[itm.type_id].id:
        continue

      self.actions[action.Sell] = {
        action.InventoryItem: self.ob.inventory.index(itm.id),
        action.Price: action.Price.index(price) }

      return itm

  def buy(self, buy_k: dict, buy_upgrade: set):
    if len(self.inventory) >= self.config.ITEM_INVENTORY_CAPACITY:
      return

    purchase = None
    best = list(self.best_heuristic.items())
    self._np_random.shuffle(best)
    for type_id, itm in best:
      # Buy top k
      if type_id in buy_k:
        owned = self.item_counts[type_id]
        k = buy_k[type_id]
        if owned < k:
          purchase = itm

      # Check if item desired and upgrade
      elif type_id in buy_upgrade and itm.heuristic > 0:
        purchase = itm

      # Buy best heuristic upgrade
      if purchase:
        self.actions[action.Buy] = {
          action.MarketItem: self.ob.market.index(purchase.id)}
        return

  def exchange(self):
    if not self.config.EXCHANGE_SYSTEM_ENABLED:
      return

    self.process_market()
    self.sell(keep_k=self.supplies, keep_best=self.wishlist)
    self.buy(buy_k=self.supplies, buy_upgrade=self.wishlist)

  def use(self):
    self.process_inventory()
    if self.config.EQUIPMENT_SYSTEM_ENABLED and not self.consume():
      self.equip(items=self.wishlist)

  def __call__(self, observation: Observation):
    '''Process observations and return actions'''
    assert self._np_random is not None, "Agent's RNG must be set."
    self.actions = {}

    self.ob = observation
    self.me = observation.agent()

    # combat level
    self.me.level = max(self.me.melee_level, self.me.range_level, self.me.mage_level)

    self.skills = {
      skill.Melee: self.me.melee_level,
      skill.Range: self.me.range_level,
      skill.Mage: self.me.mage_level,
      skill.Fishing: self.me.fishing_level,
      skill.Herbalism: self.me.herbalism_level,
      skill.Prospecting: self.me.prospecting_level,
      skill.Carving: self.me.carving_level,
      skill.Alchemy: self.me.alchemy_level
    }

    # TODO(kywch): need a consistent level variables
    # level for using armor, rations, and potion
    self.level = min(1, max(self.skills.values()))

    if self.spawnR is None:
      self.spawnR = self.me.row
    if self.spawnC is None:
      self.spawnC = self.me.col

    # When to run from death fog in BR configs
    self.fog_criterion = None
    if self.config.DEATH_FOG_ONSET is not None:
      time_alive = self.me.time_alive
      start_running = time_alive > self.config.DEATH_FOG_ONSET - 64
      run_now = time_alive % max(1, int(1 / self.config.DEATH_FOG_SPEED))
      self.fog_criterion = start_running and run_now


class Sleeper(Scripted):
  '''Do Nothing'''
  def __call__(self, obs):
    super().__call__(obs)
    return {}
class Random(Scripted):
  '''Moves randomly'''
  def __call__(self, obs):
    super().__call__(obs)

    move.rand(self.config, self.ob, self.actions, self._np_random)
    return self.actions

class Meander(Scripted):
  '''Moves randomly on safe terrain'''
  def __call__(self, obs):
    super().__call__(obs)

    move.meander(self.config, self.ob, self.actions, self._np_random)
    return self.actions

class Explore(Scripted):
  '''Actively explores towards the center'''
  def __call__(self, obs):
    super().__call__(obs)

    self.explore()

    return self.actions

class Forage(Scripted):
  '''Forages using Dijkstra's algorithm and actively explores'''
  def __call__(self, obs):
    super().__call__(obs)

    if self.forage_criterion:
      self.forage()
    else:
      self.explore()

    return self.actions

class Combat(Scripted):
  '''Forages, fights, and explores'''
  def __init__(self, config, idx):
    super().__init__(config, idx)
    self.style  = [action.Melee, action.Range, action.Mage]

  @property
  def supplies(self):
    return {
      item_system.Ration.ITEM_TYPE_ID: 2,
      item_system.Potion.ITEM_TYPE_ID: 2,
      self.ammo.ITEM_TYPE_ID: 10
    }

  @property
  def wishlist(self):
    return {
      item_system.Hat.ITEM_TYPE_ID,
      item_system.Top.ITEM_TYPE_ID,
      item_system.Bottom.ITEM_TYPE_ID,
      self.weapon.ITEM_TYPE_ID,
      self.ammo.ITEM_TYPE_ID
    }

  def __call__(self, obs):
    super().__call__(obs)
    self.use()
    self.exchange()

    self.adaptive_control_and_targeting()
    self.attack()

    return self.actions

class Gather(Scripted):
  '''Forages, fights, and explores'''
  def __init__(self, config, idx):
    super().__init__(config, idx)
    self.resource = [material.Fish, material.Herb, material.Ore, material.Tree, material.Crystal]

  @property
  def supplies(self):
    return {
      item_system.Ration.ITEM_TYPE_ID: 1,
      item_system.Potion.ITEM_TYPE_ID: 1
    }

  @property
  def wishlist(self):
    return {
      item_system.Hat.ITEM_TYPE_ID,
      item_system.Top.ITEM_TYPE_ID,
      item_system.Bottom.ITEM_TYPE_ID,
      self.tool.ITEM_TYPE_ID
    }

  def __call__(self, obs):
    super().__call__(obs)
    self.use()
    self.exchange()

    if self.forage_criterion:
      self.forage()
    elif self.fog_criterion or not self.gather(self.resource):
      self.explore()

    return self.actions

class Fisher(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.PROFESSION_SYSTEM_ENABLED:
      self.resource = [material.Fish]
    self.tool     = item_system.Rod

class Herbalist(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.PROFESSION_SYSTEM_ENABLED:
      self.resource = [material.Herb]
    self.tool     = item_system.Gloves

class Prospector(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.PROFESSION_SYSTEM_ENABLED:
      self.resource = [material.Ore]
    self.tool     = item_system.Pickaxe

class Carver(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.PROFESSION_SYSTEM_ENABLED:
      self.resource = [material.Tree]
    self.tool     = item_system.Axe

class Alchemist(Gather):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.PROFESSION_SYSTEM_ENABLED:
      self.resource = [material.Crystal]
    self.tool     = item_system.Chisel

class Melee(Combat):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.COMBAT_SYSTEM_ENABLED:
      self.style  = [action.Melee]
    self.weapon = item_system.Spear
    self.ammo   = item_system.Whetstone

class Range(Combat):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.COMBAT_SYSTEM_ENABLED:
      self.style  = [action.Range]
    self.weapon = item_system.Bow
    self.ammo   = item_system.Arrow

class Mage(Combat):
  def __init__(self, config, idx):
    super().__init__(config, idx)
    if config.COMBAT_SYSTEM_ENABLED:
      self.style  = [action.Mage]
    self.weapon = item_system.Wand
    self.ammo   = item_system.Runes
