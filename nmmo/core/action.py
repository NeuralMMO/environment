# pylint: disable=no-method-argument,unused-argument,no-self-argument,no-member
from enum import Enum, auto
import numpy as np

from nmmo.lib import utils
from nmmo.lib.utils import staticproperty
from nmmo.systems.item import Stack
from nmmo.lib.event_code import EventCode
from nmmo.core.observation import Observation


class NodeType(Enum):
  #Tree edges
  STATIC = auto()    #Traverses all edges without decisions
  SELECTION = auto() #Picks an edge to follow

  #Executable actions
  ACTION    = auto() #No arguments
  CONSTANT  = auto() #Constant argument
  VARIABLE  = auto() #Variable argument

class Node(metaclass=utils.IterableNameComparable):
  @classmethod
  def init(cls, config):
    # noop_action is used in some of the N() methods
    cls.noop_action = 1 if config.PROVIDE_NOOP_ACTION_TARGET else 0

  @staticproperty
  def edges():
    return []

  #Fill these in
  @staticproperty
  def priority():
    return None

  @staticproperty
  def type():
    return None

  @staticproperty
  def leaf():
    return False

  @classmethod
  def N(cls, config):
    return len(cls.edges)

  def deserialize(realm, entity, index: int, obs: Observation):
    return index

class Fixed:
  pass

#ActionRoot
class Action(Node):
  nodeType = NodeType.SELECTION
  hooked   = False

  @classmethod
  def init(cls, config):
    # Sets up serialization domain
    if Action.hooked:
      return

    Action.hooked = True

  #Called upon module import (see bottom of file)
  #Sets up serialization domain
  def hook(config):
    idx = 0
    arguments = []
    for action in Action.edges(config):
      action.init(config)
      for args in action.edges: # pylint: disable=not-an-iterable
        args.init(config)
        if not "edges" in args.__dict__:
          continue
        for arg in args.edges:
          arguments.append(arg)
          arg.serial = tuple([idx])
          arg.idx = idx
          idx += 1
    Action.arguments = arguments

  @staticproperty
  def n():
    return len(Action.arguments)

  # pylint: disable=invalid-overridden-method
  @classmethod
  def edges(cls, config):
    """List of valid actions"""
    edges = [Move]
    if config.COMBAT_SYSTEM_ENABLED:
      edges.append(Attack)
    if config.ITEM_SYSTEM_ENABLED:
      edges += [Use, Give, Destroy]
    if config.EXCHANGE_SYSTEM_ENABLED:
      edges += [Buy, Sell, GiveGold]
    if config.COMMUNICATION_SYSTEM_ENABLED:
      edges.append(Comm)
    return edges

class Move(Node):
  priority = 60
  nodeType = NodeType.SELECTION
  def call(realm, entity, direction):
    if direction is None:
      return

    assert entity.alive, "Dead entity cannot act"
    assert realm.map.is_valid_pos(*entity.pos), "Invalid entity position"

    r, c  = entity.pos
    ent_id = entity.ent_id
    entity.history.last_pos = (r, c)
    r_delta, c_delta = direction.delta
    r_new, c_new = r+r_delta, c+c_delta

    if not realm.map.is_valid_pos(r_new, c_new) or \
       realm.map.tiles[r_new, c_new].impassible:
      return

    # ALLOW_MOVE_INTO_OCCUPIED_TILE only applies to players, NOT npcs
    if entity.is_player and not realm.config.ALLOW_MOVE_INTO_OCCUPIED_TILE and \
       realm.map.tiles[r_new, c_new].occupied:
      return

    if entity.status.freeze > 0:
      return

    entity.set_pos(r_new, c_new)
    realm.map.tiles[r, c].remove_entity(ent_id)
    realm.map.tiles[r_new, c_new].add_entity(entity)

    # exploration record keeping. moved from entity.py, History.update()
    progress_to_center = realm.map.dist_border_center -\
      utils.linf_single(realm.map.center_coord, (r_new, c_new))
    if progress_to_center > entity.history.exploration:
      entity.history.exploration = progress_to_center
      if entity.is_player:
        realm.event_log.record(EventCode.GO_FARTHEST, entity,
                               distance=progress_to_center)

    # CHECK ME: material.Impassible includes void, so this line is not reachable
    #   Does this belong to Entity/Player.update()?
    if realm.map.tiles[r_new, c_new].void:
      entity.receive_damage(None, entity.resources.health.val)

  @staticproperty
  def edges():
    return [Direction]

  @staticproperty
  def leaf():
    return True

  def enabled(config):
    return True

class Direction(Node):
  argType = Fixed

  @staticproperty
  def edges():
    return [North, South, East, West, Stay]

  def deserialize(realm, entity, index: int, obs):
    return deserialize_fixed_arg(Direction, index)

# a quick helper function
def deserialize_fixed_arg(arg, index):
  if isinstance(index, (int, np.int64)):
    if index < 0:
      return None # so that the action will be discarded
    val = min(index, len(arg.edges)-1)
    return arg.edges[val]

  # if index is not int, it's probably already deserialized
  if index not in arg.edges:
    return None # so that the action will be discarded
  return index

class North(Node):
  delta = (-1, 0)

class South(Node):
  delta = (1, 0)

class East(Node):
  delta = (0, 1)

class West(Node):
  delta = (0, -1)

class Stay(Node):
  delta = (0, 0)

class Attack(Node):
  priority = 50
  nodeType = NodeType.SELECTION
  @staticproperty
  def n():
    return 3

  @staticproperty
  def edges():
    return [Style, Target]

  @staticproperty
  def leaf():
    return True

  def enabled(config):
    return config.COMBAT_SYSTEM_ENABLED

  def in_range(entity, stim, config, N):
    R, C = stim.shape
    R, C = R//2, C//2

    rets = set([entity])
    for r in range(R-N, R+N+1):
      for c in range(C-N, C+N+1):
        for e in stim[r, c].entities.values():
          rets.add(e)

    rets = list(rets)
    return rets

  def call(realm, entity, style, target):
    if style is None or target is None:
      return None

    assert entity.alive, "Dead entity cannot act"

    config = realm.config
    if entity.is_player and not config.COMBAT_SYSTEM_ENABLED:
      return None

    # Testing a spawn immunity against old agents to avoid spawn camping
    immunity = config.COMBAT_SPAWN_IMMUNITY
    if entity.is_player and target.is_player and \
      target.history.time_alive < immunity:
      return None

    #Check if self targeted or target already dead
    if entity.ent_id == target.ent_id or not target.alive:
      return None

    #Can't attack out of range
    if utils.linf_single(entity.pos, target.pos) > style.attack_range(config):
      return None

    #Execute attack
    entity.history.attack = {}
    entity.history.attack["target"] = target.ent_id
    entity.history.attack["style"] = style.__name__
    target.attacker = entity
    target.attacker_id.update(entity.ent_id)

    from nmmo.systems import combat
    dmg = combat.attack(realm, entity, target, style.skill)

    # record the combat tick for both entities
    # players and npcs both have latest_combat_tick in EntityState
    for ent in [entity, target]:
      ent.latest_combat_tick.update(realm.tick + 1) # because the tick is about to increment

    return dmg

class Style(Node):
  argType = Fixed
  @staticproperty
  def edges():
    return [Melee, Range, Mage]

  def deserialize(realm, entity, index: int, obs):
    return deserialize_fixed_arg(Style, index)

class Target(Node):
  argType = None

  @classmethod
  def N(cls, config):
    return config.PLAYER_N_OBS + cls.noop_action

  def deserialize(realm, entity, index: int, obs: Observation):
    if index >= len(obs.entities.ids):
      return None
    return realm.entity_or_none(obs.entities.ids[index])

class Melee(Node):
  nodeType = NodeType.ACTION
  freeze=False

  def attack_range(config):
    return config.COMBAT_MELEE_REACH

  def skill(entity):
    return entity.skills.melee

class Range(Node):
  nodeType = NodeType.ACTION
  freeze=False

  def attack_range(config):
    return config.COMBAT_RANGE_REACH

  def skill(entity):
    return entity.skills.range

class Mage(Node):
  nodeType = NodeType.ACTION
  freeze=False

  def attack_range(config):
    return config.COMBAT_MAGE_REACH

  def skill(entity):
    return entity.skills.mage

class InventoryItem(Node):
  argType  = None

  @classmethod
  def N(cls, config):
    return config.INVENTORY_N_OBS + cls.noop_action

  def deserialize(realm, entity, index: int, obs: Observation):
    if index >= len(obs.inventory.ids):
      return None
    return realm.items.get(obs.inventory.ids[index])

class Use(Node):
  priority = 10

  @staticproperty
  def edges():
    return [InventoryItem]

  def enabled(config):
    return config.ITEM_SYSTEM_ENABLED

  def call(realm, entity, item):
    if item is None or item.owner_id.val != entity.ent_id:
      return

    assert entity.alive, "Dead entity cannot act"
    assert entity.is_player, "Npcs cannot use an item"
    assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

    if not realm.config.ITEM_SYSTEM_ENABLED:
      return

    if item not in entity.inventory:
      return

    if entity.in_combat: # player cannot use item during combat
      return

    # cannot use listed items or items that have higher level
    if item.listed_price.val > 0 or item.level_gt(entity):
      return

    item.use(entity)

class Destroy(Node):
  priority = 40

  @staticproperty
  def edges():
    return [InventoryItem]

  def enabled(config):
    return config.ITEM_SYSTEM_ENABLED

  def call(realm, entity, item):
    if item is None or item.owner_id.val != entity.ent_id:
      return

    assert entity.alive, "Dead entity cannot act"
    assert entity.is_player, "Npcs cannot destroy an item"
    assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

    if not realm.config.ITEM_SYSTEM_ENABLED:
      return

    if item not in entity.inventory:
      return

    if item.equipped.val: # cannot destroy equipped item
      return

    if entity.in_combat: # player cannot destroy item during combat
      return

    item.destroy()

    realm.event_log.record(EventCode.DESTROY_ITEM, entity)

class Give(Node):
  priority = 30

  @staticproperty
  def edges():
    return [InventoryItem, Target]

  def enabled(config):
    return config.ITEM_SYSTEM_ENABLED

  def call(realm, entity, item, target):
    if item is None or item.owner_id.val != entity.ent_id or target is None:
      return

    assert entity.alive, "Dead entity cannot act"
    assert entity.is_player, "Npcs cannot give an item"
    assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

    config = realm.config
    if not config.ITEM_SYSTEM_ENABLED:
      return

    if not (target.is_player and target.alive):
      return

    if item not in entity.inventory:
      return

    # cannot give the equipped or listed item
    if item.equipped.val or item.listed_price.val:
      return

    if entity.in_combat: # player cannot give item during combat
      return

    if not (config.ITEM_ALLOW_GIFT and
            entity.ent_id != target.ent_id and                      # but not self
            target.is_player):
      return

    # NOTE: allow give within the visual range
    if utils.linf_single(entity.pos, target.pos) > config.PLAYER_VISION_RADIUS:
      return

    if not target.inventory.space:
      # receiver inventory is full - see if it has an ammo stack with the same sig
      if isinstance(item, Stack):
        if not target.inventory.has_stack(item.signature):
          # no ammo stack with the same signature, so cannot give
          return
      else: # no space, and item is not ammo stack, so cannot give
        return

    entity.inventory.remove(item)
    target.inventory.receive(item)

    realm.event_log.record(EventCode.GIVE_ITEM, entity)

class GiveGold(Node):
  priority = 30

  @staticproperty
  def edges():
    # CHECK ME: for now using Price to indicate the gold amount to give
    return [Price, Target]

  def enabled(config):
    return config.EXCHANGE_SYSTEM_ENABLED

  def call(realm, entity, amount, target):
    if amount is None or target is None:
      return

    assert entity.alive, "Dead entity cannot act"
    assert entity.is_player, "Npcs cannot give gold"

    config = realm.config
    if not config.EXCHANGE_SYSTEM_ENABLED:
      return

    if not (target.is_player and target.alive):
      return

    if entity.in_combat: # player cannot give gold during combat
      return

    if not (config.ITEM_ALLOW_GIFT and
            entity.ent_id != target.ent_id and                      # but not self
            target.is_player):
      return

    # NOTE: allow give within the visual range
    if utils.linf_single(entity.pos, target.pos) > config.PLAYER_VISION_RADIUS:
      return

    if not isinstance(amount, int):
      amount = amount.val

    if amount > entity.gold.val: # no gold to give
      return

    entity.gold.decrement(amount)
    target.gold.increment(amount)

    realm.event_log.record(EventCode.GIVE_GOLD, entity)

class MarketItem(Node):
  argType  = None

  @classmethod
  def N(cls, config):
    return config.MARKET_N_OBS + cls.noop_action

  def deserialize(realm, entity, index: int, obs: Observation):
    if index >= len(obs.market.ids):
      return None
    return realm.items.get(obs.market.ids[index])

class Buy(Node):
  priority = 20
  argType  = Fixed

  @staticproperty
  def edges():
    return [MarketItem]

  def enabled(config):
    return config.EXCHANGE_SYSTEM_ENABLED

  def call(realm, entity, item):
    if item is None or item.owner_id.val == 0:
      return

    assert entity.alive, "Dead entity cannot act"
    assert entity.is_player, "Npcs cannot buy an item"
    assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak
    assert item.equipped.val == 0, "Listed item must not be equipped"

    if not realm.config.EXCHANGE_SYSTEM_ENABLED:
      return

    if entity.gold.val < item.listed_price.val: # not enough money
      return

    if entity.ent_id == item.owner_id.val: # cannot buy own item
      return

    if entity.in_combat: # player cannot buy item during combat
      return

    if not entity.inventory.space:
      # buyer inventory is full - see if it has an ammo stack with the same sig
      if isinstance(item, Stack):
        if not entity.inventory.has_stack(item.signature):
          # no ammo stack with the same signature, so cannot give
          return
      else: # no space, and item is not ammo stack, so cannot give
        return

    # one can try to buy, but the listing might have gone (perhaps bought by other)
    realm.exchange.buy(entity, item)

class Sell(Node):
  priority = 70
  argType  = Fixed

  @staticproperty
  def edges():
    return [InventoryItem, Price]

  def enabled(config):
    return config.EXCHANGE_SYSTEM_ENABLED

  def call(realm, entity, item, price):
    if item is None or item.owner_id.val != entity.ent_id or price is None:
      return

    assert entity.alive, "Dead entity cannot act"
    assert entity.is_player, "Npcs cannot sell an item"
    assert item.quantity.val > 0, "Item quantity cannot be 0" # indicates item leak

    if not realm.config.EXCHANGE_SYSTEM_ENABLED:
      return

    if item not in entity.inventory:
      return

    if entity.in_combat: # player cannot sell item during combat
      return

    # cannot sell the equipped or listed item
    if item.equipped.val or item.listed_price.val:
      return

    if not isinstance(price, int):
      price = price.val

    if not price > 0:
      return

    realm.exchange.sell(entity, item, price, realm.tick)

def init_discrete(values):
  classes = []
  for i in values:
    name = f"Discrete_{i}"
    cls  = type(name, (object,), {"val": i})
    classes.append(cls)

  return classes

class Price(Node):
  argType  = Fixed

  @classmethod
  def init(cls, config):
    # gold should be > 0
    cls.price_range = range(1, config.PRICE_N_OBS+1)
    Price.classes = init_discrete(cls.price_range)

  @classmethod
  def index(cls, price):
    try:
      return cls.price_range.index(price)
    except ValueError:
      # use the max price, which is config.PRICE_N_OBS
      return len(cls.price_range) - 1

  @staticproperty
  def edges():
    return Price.classes

  def deserialize(realm, entity, index: int, obs):
    return deserialize_fixed_arg(Price, index)

class Token(Node):
  argType  = Fixed

  @classmethod
  def init(cls, config):
    Token.classes = init_discrete(range(1, config.COMMUNICATION_NUM_TOKENS+1))

  @staticproperty
  def edges():
    return Token.classes

  def deserialize(realm, entity, index: int, obs):
    return deserialize_fixed_arg(Token, index)

class Comm(Node):
  argType  = Fixed
  priority = 99

  @staticproperty
  def edges():
    return [Token]

  def enabled(config):
    return config.COMMUNICATION_SYSTEM_ENABLED

  def call(realm, entity, token):
    if token is None:
      return

    entity.message.update(token.val)

#TODO: Solve AGI
class BecomeSkynet:
  pass
