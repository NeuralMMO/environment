# pylint: disable=invalid-name
from __future__ import annotations

import os
import sys
import logging
import re

import nmmo
from nmmo.core.agent import Agent
from nmmo.core.terrain import MapGenerator
from nmmo.lib import utils, material, spawn

CONFIG_ATTR_PATTERN = r"^[A-Z_]+$"
GAME_SYSTEMS = ["TERRAIN", "RESOURCE", "COMBAT", "NPC", "PROGRESSION", "ITEM",
                "EQUIPMENT", "PROFESSION", "EXCHANGE", "COMMUNICATION"]

# These attributes are critical for trainer and must not change from the initial values
OBS_ATTRS = set(["MAX_HORIZON", "PLAYER_N", "MAP_N_OBS", "PLAYER_N_OBS", "TASK_EMBED_DIM",
                 "ITEM_INVENTORY_CAPACITY", "MARKET_N_OBS", "PRICE_N_OBS",
                 "COMMUNICATION_NUM_TOKENS", "COMMUNICATION_N_OBS", "PROVIDE_ACTION_TARGETS",
                 "PROVIDE_DEATH_FOG_OBS", "PROVIDE_NOOP_ACTION_TARGET"])
IMMUTABLE_ATTRS = set(["USE_CYTHON", "CURRICULUM_FILE_PATH", "PLAYER_VISION_RADIUS", "MAP_SIZE",
                       "PLAYER_BASE_HEALTH", "RESOURCE_BASE", "PROGRESSION_LEVEL_MAX"])


class Template(metaclass=utils.StaticIterable):
  def __init__(self):
    self._data = {}
    cls = type(self)

    # Set defaults from static properties
    for attr in dir(cls):
      val = getattr(cls, attr)
      if re.match(CONFIG_ATTR_PATTERN, attr) and not isinstance(val, property):
        self._data[attr] = val

  def override(self, **kwargs):
    for k, v in kwargs.items():
      err = f'CLI argument: {k} is not a Config property'
      assert hasattr(self, k), err
      self.set(k, v)

  def set(self, k, v):
    if not isinstance(v, property):
      try:
        setattr(self, k, v)
      except AttributeError:
        logging.error('Cannot set attribute: %s to %s', str(k), str(v))
        sys.exit()
    self._data[k] = v

  # pylint: disable=bad-builtin
  def print(self):
    key_len = 0
    for k in self._data:
      key_len = max(key_len, len(k))

    print('Configuration')
    for k, v in self._data.items():
      print(f'   {k:{key_len}s}: {v}')

  def items(self):
    return self._data.items()

  def __iter__(self):
    for k in self._data:
      yield k

  def keys(self):
    return self._data.keys()

  def values(self):
    return self._data.values()

def validate(config):
  err = 'config.Config is a base class. Use config.{Small, Medium Large}'''
  assert isinstance(config, Config), err
  assert config.HORIZON < config.MAX_HORIZON, 'HORIZON must be <= MAX_HORIZON'

  if not config.TERRAIN_SYSTEM_ENABLED:
    err = 'Invalid Config: {} requires Terrain'
    assert not config.RESOURCE_SYSTEM_ENABLED, err.format('Resource')
    assert not config.PROFESSION_SYSTEM_ENABLED, err.format('Profession')

  if not config.COMBAT_SYSTEM_ENABLED:
    err = 'Invalid Config: {} requires Combat'
    assert not config.NPC_SYSTEM_ENABLED, err.format('NPC')

  if not config.ITEM_SYSTEM_ENABLED:
    err = 'Invalid Config: {} requires Inventory'
    assert not config.EQUIPMENT_SYSTEM_ENABLED, err.format('Equipment')
    assert not config.PROFESSION_SYSTEM_ENABLED, err.format('Profession')
    assert not config.EXCHANGE_SYSTEM_ENABLED, err.format('Exchange')


class Config(Template):
  '''An environment configuration object

  Global constants are defined as static class variables. You can override
  any Config variable using standard CLI syntax (e.g. --NENT=128).

  The default config as of v1.5 uses 1024x1024 maps with up to 2048 agents
  and 1024 NPCs. It is suitable to time horizons of 8192+ steps. For smaller
  experiments, consider the SmallMaps config.

  Notes:
    We use Google Fire internally to replace standard manual argparse
    definitions for each Config property. This means you can subclass
    Config to add new static attributes -- CLI definitions will be
    generated automatically.
  '''
  env_initialized = False

  def __init__(self):
    super().__init__()
    self._attr_to_reset = []

    # TODO: Come up with a better way
    # to resolve mixin MRO conflicts
    for system in GAME_SYSTEMS:
      if not hasattr(self, f'{system}_SYSTEM_ENABLED'):
        self.set(f'{system}_SYSTEM_ENABLED', False)

    if __debug__:
      validate(self)

    deprecated_attrs = [
      'NENT', 'NPOP', 'AGENTS', 'NMAPS', 'FORCE_MAP_GENERATION', 'SPAWN']

    for attr in deprecated_attrs:
      assert not hasattr(self, attr), f'{attr} has been deprecated or renamed'

  @property
  def original(self):
    return self._data

  def reset(self):
    '''Reset all attributes changed during the episode'''
    for attr in self._attr_to_reset:
      setattr(self, attr, self.original[attr])

  def set(self, k, v):
    assert self.env_initialized is False, 'Cannot set config attr after env init'
    super().set(k, v)

  def set_for_episode(self, k, v):
    '''Set a config property for the current episode'''
    assert hasattr(self, k), f'Invalid config property: {k}'
    assert k not in OBS_ATTRS, f'Cannot change OBS config {k} during the episode'
    assert k not in IMMUTABLE_ATTRS, f'Cannot change {k} during the episode'
    # Cannot turn on a game system that was not enabled when the env was created
    if k.endswith('_SYSTEM_ENABLED') and self._data[k] is False and v is True:
      raise AssertionError(f'Cannot turn on {k} because it was not enabled during env init')

    # Change only the attribute and keep the original value in the data dict
    setattr(self, k, v)
    self._attr_to_reset.append(k)

  @property
  def enabled_systems(self):
    '''Return a list of the enabled systems from Env.__init__()'''
    return [k[:-len('_SYSTEM_ENABLED')]
            for k, v in self._data.items() if k.endswith('_SYSTEM_ENABLED') and v is True]

  @property
  def system_states(self):
    '''Return a one-hot encoding of each system enabled/disabled,
       which can be used as an observation and changed from episode to episode'''
    return [int(getattr(self, f'{system}_SYSTEM_ENABLED')) for system in GAME_SYSTEMS]

  def are_systems_enabled(self, systems):  # systems is a list of strings
    '''Check if all provided systems are enabled'''
    return all(s.upper() in self.enabled_systems for s in systems)

  def toggle_systems(self, target_systems):  # systems is a list of strings
    '''Activate only the provided game systems and turn off the others'''
    target_systems = [s.upper() for s in target_systems]
    for system in target_systems:
      assert system in self.enabled_systems, f'Invalid game system: {system}'
      self.set_for_episode(f'{system}_SYSTEM_ENABLED', True)

    for system in self.enabled_systems:
      if system not in target_systems:
        self.set_for_episode(f'{system}_SYSTEM_ENABLED', False)

  ############################################################################
  ### Meta-Parameters
  PLAYERS                      = [Agent]
  '''Player classes from which to spawn'''

  @property
  def PLAYER_POLICIES(self):
    '''Number of player policies'''
    return len(self.PLAYERS)

  PLAYER_N                     = None
  '''Maximum number of players spawnable in the environment'''

  @property
  def POSSIBLE_AGENTS(self):
    '''List of possible agents to spawn'''
    return list(range(1, self.PLAYER_N + 1))

  # TODO: CHECK if there could be 100+ entities within one's vision
  PLAYER_N_OBS                 = 100
  '''Number of distinct agent observations'''

  MAX_HORIZON = 2**15 - 1  # this is arbitrary
  '''Maximum number of steps the environment can run for'''

  HORIZON = 1024
  '''Number of steps before the environment resets'''

  GAME_PACKS = None
  '''List of game packs to load and sample: [(game class, sampling weight)]'''

  CURRICULUM_FILE_PATH = None
  '''Path to a curriculum task file containing a list of task specs for training'''

  TASK_EMBED_DIM = 4096
  '''Dimensionality of task embeddings'''

  ALLOW_MULTI_TASKS_PER_AGENT = False
  '''Whether to allow multiple tasks per agent'''

  PROVIDE_ACTION_TARGETS       = True
  '''Provide action targets mask'''

  PROVIDE_NOOP_ACTION_TARGET   = True
  '''Provide a no-op option for each action'''

  PROVIDE_DEATH_FOG_OBS = False
  '''Provide death fog observation'''

  ALLOW_MOVE_INTO_OCCUPIED_TILE = True
  '''Whether agents can move into tiles occupied by other agents/npcs
     However, this does not apply to spawning'''


  ############################################################################
  ### System/debug Parameters
  USE_CYTHON = True
  '''Whether to use cython modules for performance'''

  IMMORTAL = False
  '''Debug parameter: prevents agents from dying except by void'''


  ############################################################################
  ### Player Parameters
  PLAYER_BASE_HEALTH           = 100
  '''Initial agent health'''

  PLAYER_VISION_RADIUS         = 7
  '''Number of tiles an agent can see in any direction'''

  @property
  def PLAYER_VISION_DIAMETER(self):
    '''Size of the square tile crop visible to an agent'''
    return 2*self.PLAYER_VISION_RADIUS + 1

  PLAYER_HEALTH_INCREMENT      = False
  '''Whether to increment health by 1 per tick for players, like npcs'''

  DEATH_FOG_ONSET              = None
  '''How long before spawning death fog. None for no death fog'''

  DEATH_FOG_SPEED              = 1
  '''Number of tiles per tick that the fog moves in'''

  DEATH_FOG_FINAL_SIZE         = 8
  '''Number of tiles from the center that the fog stops'''

  PLAYER_LOADER                = spawn.SequentialLoader
  '''Agent loader class specifying spawn sampling'''


  ############################################################################
  ### Team Parameters
  TEAMS                        = None  # Dict[Any, List[int]]
  '''A dictionary of team assignments: key is team_id, value is a list of agent_ids'''


  ############################################################################
  ### Map Parameters
  MAP_N                        = 1
  '''Number of maps to generate'''

  MAP_N_TILE                   = len(material.All.materials)
  '''Number of distinct terrain tile types'''

  @property
  def MAP_N_OBS(self):
    '''Number of distinct tile observations'''
    return int(self.PLAYER_VISION_DIAMETER ** 2)

  MAP_SIZE                     = None
  '''Size of the whole map, including the center and borders'''

  MAP_CENTER                   = None
  '''Size of each map (number of tiles along each side), where agents can move around'''

  @property
  def MAP_BORDER(self):
    '''Number of background, void border tiles surrounding each side of the map'''
    return int((self.MAP_SIZE - self.MAP_CENTER) // 2)

  MAP_GENERATOR                = MapGenerator
  '''Specifies a user map generator. Uses default generator if unspecified.'''

  MAP_FORCE_GENERATION         = True
  '''Whether to regenerate and overwrite existing maps'''

  MAP_RESET_FROM_FRACTAL       = True
  '''Whether to regenerate the map from the fractal source'''

  MAP_GENERATE_PREVIEWS        = False
  '''Whether map generation should also save .png previews (slow + large file size)'''

  MAP_PREVIEW_DOWNSCALE        = 1
  '''Downscaling factor for png previews'''


  ############################################################################
  ### Path Parameters
  PATH_ROOT                = os.path.dirname(nmmo.__file__)
  '''Global repository directory'''

  PATH_CWD                 = os.getcwd()
  '''Working directory'''

  PATH_RESOURCE            = os.path.join(PATH_ROOT, 'resource')
  '''Resource directory'''

  PATH_TILE                = os.path.join(PATH_RESOURCE, '{}.png')
  '''Tile path -- format me with tile name'''

  PATH_MAPS                = None
  '''Generated map directory'''

  PATH_MAP_SUFFIX          = 'map{}/map.npy'
  '''Map file name'''

  PATH_FRACTAL_SUFFIX      = 'map{}/fractal.npy'
  '''Fractal file name'''


############################################################################
### Game Systems (Static Mixins)
class Terrain:
  '''Terrain Game System'''

  TERRAIN_SYSTEM_ENABLED       = True
  '''Game system flag'''

  TERRAIN_FLIP_SEED            = False
  '''Whether to negate the seed used for generation (useful for unique heldout maps)'''

  TERRAIN_FREQUENCY            = -3
  '''Base noise frequency range (log2 space)'''

  TERRAIN_FREQUENCY_OFFSET     = 7
  '''Noise frequency octave offset (log2 space)'''

  TERRAIN_LOG_INTERPOLATE_MIN  = -2
  '''Minimum interpolation log-strength for noise frequencies'''

  TERRAIN_LOG_INTERPOLATE_MAX  = 0
  '''Maximum interpolation log-strength for noise frequencies'''

  TERRAIN_TILES_PER_OCTAVE     = 8
  '''Number of octaves sampled from log2 spaced TERRAIN_FREQUENCY range'''

  TERRAIN_VOID                 = 0.0
  '''Noise threshold for void generation'''

  TERRAIN_WATER                = 0.30
  '''Noise threshold for water generation'''

  TERRAIN_GRASS                = 0.70
  '''Noise threshold for grass'''

  TERRAIN_FOILAGE              = 0.85
  '''Noise threshold for foilage (food tile)'''

  TERRAIN_RESET_TO_GRASS       = False
  '''Whether to make all tiles grass.
     Only works when MAP_RESET_FROM_FRACTAL is True'''

  TERRAIN_DISABLE_STONE        = False
  '''Disable stone (obstacle) tiles'''

  TERRAIN_SCATTER_EXTRA_RESOURCES = False
  '''Whether to scatter extra food, water on the map.
     Only works when MAP_RESET_FROM_FRACTAL is True'''

class Resource:
  '''Resource Game System'''

  RESOURCE_SYSTEM_ENABLED             = True
  '''Game system flag'''

  RESOURCE_BASE                       = 100
  '''Initial level and capacity for food and water'''

  RESOURCE_DEPLETION_RATE             = 5
  '''Depletion rate for food and water'''

  RESOURCE_STARVATION_RATE            = 10
  '''Damage per tick without food'''

  RESOURCE_DEHYDRATION_RATE           = 10
  '''Damage per tick without water'''

  RESOURCE_RESILIENT_POPULATION       = 0
  '''Training helper: proportion of population that is resilient to starvation and dehydration
     (e.g. 0.1 means 10% of the population is resilient to starvation and dehydration)
     This is to make some agents live longer during training to sample from "advanced" agents.'''

  RESOURCE_DAMAGE_REDUCTION           = 0.5
  '''Training helper: damage reduction from starvation and dehydration for resilient agents'''

  RESOURCE_FOILAGE_CAPACITY           = 1
  '''Maximum number of harvests before a foilage tile decays'''

  RESOURCE_FOILAGE_RESPAWN            = 0.025
  '''Probability that a harvested foilage tile will regenerate each tick'''

  RESOURCE_HARVEST_RESTORE_FRACTION   = 1.0
  '''Fraction of maximum capacity restored upon collecting a resource'''

  RESOURCE_HEALTH_REGEN_THRESHOLD     = 0.5
  '''Fraction of maximum resource capacity required to regen health'''

  RESOURCE_HEALTH_RESTORE_FRACTION    = 0.1
  '''Fraction of health restored per tick when above half food+water'''


# NOTE: Included self to be picklable (in torch.save) since lambdas are not picklable
def original_combat_damage_formula(self, offense, defense, multiplier, minimum_proportion):
  # pylint: disable=unused-argument
  return int(multiplier * (offense * (15 / (15 + defense))))

def alt_combat_damage_formula(self, offense, defense, multiplier, minimum_proportion):
  # pylint: disable=unused-argument
  return int(max(multiplier * offense - defense, offense * minimum_proportion))

class Combat:
  '''Combat Game System'''

  COMBAT_SYSTEM_ENABLED              = True
  '''Game system flag'''

  COMBAT_SPAWN_IMMUNITY              = 20
  '''Agents older than this many ticks cannot attack agents younger than this many ticks'''

  COMBAT_ALLOW_FLEXIBLE_STYLE        = True
  '''Whether to allow agents to attack with any style in a given turn''' 

  COMBAT_STATUS_DURATION             = 3
  '''Combat status lasts for this many ticks after the last combat event.
     Combat events include both attacking and being attacked.'''

  COMBAT_WEAKNESS_MULTIPLIER         = 1.5
  '''Multiplier for super-effective attacks'''

  COMBAT_MINIMUM_DAMAGE_PROPORTION   = 0.25
  '''Minimum proportion of damage to inflict on a target'''

  # NOTE: When using a custom function, include "self" as the first arg
  COMBAT_DAMAGE_FORMULA = original_combat_damage_formula
  '''Damage formula'''

  COMBAT_MELEE_DAMAGE                = 10
  '''Melee attack damage'''

  COMBAT_MELEE_REACH                 = 3
  '''Reach of attacks using the Melee skill'''

  COMBAT_RANGE_DAMAGE                = 10
  '''Range attack damage'''

  COMBAT_RANGE_REACH                 = 3
  '''Reach of attacks using the Range skill'''

  COMBAT_MAGE_DAMAGE                 = 10
  '''Mage attack damage'''

  COMBAT_MAGE_REACH                  = 3
  '''Reach of attacks using the Mage skill'''


def default_exp_threshold(base_exp, max_level):
  import math
  additional_exp_per_level = [round(base_exp * math.sqrt(lvl))
                              for lvl in range(1, max_level+1)]
  return [sum(additional_exp_per_level[:lvl]) for lvl in range(max_level)]

class Progression:
  '''Progression Game System'''

  PROGRESSION_SYSTEM_ENABLED        = True
  '''Game system flag'''

  PROGRESSION_BASE_LEVEL            = 1
  '''Initial skill level'''

  PROGRESSION_LEVEL_MAX             = 10
  '''Max skill level'''

  PROGRESSION_EXP_THRESHOLD         = default_exp_threshold(90, PROGRESSION_LEVEL_MAX)
  '''A list of experience thresholds for each level'''

  PROGRESSION_COMBAT_XP_SCALE       = 3
  '''Additional XP for each attack for skills Melee, Range, and Mage'''

  PROGRESSION_AMMUNITION_XP_SCALE   = 15
  '''Additional XP for each harvest for Prospecting, Carving, and Alchemy'''

  PROGRESSION_CONSUMABLE_XP_SCALE   = 30
  '''Multiplier XP for each harvest for Fishing and Herbalism'''

  PROGRESSION_MELEE_BASE_DAMAGE     = 20
  '''Base Melee attack damage'''

  PROGRESSION_MELEE_LEVEL_DAMAGE    = 5
  '''Bonus Melee attack damage per level'''

  PROGRESSION_RANGE_BASE_DAMAGE     = 20
  '''Base Range attack damage'''

  PROGRESSION_RANGE_LEVEL_DAMAGE    = 5
  '''Bonus Range attack damage per level'''

  PROGRESSION_MAGE_BASE_DAMAGE      = 20
  '''Base Mage attack damage '''

  PROGRESSION_MAGE_LEVEL_DAMAGE     = 5
  '''Bonus Mage attack damage per level'''

  PROGRESSION_BASE_DEFENSE          = 0
  '''Base defense'''

  PROGRESSION_LEVEL_DEFENSE         = 5
  '''Bonus defense per level'''


class NPC:
  '''NPC Game System'''

  NPC_SYSTEM_ENABLED                  = True
  '''Game system flag'''

  NPC_N                               = None
  '''Maximum number of NPCs spawnable in the environment'''

  NPC_DEFAULT_REFILL_DEAD_NPCS        = True
  '''Whether to refill dead NPCs'''

  NPC_SPAWN_ATTEMPTS                  = 25
  '''Number of NPC spawn attempts per tick'''

  NPC_SPAWN_AGGRESSIVE                = 0.80
  '''Percentage distance threshold from spawn for aggressive NPCs'''

  NPC_SPAWN_NEUTRAL                   = 0.50
  '''Percentage distance threshold from spawn for neutral NPCs'''

  NPC_SPAWN_PASSIVE                   = 0.00
  '''Percentage distance threshold from spawn for passive NPCs'''

  NPC_LEVEL_MIN                       = 1
  '''Minimum NPC level'''

  NPC_LEVEL_MAX                       = 10
  '''Maximum NPC level'''

  NPC_BASE_DEFENSE                    = 0
  '''Base NPC defense'''

  NPC_LEVEL_DEFENSE                   = 15
  '''Bonus NPC defense per level'''

  NPC_BASE_DAMAGE                     = 15
  '''Base NPC damage'''

  NPC_LEVEL_DAMAGE                    = 15
  '''Bonus NPC damage per level'''

  NPC_ALLOW_ATTACK_OTHER_NPCS         = False
  '''Whether NPCs can attack other NPCs'''


class Item:
  '''Inventory Game System'''

  ITEM_SYSTEM_ENABLED                 = True
  '''Game system flag'''

  ITEM_N                              = 17
  '''Number of unique base item classes'''

  ITEM_INVENTORY_CAPACITY             = 12
  '''Number of inventory spaces'''

  ITEM_ALLOW_GIFT               = True
  '''Whether agents can give gold/item to each other'''

  @property
  def INVENTORY_N_OBS(self):
    '''Number of distinct item observations'''
    return self.ITEM_INVENTORY_CAPACITY


class Equipment:
  '''Equipment Game System'''

  EQUIPMENT_SYSTEM_ENABLED             = True
  '''Game system flag'''

  WEAPON_DROP_PROB = 0.025
  '''Chance of getting a weapon while harvesting ammunition'''

  EQUIPMENT_WEAPON_BASE_DAMAGE         = 15
  '''Base weapon damage'''

  EQUIPMENT_WEAPON_LEVEL_DAMAGE        = 15
  '''Added weapon damage per level'''

  EQUIPMENT_AMMUNITION_BASE_DAMAGE     = 15
  '''Base ammunition damage'''

  EQUIPMENT_AMMUNITION_LEVEL_DAMAGE    = 15
  '''Added ammunition damage per level'''

  EQUIPMENT_TOOL_BASE_DEFENSE          = 30
  '''Base tool defense'''

  EQUIPMENT_TOOL_LEVEL_DEFENSE         = 0
  '''Added tool defense per level'''

  EQUIPMENT_ARMOR_BASE_DEFENSE         = 0
  '''Base armor defense'''

  EQUIPMENT_ARMOR_LEVEL_DEFENSE        = 10
  '''Base equipment defense'''


class Profession:
  '''Profession Game System'''

  PROFESSION_SYSTEM_ENABLED           = True
  '''Game system flag'''

  PROFESSION_TREE_CAPACITY            = 1
  '''Maximum number of harvests before a tree tile decays'''

  PROFESSION_TREE_RESPAWN             = 0.105
  '''Probability that a harvested tree tile will regenerate each tick'''

  PROFESSION_ORE_CAPACITY             = 1
  '''Maximum number of harvests before an ore tile decays'''

  PROFESSION_ORE_RESPAWN              = 0.10
  '''Probability that a harvested ore tile will regenerate each tick'''

  PROFESSION_CRYSTAL_CAPACITY         = 1
  '''Maximum number of harvests before a crystal tile decays'''

  PROFESSION_CRYSTAL_RESPAWN          = 0.10
  '''Probability that a harvested crystal tile will regenerate each tick'''

  PROFESSION_HERB_CAPACITY            = 1
  '''Maximum number of harvests before an herb tile decays'''

  PROFESSION_HERB_RESPAWN             = 0.02
  '''Probability that a harvested herb tile will regenerate each tick'''

  PROFESSION_FISH_CAPACITY            = 1
  '''Maximum number of harvests before a fish tile decays'''

  PROFESSION_FISH_RESPAWN             = 0.02
  '''Probability that a harvested fish tile will regenerate each tick'''

  def PROFESSION_CONSUMABLE_RESTORE(self, level):
    '''Amount of food/water restored by consuming a consumable item'''
    return 50 + 5*level


class Exchange:
  '''Exchange Game System'''

  EXCHANGE_SYSTEM_ENABLED             = True
  '''Game system flag'''

  EXCHANGE_BASE_GOLD                  = 1
  '''Initial gold amount'''

  EXCHANGE_LISTING_DURATION           = 5
  '''The number of ticks, during which the item is listed for sale'''

  MARKET_N_OBS = 1024
  '''Number of distinct item observations'''

  PRICE_N_OBS = 99 # make it different from PLAYER_N_OBS
  '''Number of distinct price observations
     This also determines the maximum price one can set for an item
  '''


class Communication:
  '''Exchange Game System'''

  COMMUNICATION_SYSTEM_ENABLED         = True
  '''Game system flag'''

  COMMUNICATION_N_OBS                  = 32
  '''Number of players that share the same communication obs, i.e. the same team'''

  COMMUNICATION_NUM_TOKENS             = 127
  '''Number of distinct COMM tokens'''


class AllGameSystems(
  Terrain, Resource, Combat, NPC, Progression, Item,
  Equipment, Profession, Exchange, Communication):
  pass


############################################################################
### Config presets
class Small(Config):
  '''A small config for debugging and experiments with an expensive outer loop'''

  PATH_MAPS                    = 'maps/small'

  PLAYER_N                     = 64

  MAP_PREVIEW_DOWNSCALE        = 4
  MAP_SIZE                     = 64
  MAP_CENTER                   = 32

  TERRAIN_LOG_INTERPOLATE_MIN  = 0

  NPC_N                        = 32
  NPC_LEVEL_MAX                = 5
  NPC_LEVEL_SPREAD             = 1

  PROGRESSION_SPAWN_CLUSTERS   = 4
  PROGRESSION_SPAWN_UNIFORMS   = 16

  HORIZON                      = 128


class Medium(Config):
  '''A medium config suitable for most academic-scale research'''

  PATH_MAPS                    = 'maps/medium'

  PLAYER_N                     = 128

  MAP_PREVIEW_DOWNSCALE        = 16
  MAP_SIZE                     = 160
  MAP_CENTER                   = 128

  NPC_N                        = 128
  NPC_LEVEL_MAX                = 10
  NPC_LEVEL_SPREAD             = 1

  PROGRESSION_SPAWN_CLUSTERS   = 64
  PROGRESSION_SPAWN_UNIFORMS   = 256

  HORIZON                      = 1024


class Large(Config):
  '''A large config suitable for large-scale research or fast models'''

  PATH_MAPS                    = 'maps/large'

  PLAYER_N                     = 1024

  MAP_PREVIEW_DOWNSCALE        = 64
  MAP_SIZE                     = 1056
  MAP_CENTER                   = 1024

  NPC_N                        = 1024
  NPC_LEVEL_MAX                = 15
  NPC_LEVEL_SPREAD             = 3

  PROGRESSION_SPAWN_CLUSTERS   = 1024
  PROGRESSION_SPAWN_UNIFORMS   = 4096

  HORIZON                 = 8192


class Default(Medium, AllGameSystems):
  pass
