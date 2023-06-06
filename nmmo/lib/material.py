
from nmmo.systems import item, droptable

class Material:
  capacity = 0
  tool = None
  table = None
  index = None
  respawn = 0

  def __init__(self, config):
    pass

  def __eq__(self, mtl):
    return self.index == mtl.index

  def __equals__(self, mtl):
    return self == mtl

  def harvest(self):
    return self.__class__.table

class Void(Material):
  tex   = 'void'
  index = 0

class Water(Material):
  tex   = 'water'
  index = 1

  table = droptable.Empty()

  def __init__(self, config):
    self.deplete = __class__
    self.respawn  = 1.0

class Grass(Material):
  tex   = 'grass'
  index = 2

class Scrub(Material):
  tex   = 'scrub'
  index = 3

class Foilage(Material):
  tex   = 'foilage'
  index = 4

  deplete = Scrub
  table = droptable.Empty()

  def __init__(self, config):
    if config.RESOURCE_SYSTEM_ENABLED:
      self.capacity = config.RESOURCE_FOILAGE_CAPACITY
      self.respawn  = config.RESOURCE_FOILAGE_RESPAWN

class Stone(Material):
  tex   = 'stone'
  index = 5

class Slag(Material):
  tex   = 'slag'
  index = 6

class Ore(Material):
  tex   = 'ore'
  index = 7

  deplete = Slag
  tool    = item.Pickaxe

  def __init__(self, config):
    cls = self.__class__
    if cls.table is None:
      cls.table = droptable.Standard()
      cls.table.add(item.Whetstone)

      if config.EQUIPMENT_SYSTEM_ENABLED:
        cls.table.add(item.Wand, prob=config.WEAPON_DROP_PROB)

    if config.PROFESSION_SYSTEM_ENABLED:
      self.capacity = config.PROFESSION_ORE_CAPACITY
      self.respawn  = config.PROFESSION_ORE_RESPAWN

  tool    = item.Pickaxe
  deplete = Slag

class Stump(Material):
  tex   = 'stump'
  index = 8

class Tree(Material):
  tex   = 'tree'
  index = 9

  deplete = Stump
  tool    = item.Axe

  def __init__(self, config):
    cls = self.__class__
    if cls.table is None:
      cls.table = droptable.Standard()
      cls.table.add(item.Arrow)
      if config.EQUIPMENT_SYSTEM_ENABLED:
        cls.table.add(item.Spear, prob=config.WEAPON_DROP_PROB)

    if config.PROFESSION_SYSTEM_ENABLED:
      self.capacity = config.PROFESSION_TREE_CAPACITY
      self.respawn  = config.PROFESSION_TREE_RESPAWN

class Fragment(Material):
  tex   = 'fragment'
  index = 10

class Crystal(Material):
  tex   = 'crystal'
  index = 11

  deplete = Fragment
  tool    = item.Chisel

  def __init__(self, config):
    cls = self.__class__
    if cls.table is None:
      cls.table = droptable.Standard()
      cls.table.add(item.Runes)
      if config.EQUIPMENT_SYSTEM_ENABLED:
        cls.table.add(item.Bow, prob=config.WEAPON_DROP_PROB)

    if config.PROFESSION_SYSTEM_ENABLED:
      self.capacity = config.PROFESSION_CRYSTAL_CAPACITY
      self.respawn  = config.PROFESSION_CRYSTAL_RESPAWN

class Weeds(Material):
  tex   = 'weeds'
  index = 12

class Herb(Material):
  tex   = 'herb'
  index = 13

  deplete = Weeds
  tool    = item.Gloves

  table   = droptable.Standard()
  table.add(item.Potion)

  def __init__(self, config):
    if config.PROFESSION_SYSTEM_ENABLED:
      self.capacity = config.PROFESSION_HERB_CAPACITY
      self.respawn  = config.PROFESSION_HERB_RESPAWN

class Ocean(Material):
  tex   = 'ocean'
  index = 14

class Fish(Material):
  tex   = 'fish'
  index = 15

  deplete = Ocean
  tool    = item.Rod

  table   = droptable.Standard()
  table.add(item.Ration)

  def __init__(self, config):
    if config.PROFESSION_SYSTEM_ENABLED:
      self.capacity = config.PROFESSION_FISH_CAPACITY
      self.respawn  = config.PROFESSION_FISH_RESPAWN

# TODO: Fix lint errors
# pylint: disable=all
class Meta(type):
  def __init__(self, name, bases, dict):
    self.indices = {mtl.index for mtl in self.materials}

  def __iter__(self):
    yield from self.materials

  def __contains__(self, mtl):
    if isinstance(mtl, Material):
      mtl = type(mtl)
    if isinstance(mtl, type):
      return mtl in self.materials
    return mtl in self.indices

class All(metaclass=Meta):
  '''List of all materials'''
  materials = {
    Void, Water, Grass, Scrub, Foilage,
    Stone, Slag, Ore, Stump, Tree,
    Fragment, Crystal, Weeds, Herb, Ocean, Fish}

class Impassible(metaclass=Meta):
  '''Materials that agents cannot walk through'''
  materials = {Void, Water, Stone, Ocean, Fish}

class Habitable(metaclass=Meta):
  '''Materials that agents cannot walk on'''
  materials = {Grass, Scrub, Foilage, Ore, Slag, Tree, Stump, Crystal, Fragment, Herb, Weeds}

class Harvestable(metaclass=Meta):
  '''Materials that agents can harvest'''
  materials = {Water, Foilage, Ore, Tree, Crystal, Herb, Fish}
