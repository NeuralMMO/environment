import numpy as np

class Fixed():
  def __init__(self, item):
    self.item = item

  def roll(self, realm, level):
    return [self.item(realm, level)]

class Drop:
  def __init__(self, item, prob):
    self.item = item
    self.prob = prob

  def roll(self, realm, level):
    if np.random.rand() < self.prob:
      return self.item(realm, level)

    return None

class Standard:
  def __init__(self):
    self.drops = []

  def add(self, item, prob=1.0):
    self.drops += [Drop(item, prob)]

  def roll(self, realm, level):
    ret = []
    for e in self.drops:
      drop = e.roll(realm, level)
      if drop is not None:
        ret += [drop]
    return ret

class Empty(Standard):
  def roll(self, realm, level):
    return []

class Ammunition(Standard):
  def __init__(self, item):
    super().__init__()
    self.item = item

  def roll(self, realm, level):
    return [self.item(realm, level)]

class Consumable(Standard):
  def __init__(self, item):
    super().__init__()
    self.item = item

  def roll(self, realm, level):
    return [self.item(realm, level)]
