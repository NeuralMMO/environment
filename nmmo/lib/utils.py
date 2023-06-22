# pylint: disable=all

import inspect
from collections import deque

import numpy as np


class staticproperty(property):
  def __get__(self, cls, owner):
    return self.fget.__get__(None, owner)()

class classproperty(object):
  def __init__(self, f):
    self.f = f
  def __get__(self, obj, owner):
    return self.f(owner)

class Iterable(type):
  def __iter__(cls):
    queue = deque(cls.__dict__.items())
    while len(queue) > 0:
      name, attr = queue.popleft()
      if type(name) != tuple:
        name = tuple([name])
      if not inspect.isclass(attr):
        continue
      yield name, attr

  def values(cls):
    return [e[1] for e in cls]

class StaticIterable(type):
  def __iter__(cls):
    stack = list(cls.__dict__.items())
    stack.reverse()
    for name, attr in stack:
      if name == '__module__':
        continue
      if name.startswith('__'):
        break
      yield name, attr

class NameComparable(type):
  def __hash__(self):
    return hash(self.__name__)

  def __eq__(self, other):
    return self.__name__ == other.__name__

  def __ne__(self, other):
    return self.__name__ != other.__name__

  def __lt__(self, other):
    return self.__name__ < other.__name__

  def __le__(self, other):
    return self.__name__ <= other.__name__

  def __gt__(self, other):
    return self.__name__ > other.__name__

  def __ge__(self, other):
    return self.__name__ >= other.__name__

class IterableNameComparable(Iterable, NameComparable):
  pass

def linf(pos1, pos2):
  # pos could be a single (r,c) or a vector of (r,c)s
  diff = np.abs(np.array(pos1) - np.array(pos2))
  return np.max(diff, axis=-1)

#Bounds checker
def in_bounds(r, c, shape, border=0):
  R, C = shape
  return (
    r > border and
    c > border and
    r < R - border and
    c < C - border
  )

