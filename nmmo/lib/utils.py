# pylint: disable=all
import inspect
from collections import deque
import hashlib
import numpy as np
from nmmo.entity.entity import Entity, EntityState

EntityAttr = EntityState.State.attr_name_to_col

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

def linf_single(pos1, pos2):
  # pos is a single (r,c) to avoid uneccessary function calls
  return max(abs(pos1[0]-pos2[0]), abs(pos1[1]-pos2[1]))

#Bounds checker
def in_bounds(r, c, shape, border=0):
  R, C = shape
  return (
    r > border and
    c > border and
    r < R - border and
    c < C - border
  )

def l1_map(size):
  # l1 distance from the center tile (size//2, size//2)
  x      = np.abs(np.arange(size) - size//2)
  X, Y   = np.meshgrid(x, x)
  data   = np.stack((X, Y), -1)
  return np.max(abs(data), -1)

def get_hash_embedding(func, embed_dim):
  # NOTE: This is a hacky way to get a hash embedding for a function
  # TODO: Can we get more meaningful embedding? coding LLMs are good but huge
  func_src = inspect.getsource(func)
  hash_object = hashlib.sha256(func_src.encode())
  hex_digest = hash_object.hexdigest()

  # Convert the hexadecimal hash to a numpy array with float16 data type
  hash_bytes = bytes.fromhex(hex_digest)
  hash_array = np.frombuffer(hash_bytes, dtype=np.float16)
  hash_array = np.nan_to_num(hash_array, nan=0, posinf=0, neginf=0)
  hash_array = np.resize(hash_array, (embed_dim,))
  # normalize
  hash_array -= hash_array.astype(np.float).mean()
  hash_array /= hash_array.astype(np.float).std()
  return hash_array

def identify_closest_target(entity):
  realm = entity.realm
  radius = realm.config.PLAYER_VISION_RADIUS
  visible_entities = Entity.Query.window(
    realm.datastore, entity.pos[0], entity.pos[1], radius)
  dist = linf(visible_entities[:,EntityAttr["row"]:EntityAttr["col"]+1], entity.pos)
  entity_ids = visible_entities[:,EntityAttr["id"]]

  # Filter out the entities that are not attackable
  flt_idx = visible_entities[:,EntityAttr["npc_type"]] >= 0  # no immortal (-1)
  if entity.config.NPC_SYSTEM_ENABLED and not entity.config.NPC_ALLOW_ATTACK_OTHER_NPCS:
    flt_idx &= entity_ids > 0
  dist = dist[flt_idx]
  entity_ids = entity_ids[flt_idx]

  # TODO: this could be made smarter/faster, or perhaps consider health
  if len(dist) > 1:
    closest_idx = np.argmin(dist)
    return realm.entity(entity_ids[closest_idx])
  if len(dist) == 1:
    return realm.entity(entity_ids[0])
  return None
