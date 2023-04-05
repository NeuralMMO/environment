from collections import defaultdict

import logging


class Logger:
  def __init__(self):
    self.stats = defaultdict(list)

  def log(self, key, val):
    if not isinstance(val, (int, float)):
      raise RuntimeError(f'{val} must be int or float')

    self.stats[key].append(val)
    return True

class MilestoneLogger(Logger):
  def __init__(self, log_file):
    super().__init__()
    logging.basicConfig(format='%(levelname)s:%(message)s',
      level=logging.INFO, filename=log_file, filemode='w')

  def log_min(self, key, val):
    if key in self.stats and val >= self.stats[key][-1]:
      return False

    self.log(key, val)
    return True

  def log_max(self, key, val):
    if key in self.stats and val <= self.stats[key][-1]:
      return False

    self.log(key, val)
    return True


# CHECK ME: Is this a good place to put here?
#   EventCode is used in many places, and I(kywch)'m putting it here
#   to avoid a circular import, which happened a few times with event_log.py
class EventCode:
  # Move
  EAT_FOOD = 1
  DRINK_WATER = 2

  # Attack
  SCORE_HIT = 11
  PLAYER_KILL = 12

  # Item
  CONSUME_ITEM = 21
  GIVE_ITEM = 22
  DESTROY_ITEM = 23
  HARVEST_ITEM = 24

  # Exchange
  GIVE_GOLD = 31
  LIST_ITEM = 32
  EARN_GOLD = 33
  BUY_ITEM = 34
  #SPEND_GOLD = 35 # BUY_ITEM, price has the same info

  # Level up
  LEVEL_UP = 41
