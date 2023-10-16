class EventCode:
  # Move
  EAT_FOOD = 1
  DRINK_WATER = 2
  GO_FARTHEST = 3 # record when breaking the previous record

  # Attack
  SCORE_HIT = 11
  PLAYER_KILL = 12

  # Item
  CONSUME_ITEM = 21
  GIVE_ITEM = 22
  DESTROY_ITEM = 23
  HARVEST_ITEM = 24
  EQUIP_ITEM = 25
  LOOT_ITEM = 26

  # Exchange
  GIVE_GOLD = 31
  LIST_ITEM = 32
  EARN_GOLD = 33
  BUY_ITEM = 34
  #SPEND_GOLD = 35 # BUY_ITEM, price has the same info

  # Level up
  LEVEL_UP = 41
