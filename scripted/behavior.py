# pylint: disable=all

import nmmo
from nmmo.systems.ai import move, attack, utils

def update(entity):
   '''Update validity of tracked entities'''
   if not utils.validTarget(entity, entity.attacker, entity.vision):
      entity.attacker = None
   if not utils.validTarget(entity, entity.target, entity.vision):
      entity.target = None
   if not utils.validTarget(entity, entity.closest, entity.vision):
      entity.closest = None

   if entity.__class__.__name__ != 'Player':
      return

   if not utils.validResource(entity, entity.food, entity.vision):
      entity.food = None
   if not utils.validResource(entity, entity.water, entity.vision):
      entity.water = None

def pathfind(config, ob, actions, rr, cc):
   actions[nmmo.action.Move]   = {nmmo.action.Direction: move.pathfind(config, ob, actions, rr, cc)}

def meander(realm, actions, entity):
   actions[nmmo.action.Move] = {nmmo.action.Direction: move.habitable(realm.map.tiles, entity)}

def evade(realm, actions, entity):
   actions[nmmo.action.Move] = {nmmo.action.Direction: move.antipathfind(realm.map.tiles, entity, entity.attacker)}

def hunt(realm, actions, entity):
   #Move args
   distance = utils.distance(entity, entity.target)

   direction = None
   if distance == 0:
      direction = move.random_direction()
   elif distance > 1:
      direction = move.pathfind(realm.map.tiles, entity, entity.target)

   if direction is not None:
      actions[nmmo.action.Move] = {nmmo.action.Direction: direction}

   attack(realm, actions, entity)

def attack(realm, actions, entity):
   distance = utils.distance(entity, entity.target)
   if distance > entity.skills.style.attack_range(realm.config):
      return

   actions[nmmo.action.Attack] = {nmmo.action.Style: entity.skills.style,
         nmmo.action.Target: entity.target}

def forageDP(realm, actions, entity):
   direction                 = utils.forageDP(realm.map.tiles, entity)
   actions[nmmo.action.Move] = {nmmo.action.Direction: move.towards(direction)}

#def forageDijkstra(realm, actions, entity):
def forageDijkstra(config, ob, actions, food_max, water_max):
   direction                 = utils.forageDijkstra(config, ob, food_max, water_max)
   actions[nmmo.action.Move] = {nmmo.action.Direction: move.towards(direction)}
