#pylint: disable=protected-access, invalid-name

import numpy as np

import nmmo
from nmmo.systems.ai import move, utils


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


def pathfind(realm, actions, entity, target):
  # TODO: do not access realm._np_random directly. ALSO see below for all other uses
  actions[nmmo.action.Move] = {
    nmmo.action.Direction: move.pathfind(realm.map, entity, target, realm._np_random)}


def explore(realm, actions, entity):
  sz = realm.config.TERRAIN_SIZE
  r, c = entity.pos

  spawnR, spawnC = entity.spawnPos
  centR, centC = sz//2, sz//2

  vR, vC = centR-spawnR, centC-spawnC

  mmag = max(abs(vR), abs(vC))
  rr = r + int(np.round(entity.vision*vR/mmag))
  cc = c + int(np.round(entity.vision*vC/mmag))

  tile = realm.map.tiles[rr, cc]
  pathfind(realm, actions, entity, tile)


def meander(realm, actions, entity):
  actions[nmmo.action.Move] = {
    nmmo.action.Direction: move.habitable(realm.map, entity, realm._np_random)}


def evade(realm, actions, entity):
  actions[nmmo.action.Move] = {
    nmmo.action.Direction: move.antipathfind(realm.map, entity, entity.attacker,
                                                realm._np_random)}


def hunt(realm, actions, entity):
  # Move args
  distance = utils.lInfty(entity.pos, entity.target.pos)

  if distance > 1:
    actions[nmmo.action.Move] = {nmmo.action.Direction: move.pathfind(realm.map,
                                                                      entity,
                                                                      entity.target,
                                                                      realm._np_random)}
  elif distance == 0:
    actions[nmmo.action.Move] = {
        nmmo.action.Direction: move.random_direction(realm._np_random)}

  attack(realm, actions, entity)

def attack(realm, actions, entity):
  distance = utils.lInfty(entity.pos, entity.target.pos)
  if distance > entity.skills.style.attack_range(realm.config):
    return

  actions[nmmo.action.Attack] = {
    nmmo.action.Style: entity.skills.style,
    nmmo.action.Target: entity.target}
