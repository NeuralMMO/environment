# pylint: disable=all

import numpy as np

import nmmo
from nmmo.core.observation import Observation
from nmmo.entity.entity import EntityState

from scripted import utils

def closestTarget(config, ob: Observation):
  shortestDist = np.inf
  closestAgent = None

  agent  = ob.agent()

  start = (agent.row, agent.col)

  for target in ob.entities.values:
    target = EntityState.parse_array(target)
    if target.id == agent.id:
      continue

    dist = utils.l1(start, (target.row, target.col))

    if dist < shortestDist and dist != 0:
      shortestDist = dist
      closestAgent = target

  if closestAgent is None:
    return None, None

  return closestAgent, shortestDist

def attacker(config, ob: Observation):
  agent = ob.agent()

  attacker_id = agent.attacker_id

  if attacker_id == 0:
    return None, None

  target = ob.entity(attacker_id)
  if target == None:
    return None, None
      
  return target, utils.l1((agent.row, agent.col), (target.row, target.col))

def target(config, actions, style, targetID):
  actions[nmmo.action.Attack] = {
        nmmo.action.Style: style,
        nmmo.action.Target: targetID}

