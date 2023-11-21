# pylint: disable=invalid-name, unused-argument
import numpy as np

import nmmo
from nmmo.core.observation import Observation
from nmmo.entity.entity import EntityState
from nmmo.lib import utils


def closestTarget(config, ob: Observation):
  shortestDist = np.inf
  closestAgent = None

  agent  = ob.agent
  start = (agent.row, agent.col)

  for target_ent in ob.entities.values:
    target_ent = EntityState.parse_array(target_ent)
    if target_ent.id == agent.id:
      continue

    dist = utils.linf_single(start, (target_ent.row, target_ent.col))
    if dist < shortestDist and dist != 0:
      shortestDist = dist
      closestAgent = target_ent

  if closestAgent is None:
    return None, None

  return closestAgent, shortestDist

def attacker(config, ob: Observation):
  agent = ob.agent

  attacker_id = agent.attacker_id
  if attacker_id == 0:
    return None, None

  target_ent = ob.entity(attacker_id)
  if target_ent is None:
    return None, None

  return target_ent,\
         utils.linf_single((agent.row, agent.col), (target_ent.row, target_ent.col))

def target(config, actions, style, targetID):
  actions[nmmo.action.Attack] = {
        nmmo.action.Style: style,
        nmmo.action.Target: targetID}
