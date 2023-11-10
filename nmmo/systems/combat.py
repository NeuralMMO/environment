#Various utilities for managing combat, including hit/damage

import numpy as np

from nmmo.systems import skill as Skill
from nmmo.lib.event_code import EventCode

def level(skills):
  return max(e.level.val for e in skills.skills)

def damage_multiplier(config, skill, targ):
  skills = [targ.skills.melee, targ.skills.range, targ.skills.mage]
  exp    = [s.exp for s in skills]

  if max(exp) == min(exp):
    return 1.0

  idx    = np.argmax([exp])
  targ   = skills[idx]

  if isinstance(skill, targ.weakness):
    return config.COMBAT_WEAKNESS_MULTIPLIER

  return 1.0

# pylint: disable=unnecessary-lambda-assignment
def attack(realm, attacker, target, skill_fn):
  config       = attacker.config
  skill        = skill_fn(attacker)
  skill_type   = type(skill)
  skill_name   = skill_type.__name__

  # Per-style offense/defense
  level_damage = 0
  if skill_type == Skill.Melee:
    base_damage  = config.COMBAT_MELEE_DAMAGE

    if config.PROGRESSION_SYSTEM_ENABLED:
      base_damage  = config.PROGRESSION_MELEE_BASE_DAMAGE
      level_damage = config.PROGRESSION_MELEE_LEVEL_DAMAGE

    offense_fn   = lambda e: e.melee_attack
    defense_fn   = lambda e: e.melee_defense

  elif skill_type == Skill.Range:
    base_damage  = config.COMBAT_RANGE_DAMAGE

    if config.PROGRESSION_SYSTEM_ENABLED:
      base_damage  = config.PROGRESSION_RANGE_BASE_DAMAGE
      level_damage = config.PROGRESSION_RANGE_LEVEL_DAMAGE

    offense_fn   = lambda e: e.range_attack
    defense_fn   = lambda e: e.range_defense

  elif skill_type == Skill.Mage:
    base_damage  = config.COMBAT_MAGE_DAMAGE

    if config.PROGRESSION_SYSTEM_ENABLED:
      base_damage  = config.PROGRESSION_MAGE_BASE_DAMAGE
      level_damage = config.PROGRESSION_MAGE_LEVEL_DAMAGE

    offense_fn   = lambda e: e.mage_attack
    defense_fn   = lambda e: e.mage_defense

  elif __debug__:
    assert False, 'Attack skill must be Melee, Range, or Mage'

  # Compute modifiers
  multiplier        = damage_multiplier(config, skill, target)

  # NOTE: skill offense and defense are only for agents, NOT npcs
  skill_offense = base_damage
  if attacker.is_player:
    skill_offense += level_damage * skill.level.val
  if attacker.is_npc and config.EQUIPMENT_SYSTEM_ENABLED:
    # NOTE: In this case, npc off/def is set only with equipment. Revisit this.
    skill_offense = 0

  if config.PROGRESSION_SYSTEM_ENABLED and target.is_player:
    skill_defense = config.PROGRESSION_BASE_DEFENSE  + \
      config.PROGRESSION_LEVEL_DEFENSE*level(target.skills)
  else:
    skill_defense = 0

  if config.EQUIPMENT_SYSTEM_ENABLED:
    equipment_offense = attacker.equipment.total(offense_fn)
    equipment_defense = target.equipment.total(defense_fn)

    # after tallying ammo damage, consume ammo (i.e., fire) when the skill type matches
    ammunition = attacker.equipment.ammunition.item
    if ammunition is not None and getattr(ammunition, skill_name.lower() + '_attack').val > 0:
      ammunition.fire(attacker)

  else:
    equipment_offense = 0
    equipment_defense = 0

  # Total damage calculation
  offense = skill_offense + equipment_offense
  defense = skill_defense + equipment_defense
  min_damage_prop = config.COMBAT_MINIMUM_DAMAGE_PROPORTION
  damage  = config.COMBAT_DAMAGE_FORMULA(offense, defense, multiplier, min_damage_prop)
  damage  = max(int(damage), 0)

  if attacker.is_player:
    realm.event_log.record(EventCode.SCORE_HIT, attacker, target=target,
                           combat_style=skill_type, damage=damage)

  attacker.apply_damage(damage, skill.__class__.__name__.lower())
  target.receive_damage(attacker, damage)

  return damage


def danger(config, pos):
  border = config.MAP_BORDER
  center = config.MAP_CENTER
  r, c   = pos

  #Distance from border
  r_dist  = min(r - border, center + border - r - 1)
  c_dist  = min(c - border, center + border - c - 1)
  dist   = min(r_dist, c_dist)
  norm   = 2 * dist / center

  return norm

def spawn(config, dnger, np_random):
  border = config.MAP_BORDER
  center = config.MAP_CENTER
  mid    = center // 2

  dist       = dnger * center / 2
  max_offset = mid - dist
  offset     = mid + border + np_random.integers(-max_offset, max_offset)

  rng = np_random.random()
  if rng < 0.25:
    r = border + dist
    c = offset
  elif rng < 0.5:
    r = border + center - dist - 1
    c = offset
  elif rng < 0.75:
    c = border + dist
    r = offset
  else:
    c = border + center - dist - 1
    r = offset

  if __debug__:
    assert dnger == danger(config, (r,c)), 'Agent spawned at incorrect radius'

  r = int(r)
  c = int(c)

  return r, c
