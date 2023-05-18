from __future__ import annotations

from typing import Dict

from nmmo.core.agent import Agent
from nmmo.entity.player import Player
from nmmo.lib.log import Logger, MilestoneLogger


class LogHelper:
  @staticmethod
  def create(realm) -> LogHelper:
    if realm.config.LOG_ENV:
      return SimpleLogHelper(realm)
    return DummyLogHelper()

class DummyLogHelper(LogHelper):
  def reset(self) -> None:
    pass

  def update(self, dead_players: Dict[int, Player]) -> None:
    pass

  def log_milestone(self, milestone: str, value: float) -> None:
    pass

  def log_event(self, event: str, value: float) -> None:
    pass

class SimpleLogHelper(LogHelper):
  def __init__(self, realm) -> None:
    self.realm = realm
    self.config = realm.config

    self.reset()

  def reset(self):
    self._env_logger    = Logger()
    self._player_logger = Logger()
    self._event_logger  = DummyLogHelper()
    self._milestone_logger = DummyLogHelper()

    if self.config.LOG_EVENTS:
      self._event_logger = Logger()

    if self.config.LOG_MILESTONES:
      self._milestone_logger = MilestoneLogger(self.config.LOG_FILE)

    self._player_stats_funcs = {}
    self._register_player_stats()

  def log_milestone(self, milestone: str, value: float) -> None:
    if self.config.LOG_MILESTONES:
      self._milestone_logger.log(milestone, value)

  def log_event(self, event: str, value: float) -> None:
    if self.config.LOG_EVENTS:
      self._event_logger.log(event, value)

  @property
  def packet(self):
    packet = {'Env': self._env_logger.stats,
              'Player': self._player_logger.stats}

    if self.config.LOG_EVENTS:
      packet['Event'] = self._event_logger.stats
    else:
      packet['Event'] = 'Unavailable: config.LOG_EVENTS = False'

    if self.config.LOG_MILESTONES:
      packet['Milestone'] = self._event_logger.stats
    else:
      packet['Milestone'] = 'Unavailable: config.LOG_MILESTONES = False'

    return packet

  def _register_player_stat(self, name: str, func: callable):
    assert name not in self._player_stats_funcs
    self._player_stats_funcs[name] = func

  def _register_player_stats(self):
    self._register_player_stat('Basic/TimeAlive', lambda player: player.history.time_alive.val)
    # Skills
    if self.config.PROGRESSION_SYSTEM_ENABLED:
      if self.config.COMBAT_SYSTEM_ENABLED:
        self._register_player_stat('Skill/Mage', lambda player: player.skills.mage.level.val)
        self._register_player_stat('Skill/Range', lambda player: player.skills.range.level.val)
        self._register_player_stat('Skill/Melee', lambda player: player.skills.melee.level.val)
      if self.config.PROFESSION_SYSTEM_ENABLED:
        self._register_player_stat('Skill/Fishing', lambda player: player.skills.fishing.level.val)
        self._register_player_stat('Skill/Herbalism',
          lambda player: player.skills.herbalism.level.val)
        self._register_player_stat('Skill/Prospecting',
          lambda player: player.skills.prospecting.level.val)
        self._register_player_stat('Skill/Carving',
          lambda player: player.skills.carving.level.val)
        self._register_player_stat('Skill/Alchemy',
        lambda player: player.skills.alchemy.level.val)
      if self.config.EQUIPMENT_SYSTEM_ENABLED:
        self._register_player_stat('Item/Held-Level',
          lambda player: player.inventory.equipment.held.item.level.val \
            if player.inventory.equipment.held.item else 0)
        self._register_player_stat('Item/Equipment-Total',
          lambda player: player.equipment.total(lambda e: e.level))

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      self._register_player_stat('Exchange/Player-Sells', lambda player: player.sells)
      self._register_player_stat('Exchange/Player-Buys',  lambda player: player.buys)
      self._register_player_stat('Exchange/Player-Wealth',  lambda player: player.gold.val)

    # Item usage
    if self.config.PROFESSION_SYSTEM_ENABLED:
      self._register_player_stat('Item/Ration-Consumed', lambda player: player.ration_consumed)
      self._register_player_stat('Item/Poultice-Consumed', lambda player: player.poultice_consumed)
      self._register_player_stat('Item/Ration-Level', lambda player: player.ration_level_consumed)
      self._register_player_stat('Item/Poultice-Level',
        lambda player: player.poultice_level_consumed)

  def update(self, dead_players: Dict[int, Player]) -> None:
    for player in dead_players.values():
      for key, val in self._player_stats(player).items():
        self._player_logger.log(key, val)

    # TODO: handle env logging

  def _player_stats(self, player: Agent) -> Dict[str, float]:
    stats = {}
    policy = player.policy

    for key, stat_func in self._player_stats_funcs.items():
      stats[f'{key}_{policy}'] = stat_func(player)

    stats['Time_Alive'] = player.history.time_alive.val

    return stats
