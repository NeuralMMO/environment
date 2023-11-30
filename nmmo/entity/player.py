from nmmo.systems.skill import Skills
from nmmo.entity import entity
from nmmo.lib.event_code import EventCode
from nmmo.lib import spawn

# pylint: disable=no-member
class Player(entity.Entity):
  def __init__(self, realm, pos, agent, resilient=False):
    super().__init__(realm, pos, agent.iden, agent.policy)

    self.agent    = agent
    self._immortal = realm.config.IMMORTAL
    self.resources.resilient = resilient
    self.my_task = None
    self._make_mortal_tick = None  # set to realm.tick when the player is made mortal

    # Scripted hooks
    self.target = None
    self.vision = 7

    # Logs
    self.buys                     = 0
    self.sells                    = 0
    self.ration_consumed          = 0
    self.poultice_consumed        = 0
    self.ration_level_consumed    = 0
    self.poultice_level_consumed  = 0

    # initialize skills with the base level
    self.skills = Skills(realm, self)
    if realm.config.PROGRESSION_SYSTEM_ENABLED:
      for skill in self.skills.skills:
        skill.level.update(realm.config.PROGRESSION_BASE_LEVEL)

    # Gold: initialize with 1 gold (EXCHANGE_BASE_GOLD).
    # If the base amount is more than 1, alss check the npc's init gold.
    if realm.config.EXCHANGE_SYSTEM_ENABLED:
      self.gold.update(realm.config.EXCHANGE_BASE_GOLD)

  @property
  def serial(self):
    return self.ent_id

  @property
  def is_player(self) -> bool:
    return True

  @property
  def level(self) -> int:
    # a player's level is the max of all skills
    # CHECK ME: the initial level is 1 because of Basic skills,
    #   which are harvesting food/water and don't progress
    return max(e.level.val for e in self.skills.skills)

  def _set_immortal(self, value=True, duration=None):
    self._immortal = value
    # NOTE: a hack to mark the player as immortal in action targets
    self.npc_type.update(-1 if value else 0)

    if value and duration is not None:
      self._make_mortal_tick = self.realm.tick + duration
    if value is False:
      self._make_mortal_tick = None

  def make_recon(self, new_pos=None):
    # NOTE: scout cannot act and cannot die
    self.status.freeze.update(self.config.MAX_HORIZON)
    self._set_immortal()
    self._recon = True
    if new_pos is not None:
      if self.ent_id in self.realm.map.tiles[self.pos].entities:
        self.realm.map.tiles[self.pos].remove_entity(self.ent_id)
      self.realm.map.tiles[new_pos].add_entity(self)
      self.set_pos(*new_pos)

  def apply_damage(self, dmg, style):
    super().apply_damage(dmg, style)
    self.skills.apply_damage(style)

  # TODO(daveey): The returns for this function are a mess
  def receive_damage(self, source, dmg):
    if self.immortal:
      return False

    # super().receive_damage returns True if self is alive after taking dmg
    if super().receive_damage(source, dmg):
      return True

    if not self.config.ITEM_SYSTEM_ENABLED:
      return False

    # starting from here, source receive gold & inventory items
    if self.config.EXCHANGE_SYSTEM_ENABLED and source is not None:
      if self.gold.val > 0:
        source.gold.increment(self.gold.val)
        self.realm.event_log.record(EventCode.LOOT_GOLD, source, amount=self.gold.val, target=self)
        self.gold.update(0)

    # TODO: make source receive the highest-level items first
    #   because source cannot take it if the inventory is full
    item_list = list(self.inventory.items)
    self._np_random.shuffle(item_list)
    for item in item_list:
      self.inventory.remove(item)

      # if source is None or NPC, destroy the item
      if source.is_player:
        # inventory.receive() returns True if the item is received
        # if source doesn't have space, inventory.receive() destroys the item
        if source.inventory.receive(item):
          self.realm.event_log.record(EventCode.LOOT_ITEM, source, item=item, target=self)
      else:
        item.destroy()

    # CHECK ME: this is an empty function. do we still need this?
    self.skills.receive_damage(dmg)
    return False

  @property
  def equipment(self):
    return self.inventory.equipment

  def packet(self):
    data = super().packet()
    data['entID']     = self.ent_id
    data['resource']  = self.resources.packet()
    data['skills']    = self.skills.packet()
    data['inventory'] = self.inventory.packet()
    # added for the 2.0 web client
    data["metrics"] = {
      "PlayerDefeats": self.history.player_kills,
      "TimeAlive": self.time_alive.val,
      "Gold": self.gold.val,
      "DamageTaken": self.history.damage_received,}
    return data

  def update(self, realm, actions):
    '''Post-action update. Do not include history'''
    super().update(realm, actions)

    # Spawn battle royale style death fog
    # Starts at 0 damage on the specified config tick
    # Moves in from the edges by 1 damage per tile per tick
    # So after 10 ticks, you take 10 damage at the edge and 1 damage
    # 10 tiles in, 0 damage in farther
    # This means all agents will be force killed around
    # MAP_CENTER / 2 + 100 ticks after spawning
    fog = self.config.DEATH_FOG_ONSET
    if fog is not None and self.realm.tick >= fog:
      dmg = self.realm.fog_map[self.pos]
      if dmg > 0.5:  # fog_map has float values
        self.receive_damage(None, round(dmg))

    if not self.alive:
      return

    if self.config.PLAYER_HEALTH_INCREMENT > 0:
      self.resources.health.increment(self.config.PLAYER_HEALTH_INCREMENT)
    self.resources.update(self.immortal)
    self.skills.update()

    if self._make_mortal_tick is not None and self.realm.tick >= self._make_mortal_tick:
      self._set_immortal(False)

  def resurrect(self, health_prop=0.5, freeze_duration=10, edge_spawn=True):
    # Respawn dead players at the edge
    assert not self.alive, "Player is not dead"
    self.status.freeze.update(freeze_duration)
    self.resources.health.update(self.config.PLAYER_BASE_HEALTH*health_prop)
    if self.config.RESOURCE_SYSTEM_ENABLED:
      self.resources.water.update(self.config.RESOURCE_BASE)
      self.resources.food.update(self.config.RESOURCE_BASE)

    if edge_spawn:
      new_spawn_pos = spawn.get_random_coord(self.config, self._np_random, edge=True)
    else:
      while True:
        new_spawn_pos = spawn.get_random_coord(self.config, self._np_random, edge=False)
        if self.realm.map.tiles[new_spawn_pos].habitable:
          break

    self.set_pos(*new_spawn_pos)
    self.message.update(0)
    self.realm.players.spawn_entity(self)  # put back to the system
    self._set_immortal(duration=freeze_duration)
    if self.my_task:
      # NOTE: Only one task per agent is supported for now
      self.my_task.reset()  # Agent's task progress need to be reset
