import numpy as np

from nmmo.lib.colors import Neon
from nmmo.systems import combat

from .render_utils import normalize

# pylint: disable=unused-argument
class OverlayRegistry:
  def __init__(self, realm, renderer):
    '''Manager class for overlays

    Args:
        config: A Config object
        realm: An environment
    '''
    self.initialized = False

    self.realm  = realm
    self.config = realm.config
    self.renderer = renderer

    self.overlays = {
       #'counts':     Counts, # TODO: change population to team
       'skills':     Skills}

  def init(self, *args):
    self.initialized = True
    for cmd, overlay in self.overlays.items():
      self.overlays[cmd] = overlay(self.config, self.realm, self.renderer, *args)
    return self

  def step(self, cmd):
    '''Per-tick overlay updates

    Args:
        cmd: User command returned by the client
    '''
    if not self.initialized:
      self.init()

    for overlay in self.overlays.values():
      overlay.update()

    if cmd in self.overlays:
      self.overlays[cmd].register()


class Overlay:
  '''Define a overlay for visualization in the client

  Overlays are color images of the same size as the game map.
  They are rendered over the environment with transparency and
  can be used to gain insight about agent behaviors.'''
  def __init__(self, config, realm, renderer, *args):
    '''
    Args:
        config: A Config object
        realm: An environment
    '''
    self.config     = config
    self.realm      = realm
    self.renderer   = renderer

    self.size       = config.MAP_SIZE
    self.values     = np.zeros((self.size, self.size))

  def update(self):
    '''Compute per-tick updates to this overlay. Override per overlay.

    Args:
        obs: Observation returned by the environment
    '''

  def register(self):
    '''Compute the overlay and register it within realm. Override per overlay.'''


class Skills(Overlay):
  def __init__(self, config, realm, renderer, *args):
    '''Indicates whether agents specialize in foraging or combat'''
    super().__init__(config, realm, renderer)
    self.num_skill = 2

    self.values  = np.zeros((self.size, self.size, self.num_skill))

  def update(self):
    '''Computes a count-based exploration map by painting
    tiles as agents walk over them'''
    for agent in self.realm.players.values():
      r, c = agent.pos

      skill_lvl  = (agent.skills.food.level.val + agent.skills.water.level.val)/2.0
      combat_lvl = combat.level(agent.skills)

      if skill_lvl == 10 and combat_lvl == 3:
        continue

      self.values[r, c, 0] = skill_lvl
      self.values[r, c, 1] = combat_lvl

  def register(self):
    values = np.zeros((self.size, self.size, self.num_skill))
    for idx in range(self.num_skill):
      ary  = self.values[:, :, idx]
      vals = ary[ary != 0]
      mean = np.mean(vals)
      std  = np.std(vals)
      if std == 0:
        std = 1

      values[:, :, idx] = (ary - mean) / std
      values[ary == 0] = 0

    colors    = np.array([Neon.BLUE.rgb, Neon.BLOOD.rgb])
    colorized = np.zeros((self.size, self.size, 3))
    amax      = np.argmax(values, -1)

    for idx in range(self.num_skill):
      colorized[amax == idx] = colors[idx] / 255
      colorized[values[:, :, idx] == 0] = 0

    self.renderer.register(colorized)


# CHECK ME: this was based on population, so disabling it for now
#   We may want this back for the team-level analysis
class Counts(Overlay):
  def __init__(self, config, realm, renderer, *args):
    super().__init__(config, realm, renderer)
    self.values = np.zeros((self.size, self.size, config.PLAYER_POLICIES))

  def update(self):
    '''Computes a count-based exploration map by painting
    tiles as agents walk over them'''
    for ent_id, agent in self.realm.players.items():
      r, c = agent.pos
      self.values[r, c][ent_id] += 1

  def register(self):
    colors    = self.realm.players.palette.colors
    colors    = np.array([colors[pop].rgb
                          for pop in range(self.config.PLAYER_POLICIES)])

    colorized = self.values[:, :, :, None] * colors / 255
    colorized = np.sum(colorized, -2)
    count_sum  = np.sum(self.values[:, :], -1)
    data      = normalize(count_sum)[..., None]

    count_sum[count_sum==0] = 1
    colorized = colorized * data / count_sum[..., None]

    self.renderer.register(colorized)
