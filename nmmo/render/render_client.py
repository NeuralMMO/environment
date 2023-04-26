from __future__ import annotations
import numpy as np

from nmmo.render.websocket import Application
from nmmo.render.overlay import OverlayRegistry
from nmmo.render.render_utils import patch_packet


# Render is external to the game
class WebsocketRenderer:
  def __init__(self):
    # CHECK ME: It seems the renderer works fine without realm
    self._client = Application(realm=None) # Do we need to pass realm?
    self.overlay_pos = [256, 256]

  def render(self, packet):
    packet = {
      'pos': self.overlay_pos,
      'wilderness': 0,
      **packet }

    pos, _ = self._client.update(packet)
    self.overlay_pos = pos


class OnlineRenderer(WebsocketRenderer):
  def __init__(self, env) -> None:
    super().__init__()
    self._realm = env.realm
    self._config = env.config

    self.overlay    = None
    self.registry   = OverlayRegistry(env.realm, renderer=self)

    self.packet = None

  ############################################################################
  ### Client data
  def render(self) -> None:
    '''Data packet used by the renderer

    Returns:
        packet: A packet of data for the client
    '''
    assert self._realm.tick is not None, 'render before reset'

    packet = {
      'config': self._config,
      'pos': self.overlay_pos,
      'wilderness': 0, # CHECK ME: what is this? copy pasted from the old version
      **self._realm.packet()
    }

    # TODO: a hack to make the client work
    packet = patch_packet(packet, self._realm)

    if self.overlay is not None:
      packet['overlay'] = self.overlay
      self.overlay = None

    # save the packet for investigation
    self.packet = packet

    # pass the packet to renderer
    pos, cmd = self._client.update(self.packet)

    self.overlay_pos = pos
    self.registry.step(cmd)

  def register(self, overlay: np.ndarray) -> None:
    '''Register an overlay to be sent to the client

    The intended use of this function is: User types overlay ->
    client sends cmd to server -> server computes overlay update ->
    register(overlay) -> overlay is sent to client -> overlay rendered

    Args:
        overlay: A map-sized (self.size) array of floating point values
        overlay must be a numpy array of dimension (*(env.size), 3)
    '''
    self.overlay = overlay.tolist()
