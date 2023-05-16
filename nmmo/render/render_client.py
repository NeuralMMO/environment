from __future__ import annotations
import numpy as np

from nmmo.render import websocket
from nmmo.render.overlay import OverlayRegistry
from nmmo.render.render_utils import patch_packet


# Render is external to the game
class WebsocketRenderer:
  def __init__(self, realm=None) -> None:
    self._client = websocket.Application(realm)
    self.overlay_pos = [256, 256]

    self._realm = realm

    self.overlay = None
    self.registry = OverlayRegistry(realm, renderer=self) if realm else None

    self.packet = None

  def render_packet(self, packet) -> None:
    packet = {
      'pos': self.overlay_pos,
      'wilderness': 0, # obsolete, but maintained for compatibility
      **packet }

    self.overlay_pos, _ = self._client.update(packet)

  def render_realm(self) -> None:
    assert self._realm is not None, 'This function requires a realm'
    assert self._realm.tick is not None, 'render before reset'

    packet = {
      'config': self._realm.config,
      'pos': self.overlay_pos,
      'wilderness': 0,
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
