import os
import json
import logging
import lzma
import pickle
from typing import Dict

from .render_utils import np_encoder, patch_packet

class ReplayHelper:
  def __init__(self):
    self._realm = None

  def set_realm(self, realm) -> None:
    self._realm = realm

  def reset(self):
    pass

  def update(self):
    pass

  def save(self, filename_prefix, compress):
    pass

class DummyReplayHelper(ReplayHelper):
  pass

class FileReplayHelper(ReplayHelper):
  def __init__(self, realm=None):
    super().__init__()
    self._realm = realm
    self.packets = None
    self.map = None
    self._i = 0

  def reset(self):
    self.packets = []
    self.map = None
    self._i = 0
    self.update() # to capture the initial packet

  def __len__(self):
    return len(self.packets)

  def __iter__(self):
    self._i = 0
    return self

  def __next__(self):
    if self._i >= len(self.packets):
      raise StopIteration
    packet = self.packets[self._i]
    packet['environment'] = self.map
    self._i += 1
    return packet

  def _packet(self):
    assert self._realm is not None, 'Realm not set'

    # TODO: remove patch_packet
    packet = patch_packet(self._realm.packet(), self._realm)

    if "environment" in packet:
      self.map = packet["environment"]
      del packet["environment"]
    if "config" in packet:
      del packet["config"]

    # Include the attributes that the web client refers
    packet["config"] = {
      "PLAYER_DEATH_FOG": self._realm.config.PLAYER_DEATH_FOG,
      "PLAYER_DEATH_FOG_FINAL_SIZE": self._realm.config.PLAYER_DEATH_FOG_FINAL_SIZE,
      "PLAYER_DEATH_FOG_SPEED": self._realm.config.PLAYER_DEATH_FOG_SPEED,
    }

    return packet

  def _metadata(self) -> Dict:
    return {
      'event_log': self._realm.event_log.get_data(),
      'event_attr_col': self._realm.event_log.attr_to_col
    }

  def update(self):
    self.packets.append(self._packet())

  def save(self, filename_prefix, compress=False):
    replay_file = f'{filename_prefix}.replay.json'
    metadata_file = f'{filename_prefix}.metadata.pkl'

    data = json.dumps({
      'map': self.map,
      'packets': self.packets
    }, default=np_encoder).encode('utf8')

    if compress:
      replay_file = f'{filename_prefix}.replay.lzma'
      data = lzma.compress(data, format=lzma.FORMAT_ALONE)

    logging.info('Saving replay to %s ...', replay_file)

    with open(replay_file, 'wb') as out:
      out.write(data)

    with open(metadata_file, 'wb') as out:
      pickle.dump(self._metadata(), out)

  @classmethod
  def load(cls, replay_file):
    extension = os.path.splitext(replay_file)[1]
    with open(replay_file, 'rb') as fp:
      data = fp.read()

    if extension != '.json':
      data = lzma.decompress(data, format=lzma.FORMAT_ALONE)
    data = json.loads(data.decode('utf-8'))

    replay_helper = FileReplayHelper()
    replay_helper.map = data['map']
    replay_helper.packets = data['packets']

    return replay_helper
