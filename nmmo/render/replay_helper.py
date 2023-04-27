import json
import lzma
import logging

from .render_utils import np_encoder, patch_packet


class ReplayHelper:
  @staticmethod
  def create(realm):
    if realm.config.SAVE_REPLAY:
      return ReplayFileHelper(realm)

    return DummyReplayHelper()


class DummyReplayHelper(ReplayHelper):
  def reset(self):
    pass

  def update(self):
    pass

  def save(self, save_path, compress):
    pass


class ReplayFileHelper(ReplayHelper):
  def __init__(self, realm=None):
    self._realm = realm
    self.packets = None
    self.map = None
    self._i = 0

  def reset(self):
    self.packets = []
    self.map = None
    self._i = 0

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

  def update(self, packet=None):
    if packet is None:
      if self._realm is None:
        return
      # TODO: patch_packet is a hack. best to remove, if possible
      packet = patch_packet(self._realm.packet(), self._realm)

    data = {}
    for key, val in packet.items():
      if key == 'environment':
        self.map = val
        continue
      if key == 'config':
        continue
      data[key] = val

    self.packets.append(data)

  def save(self, save_file, compress=True):
    logging.info('Saving replay to %s ...', save_file)

    data = {
      'map': self.map,
      'packets': self.packets }

    data = json.dumps(data, default=np_encoder).encode('utf8')
    if compress:
      data = lzma.compress(data, format=lzma.FORMAT_ALONE)

    with open(save_file, 'wb') as out:
      out.write(data)

  @classmethod
  def load(cls, replay_file, decompress=True):
    with open(replay_file, 'rb') as fp:
      data = fp.read()

    if decompress:
      data = lzma.decompress(data, format=lzma.FORMAT_ALONE)
    data = json.loads(data.decode('utf-8'))

    replay_helper = ReplayFileHelper()
    replay_helper.map = data['map']
    replay_helper.packets = data['packets']

    return replay_helper
