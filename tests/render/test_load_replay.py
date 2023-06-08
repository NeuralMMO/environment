'''Manual test for rendering replay'''

if __name__ == '__main__':
  import time

  # pylint: disable=import-error
  from nmmo.render.render_client import WebsocketRenderer
  from nmmo.render.replay_helper import FileReplayHelper

  # open a client
  renderer = WebsocketRenderer()
  time.sleep(3)

  # load a replay
  replay = FileReplayHelper.load('replay_dev.json', decompress=False)

  # run the replay
  for packet in replay:
    renderer.render_packet(packet)
    time.sleep(1)
