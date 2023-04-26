'''Manual test for rendering replay'''

if __name__ == '__main__':
  import time

  from nmmo.render.render_client import WebsocketRenderer
  from nmmo.render.packet_manager import SimplePacketManager

  # open a client
  renderer = WebsocketRenderer()
  time.sleep(3)

  # load a replay
  replay = SimplePacketManager.load('replay_dev.json', decompress=False)

  # run the replay
  for packet in replay:
    renderer.render(packet)
    time.sleep(1.5)
