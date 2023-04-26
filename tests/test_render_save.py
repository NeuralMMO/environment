'''Manual test for render client connectivity'''

if __name__ == '__main__':
  import nmmo
  from nmmo.render.render_client import OnlineRenderer
  from tests.testhelpers import ScriptedAgentTestConfig

  TEST_HORIZON = 30

  # config.RENDER option is gone, 
  # RENDER can be done without setting any config
  config = ScriptedAgentTestConfig()
  env = nmmo.Env(config)

  env.reset()

  # the renderer is external to the env, so need to manually initiate it
  renderer = OnlineRenderer(env)

  for tick in range(TEST_HORIZON):
    env.step({})
    renderer.render()

  # save the packet: this is possible because config.SAVE_REPLAY = True
  env.packet_manager.save('replay_dev.json', compress=False)
