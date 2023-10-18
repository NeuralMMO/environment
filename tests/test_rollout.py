import nmmo
from scripted.baselines import Random

class SimpleConfig(nmmo.config.Small, nmmo.config.Combat):
  pass

def test_rollout():
  config = SimpleConfig()  #nmmo.config.Default()
  config.set("PLAYERS", [Random])

  env = nmmo.Env(config)
  env.reset()
  for _ in range(64):
    env.step({})

  env.reset()

if __name__ == '__main__':
  test_rollout()
