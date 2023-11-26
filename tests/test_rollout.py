import nmmo
from scripted.baselines import Random

class SimpleConfig(nmmo.config.Small, nmmo.config.Combat):
  pass

def test_rollout():
  config = nmmo.config.Default()  # SimpleConfig()
  config.set("PLAYERS", [Random])
  config.set("USE_CYTHON", True)

  env = nmmo.Env(config)
  env.reset()
  for _ in range(64):
    env.step({})

  env.reset()

if __name__ == '__main__':
  test_rollout()
