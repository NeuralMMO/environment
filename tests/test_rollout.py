import nmmo
from scripted.baselines import Random

def test_rollout():
  config = nmmo.config.Default()
  config.PLAYERS = [Random]

  env = nmmo.Env(config)
  env.reset()
  for _ in range(128):
    env.step({})

if __name__ == '__main__':
  test_rollout()
