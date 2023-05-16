

import nmmo
from scripted import baselines

def test_pettingzoo_api():
  config = nmmo.config.Default()
  config.PLAYERS = [baselines.Random]
  # ensv = nmmo.Env(config)
  # TODO: disabled due to Env not implementing the correct PettinZoo step() API
  # parallel_api_test(env, num_cycles=1000)


if __name__ == '__main__':
  test_pettingzoo_api()
