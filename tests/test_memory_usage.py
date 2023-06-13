# pylint: disable=bad-builtin, unused-variable
import psutil

import nmmo

def test_memory_usage():
  env = nmmo.Env()
  process = psutil.Process()
  print("memory", process.memory_info().rss)

if __name__ == '__main__':
  test_memory_usage()
