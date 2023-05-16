
#pylint: disable=unused-argument

import logging
logging.basicConfig(level=logging.INFO, stream=None)

def pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
  if unit == 'seconds':
    prefix = 'millisec'
    scale = 1000
  elif unit == 'operations':
    prefix = ''
    scale = 1
  else:
    raise RuntimeError(f"Unexpected measurement unit {unit}")
  return prefix, scale
