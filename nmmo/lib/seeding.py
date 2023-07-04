# copied from https://github.com/openai/gym/blob/master/gym/utils/seeding.py

"""Set of random number generator functions: seeding, generator, hashing seeds."""
from typing import Any, Optional, Tuple

import numpy as np

from gym import error


class RandomNumberGenerator(np.random.Generator):
  def __init__(self, bit_generator):
    super().__init__(bit_generator)
    self._dir_seq_len = 1024
    self._wrap = self._dir_seq_len - 1
    self._dir_seq = self.integers(0, 4, size=self._dir_seq_len).astype(np.uint8)
    self._dir_idx = 0

  # provide a random direction from the pre-generated sequence
  def get_direction(self):
    self._dir_idx = (self._dir_idx + 1) & self._wrap
    return self._dir_seq[self._dir_idx]

def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, Any]:
  """Generates a random number generator from the seed and returns the Generator and seed.

  Args:
      seed: The seed used to create the generator

  Returns:
      The generator and resulting seed

  Raises:
      Error: Seed must be a non-negative integer or omitted
  """
  if seed is not None and not (isinstance(seed, int) and 0 <= seed):
    raise error.Error(f"Seed must be a non-negative integer or omitted, not {seed}")

  seed_seq = np.random.SeedSequence(seed)
  np_seed = seed_seq.entropy
  rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
  return rng, np_seed
