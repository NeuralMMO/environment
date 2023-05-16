import numpy as np

class ExperienceCalculator:
  def __init__(self, num_levels=15):
    self.exp = np.array([0] + [10*2**i for i in range(num_levels)])

  def exp_at_level(self, level):
    return int(self.exp[level - 1])

  def level_at_exp(self, exp):
    if exp >= self.exp[-1]:
      return len(self.exp)
    return np.argmin(exp >= self.exp)
