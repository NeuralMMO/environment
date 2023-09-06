import unittest

import numpy as np

import nmmo
import nmmo.systems.skill
from tests.testhelpers import ScriptedAgentTestConfig, ScriptedAgentTestEnv


class TestSkillLevel(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.config = ScriptedAgentTestConfig()
    cls.config.PROGRESSION_EXP_THRESHOLD = [0, 10, 20, 30, 40, 50]
    cls.config.PROGRESSION_LEVEL_MAX = len(cls.config.PROGRESSION_EXP_THRESHOLD)
    cls.env = ScriptedAgentTestEnv(cls.config)

  def test_experience_calculator(self):
    exp_calculator = nmmo.systems.skill.ExperienceCalculator(self.config)

    self.assertTrue(np.array_equal(self.config.PROGRESSION_EXP_THRESHOLD,
                                   exp_calculator.exp_threshold))

    for level in range(1, self.config.PROGRESSION_LEVEL_MAX + 1):
      self.assertEqual(exp_calculator.level_at_exp(exp_calculator.exp_at_level(level)), level)

    self.assertEqual(exp_calculator.exp_at_level(-1),  # invalid level
                     min(self.config.PROGRESSION_EXP_THRESHOLD))
    self.assertEqual(exp_calculator.exp_at_level(30),  # level above the max
                     max(self.config.PROGRESSION_EXP_THRESHOLD))

    self.assertEqual(exp_calculator.level_at_exp(0), 1)
    self.assertEqual(exp_calculator.level_at_exp(5), 1)
    self.assertEqual(exp_calculator.level_at_exp(45), 5)
    self.assertEqual(exp_calculator.level_at_exp(50), 6)
    self.assertEqual(exp_calculator.level_at_exp(100), 6)

  def test_add_xp(self):
    self.env.reset()
    player = self.env.realm.players[1]

    skill_list = ["melee", "range", "mage",
                  "fishing", "herbalism", "prospecting", "carving", "alchemy"]

    # check the initial levels and exp
    for skill in skill_list:
      self.assertEqual(getattr(player.skills, skill).level.val, 1)
      self.assertEqual(getattr(player.skills, skill).exp.val, 0)

    # add 1 exp to melee, does NOT level up
    player.skills.melee.add_xp(1)
    for skill in skill_list:
      if skill == "melee":
        self.assertEqual(getattr(player.skills, skill).level.val, 1)
        self.assertEqual(getattr(player.skills, skill).exp.val, 1)
      else:
        self.assertEqual(getattr(player.skills, skill).level.val, 1)
        self.assertEqual(getattr(player.skills, skill).exp.val, 0)

    # add 30 exp to fishing, levels up to 3
    player.skills.fishing.add_xp(30)
    for skill in skill_list:
      if skill == "melee":
        self.assertEqual(getattr(player.skills, skill).level.val, 1)
        self.assertEqual(getattr(player.skills, skill).exp.val, 1)
      elif skill == "fishing":
        self.assertEqual(getattr(player.skills, skill).level.val, 4)
        self.assertEqual(getattr(player.skills, skill).exp.val, 30)
      else:
        self.assertEqual(getattr(player.skills, skill).level.val, 1)
        self.assertEqual(getattr(player.skills, skill).exp.val, 0)


if __name__ == '__main__':
  unittest.main()

  # config = nmmo.config.Default()
  # exp_calculator = nmmo.systems.skill.ExperienceCalculator(config)

  # print(exp_calculator.exp_threshold)
  # print(exp_calculator.exp_at_level(10))
  # print(exp_calculator.level_at_exp(150)) # 2
  # print(exp_calculator.level_at_exp(300)) # 3
  # print(exp_calculator.level_at_exp(1000)) # 7
