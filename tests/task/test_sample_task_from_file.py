import unittest

import nmmo
from tests.testhelpers import ScriptedAgentTestConfig

class TestSampleTaskFromFile(unittest.TestCase):
  def test_sample_task_from_file(self):
    # init the env with the pickled training task spec
    config = ScriptedAgentTestConfig()
    config.CURRICULUM_FILE_PATH = 'tests/task/sample_curriculum.pkl'
    env = nmmo.Env(config)

    # env.reset() samples and instantiates a task for each agent
    #   when sample_traning_tasks is set True
    env.reset()

    self.assertEqual(len(env.possible_agents), len(env.tasks))
    # for the training tasks, the task assignee and subject should be the same
    for task in env.tasks:
      self.assertEqual(task.assignee, task.subject)

if __name__ == '__main__':
  unittest.main()
