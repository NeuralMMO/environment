import unittest

import nmmo
from nmmo.core.env import Env
from nmmo.task.task_api import Task, nmmo_default_task
from tests.testhelpers import profile_env_step

PROFILE_PERF = False

class TestTaskSystemPerf(unittest.TestCase):
  def test_nmmo_default_task(self):
    config = nmmo.config.Default()
    env = Env(config)
    agent_list = env.possible_agents

    for test_mode in [None, 'no_task', 'dummy_eval_fn', 'pure_func_eval']:

      # create tasks
      if test_mode == 'pure_func_eval':
        def create_stay_alive_eval_wo_group(agent_id: int):
          return lambda gs: agent_id in gs.alive_agents
        tasks = [Task(create_stay_alive_eval_wo_group(agent_id), assignee=agent_id)
                for agent_id in agent_list]
      else:
        tasks = nmmo_default_task(agent_list, test_mode)

      # check tasks
      for agent_id in agent_list:
        if test_mode is None:
          self.assertTrue('TickGE' in tasks[agent_id-1].name) # default task
        if test_mode != 'no_task':
          self.assertTrue(f'assignee:({agent_id},)' in tasks[agent_id-1].name)

      # pylint: disable=cell-var-from-loop
      if PROFILE_PERF:
        test_cond = 'default' if test_mode is None else test_mode
        profile_env_step(tasks=tasks, condition=test_cond)
      else:
        env.reset(make_task_fn=lambda: tasks)
        for _ in range(3):
          env.step({})

    # DONE


if __name__ == '__main__':
  unittest.main()

  # """ Tested on Win 11, docker
  # === Test condition: default (StayAlive-based Predicate) ===
  # - env.step({}): 13.398321460997977
  # - env.realm.step(): 3.6524868449996575
  # - env._compute_observations(): 3.2038183499971638
  # - obs.to_gym(), ActionTarget: 2.30746804500086
  # - env._compute_rewards(): 2.7206644940015394

  # === Test condition: no_task ===
  # - env.step({}): 10.576253965999058
  # - env.realm.step(): 3.674701832998835
  # - env._compute_observations(): 3.260661373002222
  # - obs.to_gym(), ActionTarget: 2.313872797996737
  # - env._compute_rewards(): 0.009020475001307204

  # === Test condition: dummy_eval_fn -based Predicate ===
  # - env.step({}): 12.797982947995479
  # - env.realm.step(): 3.604593793003005
  # - env._compute_observations(): 3.2095355240016943
  # - obs.to_gym(), ActionTarget: 2.313207338003849
  # - env._compute_rewards(): 2.266267291997792

  # === Test condition: pure_func_eval WITHOUT Predicate ===
  # - env.step({}): 10.637560240997118
  # - env.realm.step(): 3.633970066999609
  # - env._compute_observations(): 3.2308093659958104
  # - obs.to_gym(), ActionTarget: 2.331246039000689
  # - env._compute_rewards(): 0.0988905300037004
  # """
