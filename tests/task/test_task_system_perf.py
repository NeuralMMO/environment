import unittest

from nmmo.core.env import Env
from nmmo.task.task_api import Task, nmmo_default_task
from tests.testhelpers import profile_env_step, ScriptedAgentTestConfig

PROFILE_PERF = False

class TestTaskSystemPerf(unittest.TestCase):
  def test_nmmo_default_task(self):
    config = ScriptedAgentTestConfig()
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
          self.assertTrue('StayAlive' in tasks[agent_id-1].name) # default task
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
  # === Test condition: default ===
  # - env.step({}): 12.302560470998287
  # - env.realm.step(): 3.8562550359929446
  # - env._compute_observations(): 3.3712658310032566
  # - obs.to_gym(), ActionTarget: 2.477421684998262
  # - env._compute_rewards(): 1.4060252049966948

  # === Test condition: no_task ===
  # - env.step({}): 10.818232985999202
  # - env.realm.step(): 3.79689467499702
  # - env._compute_observations(): 3.3100888289991417
  # - obs.to_gym(), ActionTarget: 2.409053840994602
  # - env._compute_rewards(): 0.00781778599775862

  # === Test condition: dummy_eval_fn, using Predicate class ===
  # - env.step({}): 11.989140973004396
  # - env.realm.step(): 3.8649445789997117
  # - env._compute_observations(): 3.344463708999683
  # - obs.to_gym(), ActionTarget: 2.431279453005118
  # - env._compute_rewards(): 1.119989460996294

  # === Test condition: pure_func_eval, WITHOUT Predicate class ===
  # - env.step({}): 11.032341518002795
  # - env.realm.step(): 3.8636899659977644
  # - env._compute_observations(): 3.3460479429995758
  # - obs.to_gym(), ActionTarget: 2.498140270996373
  # - env._compute_rewards(): 0.055145307997008786
  # """
