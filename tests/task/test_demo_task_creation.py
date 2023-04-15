import unittest

from tests.testhelpers import ScriptedAgentTestConfig
from scripted.baselines import Sleeper

from nmmo.core.env import Env as TaskEnv
from nmmo.task.predicate.core import predicate
from nmmo.task.predicate.gold_predicate import HoardGold
from nmmo.task.predicate.base_predicate import AllDead, StayAlive
from nmmo.task.game_state import GameState
from nmmo.task.group import Group
from nmmo.task.utils import TeamHelper
from nmmo.task.task_api import PredicateTask, MultiTask, Repeat

class TestDemoTask(unittest.TestCase):

  def test_example_user_task_definition(self):
    config = ScriptedAgentTestConfig()
    config.PLAYERS = [Sleeper]
    env = TaskEnv(config)
    team_helper = TeamHelper(list(range(1, config.PLAYER_N+1)), len(config.PLAYERS))

    # Define Predicate Utilities
    @predicate
    def CustomPredicate(gs: GameState,
                        subject: Group,
                        target: Group):
        t1 = HoardGold(subject, 10) & AllDead(target)
        return t1
      
    # Define Task Utilities
    class CompletionChangeTask(PredicateTask):
       def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self._previous = 0

       def _rewards(self, gs: GameState):
          infos = {int(ent_id): {self._predicate.name: self.evaluate(gs)}
             for ent_id in self._assignee}
          change = self.evaluate(gs) - self._previous
          self._previous = self.evaluate(gs)
          rewards = {int(ent_id): change for ent_id in self._assignee}
          return rewards, infos

    # Creation of the actual tasks
    # TODO(mark) We should greatly simplify team helper in the next iteration
    stay_alive_tasks = [Repeat(agent, StayAlive(agent)) for agent in team_helper.all()]
    custom_task_2 = CompletionChangeTask(team_helper.own_team(1),
                                          CustomPredicate(subject=team_helper.own_team(1),
                                                          target = team_helper.left_team(1)))
    task = MultiTask(
       (custom_task_2, 5),
       *stay_alive_tasks
    )

    # Test rollout
    env.change_task(task)
    for _ in range(50):
       env.step({})

    # DONE

if __name__ == '__main__':
  unittest.main()
