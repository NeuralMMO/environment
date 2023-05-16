import json
import random
from typing import List


# pylint: disable=abstract-method, super-init-not-called

class Task():
  def completed(self, realm, entity) -> bool:
    raise NotImplementedError

  def description(self) -> List:
    return self.__class__.__name__

  def to_string(self) -> str:
    return json.dumps(self.description())

###############################################################

class TaskTarget():
  def __init__(self, name: str, agents: List[str]) -> None:
    self._name = name
    self._agents = agents

  def agents(self) ->  List[int]:
    return self._agents

  def description(self) -> List:
    return self._name

  def member(self, member):
    assert member < len(self._agents)
    return TaskTarget(f"{self.description()}.{member}", [self._agents[member]])

class TargetTask(Task):
  def __init__(self, target: TaskTarget) -> None:
    self._target = target

  def description(self) -> List:
    return [super().description(), self._target.description()]

  def completed(self, realm, entity) -> bool:
    raise NotImplementedError

###############################################################

class TeamHelper():
  def __init__(self, agents: List[int], num_teams: int) -> None:
    assert len(agents) % num_teams == 0
    self.team_size = len(agents) // num_teams
    self._teams = [
      list(agents[i * self.team_size : (i+1) * self.team_size])
      for i in range(num_teams)
    ]
    self._agent_to_team = {a: tid for tid, t in enumerate(self._teams) for a in t}

  def own_team(self, agent_id: int) -> TaskTarget:
    return TaskTarget("Team.Self", self._teams[self._agent_to_team[agent_id]])

  def left_team(self, agent_id: int) -> TaskTarget:
    return TaskTarget("Team.Left", self._teams[
      (self._agent_to_team[agent_id] -1) % len(self._teams)
    ])

  def right_team(self, agent_id: int) -> TaskTarget:
    return TaskTarget("Team.Right", self._teams[
      (self._agent_to_team[agent_id] + 1) % len(self._teams)
    ])

  def all(self) -> TaskTarget:
    return TaskTarget("All", list(self._agent_to_team.keys()))

###############################################################

class AND(Task):
  def __init__(self, *tasks: Task) -> None:
    super().__init__()
    assert len(tasks) > 0
    self._tasks = tasks

  def completed(self, realm, entity) -> bool:
    return all(t.completed(realm, entity) for t in self._tasks)

  def description(self) -> List:
    return ["AND"] + [t.description() for t in self._tasks]

class OR(Task):
  def __init__(self, *tasks: Task) -> None:
    super().__init__()
    assert len(tasks) > 0
    self._tasks = tasks

  def completed(self, realm, entity) -> bool:
    return any(t.completed(realm, entity) for t in self._tasks)

  def description(self) -> List:
    return ["OR"] + [t.description() for t in self._tasks]

class NOT(Task):
  def __init__(self, task: Task) -> None:
    super().__init__()
    self._task = task

  def completed(self, realm, entity) -> bool:
    return not self._task.completed(realm, entity)

  def description(self) -> List:
    return ["NOT", self._task.description()]

###############################################################

class InflictDamage(TargetTask):
  def __init__(self, target: TaskTarget, damage_type: int, quantity: int):
    super().__init__(target)
    self._damage_type = damage_type
    self._quantity = quantity

  def completed(self, realm, entity) -> bool:
    # TODO(daveey) damage_type is ignored, needs to be added to entity.history
    return sum(
      realm.players[a].history.damage_inflicted for a in self._target.agents()
    ) >= self._quantity

  def description(self) -> List:
    return super().description() + [self._damage_type, self._quantity]

class Defend(TargetTask):
  def __init__(self, target, num_steps) -> None:
    super().__init__(target)
    self._num_steps = num_steps

  def completed(self, realm, entity) -> bool:
    # TODO(daveey) need a way to specify time horizon
    return realm.tick >= self._num_steps and all(
      realm.players[a].alive for a in self._target.agents()
    )

  def description(self) -> List:
    return super().description() + [self._num_steps]

class Inflict(TargetTask):
  def __init__(self, target: TaskTarget, damage_type, quantity: int):
    '''
    target: The team that is completing the task. Any agent may complete
    damage_type: Can use skills.Melee/Range/Mage
    quantity: Minimum damage to inflict in a single hit
    '''

class Defeat(TargetTask):
  def __init__(self, target: TaskTarget, entity_type, level: int):
    '''
    target: The team that is completing the task. Any agent may complete
    entity type: entity.Player or entity.NPC
    level: minimum target level to defeat
      '''

class Achieve(TargetTask):
  def __init__(self, target: TaskTarget, skill, level: int):
    '''
    target: The team that is completing the task. Any agent may complete.
    skill: systems.skill to advance
    level: level to reach
    '''

class Harvest(TargetTask):
  def __init__(self, target: TaskTarget, resource, level: int):
    '''
    target: The team that is completing the task. Any agent may complete
    resource: lib.material to harvest
    level: minimum material level to harvest
    '''

class Equip(Task):
  def __init__(self, target: TaskTarget, item, level: int):
    '''
    target: The team that is completing the task. Any agent may complete.
    item: systems.item to equip
    level: Minimum level of that item
    '''

class Hoard(Task):
  def __init__(self, target: TaskTarget, gold):
    '''
    target: The team that is completing the task. Completed across the team
    gold: reach this amount of gold held at one time (inventory.gold sum over team)
    '''

class Group(Task):
  def __init__(self, target: TaskTarget, num_teammates: int, distance: int):
    '''
    target: The team that is completing the task. Completed across the team
    num_teammates: Number of teammates to group together
    distance: Max distance to nearest teammate
    '''

class Spread(Task):
  def __init__(self, target: TaskTarget, num_teammates: int, distance: int):
    '''
    target: The team that is completing the task. Completed across the team
    num_teammates: Number of teammates to group together
    distance: Min distance to nearest teammate
    '''

class Eliminate(Task):
  def __init__(self, target: TaskTarget, opponent_team):
    '''
    target: The team that is completing the task. Completed across the team
    opponent_team: left/right/any team to be eliminated (all agents defeated)
    '''

###############################################################

class TaskSampler():
  def __init__(self) -> None:
    self._task_specs = []
    self._task_spec_weights = []

  def add_task_spec(self, task_class, param_space = None, weight: float = 1):
    self._task_specs.append((task_class, param_space or []))
    self._task_spec_weights.append(weight)

  def sample(self,
             min_clauses: int = 1,
             max_clauses: int = 1,
             min_clause_size: int = 1,
             max_clause_size: int = 1,
             not_p: float = 0.0) -> Task:

    clauses = []
    for _ in range(0, random.randint(min_clauses, max_clauses)):
      task_specs = random.choices(
        self._task_specs,
        weights = self._task_spec_weights,
        k = random.randint(min_clause_size, max_clause_size)
      )
      tasks = []
      for task_class, task_param_space in task_specs:
        task = task_class(*[random.choice(tp) for tp in task_param_space])
        if random.random() < not_p:
          task = NOT(task)
        tasks.append(task)

      if len(tasks) == 1:
        clauses.append(tasks[0])
      else:
        clauses.append(AND(*tasks))

    if len(clauses) == 1:
      return clauses[0]

    return OR(*clauses)

  @staticmethod
  def create_default_task_sampler(team_helper: TeamHelper, agent_id: int):
    neighbors = [team_helper.left_team(agent_id), team_helper.right_team(agent_id)]
    own_team = team_helper.own_team(agent_id)
    team_mates = [own_team.member(m) for m in range(team_helper.team_size)]
    sampler = TaskSampler()

    sampler.add_task_spec(InflictDamage, [neighbors + [own_team], [0, 1, 2], [0, 100, 1000]])
    sampler.add_task_spec(Defend, [team_mates, [512, 1024]])

    return sampler
