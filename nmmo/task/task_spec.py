import functools
from dataclasses import dataclass, field
from typing import Iterable, Dict, List, Union, Type
from types import FunctionType
from copy import deepcopy
from tqdm import tqdm

import numpy as np

import nmmo
from nmmo.task.task_api import Task, make_same_task
from nmmo.task.predicate_api import Predicate, make_predicate
from nmmo.task.group import Group
from nmmo.task import base_predicates as bp
from nmmo.lib.team_helper import TeamHelper

""" task_spec

    eval_fn can come from the base_predicates.py or could be custom functions like above
    eval_fn_kwargs are the additional args that go into predicate. There are also special keys
      * "target" must be ["left_team", "right_team", "left_team_leader", "right_team_leader"]
          these str will be translated into the actual agent ids

    task_cls specifies the task class to be used. Default is Task.
    task_kwargs are the optional, additional args that go into the task.
 
    reward_to: must be in ["team", "agent"]
      * "team" create a single team task, in which all team members get rewarded
      * "agent" create a task for each agent, in which only the agent gets rewarded
    
    sampling_weight specifies the weight of the task in the curriculum sampling. Default is 1
"""

REWARD_TO = ["agent", "team"]
VALID_TARGET = ["left_team", "left_team_leader",
                "right_team", "right_team_leader",
                "my_team_leader", "all_foes"]

@dataclass
class TaskSpec:
  eval_fn: FunctionType
  eval_fn_kwargs: Dict
  task_cls: Type[Task] = Task
  task_kwargs: Dict = field(default_factory=dict)
  reward_to: str = "agent"
  sampling_weight: float = 1.0
  embedding: np.ndarray = None
  predicate: Predicate = None
  tags: List[str] = field(default_factory=list)

  def __post_init__(self):
    if self.predicate is None:
      assert isinstance(self.eval_fn, FunctionType), \
        "eval_fn must be a function"
    else:
      assert self.eval_fn is None, "Cannot specify both eval_fn and predicate"
    assert self.reward_to in REWARD_TO, \
      f"reward_to must be in {REWARD_TO}"
    if "target" in self.eval_fn_kwargs:
      assert self.eval_fn_kwargs["target"] in VALID_TARGET, \
      f"target must be in {VALID_TARGET}"

  @functools.cached_property
  def name(self):
    # pylint: disable=no-member
    kwargs_str = []
    for key, val in self.eval_fn_kwargs.items():
      val_str = str(val)
      if isinstance(val, type):
        val_str = val.__name__
      kwargs_str.append(f"{key}:{val_str}_")
    kwargs_str = "(" + "".join(kwargs_str)[:-1] + ")" # remove the last _
    pred_name = self.eval_fn.__name__ if self.predicate is None else self.predicate.name
    return "_".join([self.task_cls.__name__, pred_name,
                     kwargs_str, "reward_to:" + self.reward_to])

def make_task_from_spec(assign_to: Union[Iterable[int], Dict],
                        task_spec: List[TaskSpec]) -> List[Task]:
  """
  Args:
    assign_to: either a Dict with { team_id: [agent_id]} or a List of agent ids
    task_spec: a list of tuples (reward_to, eval_fn, pred_fn_kwargs, task_kwargs)
    
    each tuple is assigned to the teams
  """
  teams = assign_to
  if not isinstance(teams, Dict): # convert agent id list to the team dict format
    teams = {idx: [agent_id] for idx, agent_id in enumerate(assign_to)}
  team_list = list(teams.keys())
  team_helper = TeamHelper(teams)

  # assign task spec to teams (assign_to)
  tasks = []
  for idx in range(min(len(team_list), len(task_spec))):
    team_id = team_list[idx]

    # map local vars to spec attributes
    reward_to = task_spec[idx].reward_to
    pred_fn = task_spec[idx].eval_fn
    pred_fn_kwargs = deepcopy(task_spec[idx].eval_fn_kwargs)
    task_cls = task_spec[idx].task_cls
    task_kwargs = deepcopy(task_spec[idx].task_kwargs)
    task_kwargs["embedding"] = task_spec[idx].embedding # to pass to task_cls
    task_kwargs["spec_name"] = task_spec[idx].name
    task_kwargs["reward_to"] = task_spec[idx].reward_to
    task_kwargs["tags"] = task_spec[idx].tags
    predicate = task_spec[idx].predicate

    # reserve "target" for relative agent mapping
    target_keys = [key for key in pred_fn_kwargs.keys() if key.startswith("target")]
    for key in target_keys:
      target_keyword = pred_fn_kwargs.pop(key)
      assert target_keyword in VALID_TARGET, "Invalid target"
      # translate target to specific agent ids using team_helper
      target_ent = team_helper.get_target_agent(team_id, target_keyword)
      pred_fn_kwargs[key] = target_ent

    # handle some special cases and instantiate the predicate first
    if pred_fn is not None and isinstance(pred_fn, FunctionType):
      # if a function is provided as a predicate
      pred_cls = make_predicate(pred_fn)

    # TODO: should create a test for these
    if (pred_fn in [bp.AllDead]) or \
       (pred_fn in [bp.StayAlive] and "target" in pred_fn_kwargs):
      # use the target as the predicate subject
      target_ent = pred_fn_kwargs.pop("target") # remove target
      predicate = pred_cls(Group(target_ent), **pred_fn_kwargs)

    # create the task
    if reward_to == "team":
      assignee = team_helper.teams[team_id]
      if predicate is None:
        predicate = pred_cls(Group(assignee), **pred_fn_kwargs)
        tasks.append(predicate.create_task(task_cls=task_cls, **task_kwargs))
      else:
        # this branch is for the cases like AllDead, StayAlive
        tasks.append(predicate.create_task(assignee=assignee, task_cls=task_cls,
                                           **task_kwargs))

    elif reward_to == "agent":
      agent_list = team_helper.teams[team_id]
      if predicate is None:
        tasks += make_same_task(pred_cls, agent_list, pred_kwargs=pred_fn_kwargs,
                                task_cls=task_cls, task_kwargs=task_kwargs)
      else:
        # this branch is for the cases like AllDead, StayAlive
        tasks += [predicate.create_task(assignee=agent_id, task_cls=task_cls, **task_kwargs)
                  for agent_id in agent_list]

  return tasks

# pylint: disable=bare-except,cell-var-from-loop
def check_task_spec(spec_list: List[TaskSpec], debug=False) -> List[Dict]:
  teams = {0: [1, 2, 3], 3: [4, 5], 7: [6, 7], 11: [8, 9], 14: [10, 11]}
  config = nmmo.config.Default()
  config.set("PLAYER_N", 11)
  config.set("TEAMS", teams)
  env = nmmo.Env(config)
  results = []
  for single_spec in tqdm(spec_list):
    result = {"spec_name": single_spec.name}
    try:
      env.reset(make_task_fn=lambda: make_task_from_spec(teams, [single_spec]))
      for _ in range(3):
        env.step({})
      result["runnable"] = True
    except:
      result["runnable"] = False
      if debug:
        raise
    results.append(result)
  return results
