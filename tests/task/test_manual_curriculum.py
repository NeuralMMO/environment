'''Manual test for creating learning curriculum manually'''
# pylint: disable=invalid-name,redefined-outer-name,bad-builtin
# pylint: disable=wildcard-import,unused-wildcard-import
from typing import List

import nmmo.lib.material as m
from nmmo.task.base_predicates import *
from nmmo.task.task_api import OngoingTask
from nmmo.task import constraint as c
from nmmo.task.task_spec import TaskSpec, check_task_spec

EVENT_NUMBER_GOAL = [3, 4, 5, 7, 9, 12, 15, 20, 30, 50]
INFREQUENT_GOAL = list(range(1, 10))
STAY_ALIVE_GOAL = [50, 100, 150, 200, 300, 500]
TEAM_NUMBER_GOAL = [10, 20, 30, 50, 70, 100]
LEVEL_GOAL = list(range(1, 10)) # TODO: get config
AGENT_NUM_GOAL = [1, 2, 3, 4, 5] # competition team size: 8
ITEM_NUM_GOAL = AGENT_NUM_GOAL
TEAM_ITEM_GOAL = [1, 3, 5, 7, 10, 15, 20]
SKILLS = c.combat_skills + c.harvest_skills
COMBAT_STYLE = c.combat_skills
ALL_ITEM = c.armour + c.weapons + c.tools + c.ammunition + c.consumables
EQUIP_ITEM = c.armour + c.weapons + c.tools + c.ammunition
HARVEST_ITEM = c.weapons + c.ammunition + c.consumables

task_spec: List[TaskSpec] = []

# explore, eat, drink, attack any agent, harvest any item, level up any skill
#   which can happen frequently
essential_skills = ['GO_FARTHEST', 'EAT_FOOD', 'DRINK_WATER',
                    'SCORE_HIT', 'HARVEST_ITEM', 'LEVEL_UP']
for event_code in essential_skills:
  for cnt in EVENT_NUMBER_GOAL:
    task_spec.append(TaskSpec(eval_fn=CountEvent,
                              eval_fn_kwargs={'event': event_code, 'N': cnt},
                              sampling_weight=30))

# item/market skills, which happen less frequently or should not do too much
item_skills = ['CONSUME_ITEM', 'GIVE_ITEM', 'DESTROY_ITEM', 'EQUIP_ITEM',
               'GIVE_GOLD', 'LIST_ITEM', 'EARN_GOLD', 'BUY_ITEM']
for event_code in item_skills:
  task_spec += [TaskSpec(eval_fn=CountEvent, eval_fn_kwargs={'event': event_code, 'N': cnt})
                for cnt in INFREQUENT_GOAL] # less than 10

# find resource tiles
for resource in m.Harvestable:
  for reward_to in ['agent', 'team']:
    task_spec.append(TaskSpec(eval_fn=CanSeeTile, eval_fn_kwargs={'tile_type': resource},
                              reward_to=reward_to, sampling_weight=10))

# stay alive ... like ... for 300 ticks
# i.e., getting incremental reward for each tick alive as an individual or a team
for reward_to in ['agent', 'team']:
  for num_tick in STAY_ALIVE_GOAL:
    task_spec.append(TaskSpec(eval_fn=TickGE, eval_fn_kwargs={'num_tick': num_tick},
                              reward_to=reward_to))

# protect the leader: get reward for each tick the leader is alive
# NOTE: a tuple of length four, to pass in the task_kwargs
task_spec.append(TaskSpec(eval_fn=StayAlive, eval_fn_kwargs={'target': 'my_team_leader'},
                          reward_to='team', task_cls=OngoingTask))

# want the other team or team leader to die
for target in ['left_team', 'left_team_leader', 'right_team', 'right_team_leader']:
  task_spec.append(TaskSpec(eval_fn=AllDead, eval_fn_kwargs={'target': target},
                            reward_to='team'))

# occupy the center tile, assuming the Medium map size
# TODO: it'd be better to have some intermediate targets toward the center
for reward_to in ['agent', 'team']:
  task_spec.append(TaskSpec(eval_fn=OccupyTile, eval_fn_kwargs={'row': 80, 'col': 80},
                            reward_to=reward_to)) # TODO: get config for map size

# form a tight formation, for a certain number of ticks
def PracticeFormation(gs, subject, dist, num_tick):
  return AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick)
for dist in [1, 3, 5, 10]:
  task_spec += [TaskSpec(eval_fn=PracticeFormation,
                         eval_fn_kwargs={'dist': dist, 'num_tick': num_tick},
                         reward_to='team') for num_tick in STAY_ALIVE_GOAL]

# find the other team leader
for reward_to in ['agent', 'team']:
  for target in ['left_team_leader', 'right_team_leader']:
    task_spec.append(TaskSpec(eval_fn=CanSeeAgent, eval_fn_kwargs={'target': target},
                              reward_to=reward_to))

# find the other team (any agent)
for reward_to in ['agent']: #, 'team']:
  for target in ['left_team', 'right_team']:
    task_spec.append(TaskSpec(eval_fn=CanSeeGroup, eval_fn_kwargs={'target': target},
                              reward_to=reward_to))

# explore the map -- sum the l-inf distance traveled by all subjects
for dist in [10, 20, 30, 50, 100]: # each agent
  task_spec.append(TaskSpec(eval_fn=DistanceTraveled, eval_fn_kwargs={'dist': dist}))
for dist in [30, 50, 70, 100, 150, 200, 300, 500]: # summed over all team members
  task_spec.append(TaskSpec(eval_fn=DistanceTraveled, eval_fn_kwargs={'dist': dist},
                            reward_to='team'))

# level up a skill
for skill in SKILLS:
  for level in LEVEL_GOAL[1:]:
    # since this is an agent task, num_agent must be 1
    task_spec.append(TaskSpec(eval_fn=AttainSkill,
                              eval_fn_kwargs={'skill': skill, 'level': level, 'num_agent': 1},
                              reward_to='agent',
                              sampling_weight=10*(5-level) if level < 5 else 1))

# make attain skill a team task by varying the number of agents
for skill in SKILLS:
  for level in LEVEL_GOAL[1:]:
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(
          TaskSpec(eval_fn=AttainSkill,
                   eval_fn_kwargs={'skill': skill, 'level': level, 'num_agent': num_agent},
                   reward_to='team'))

# practice specific combat style
for style in COMBAT_STYLE:
  for cnt in EVENT_NUMBER_GOAL:
    task_spec.append(TaskSpec(eval_fn=ScoreHit, eval_fn_kwargs={'combat_style': style, 'N': cnt},
                              sampling_weight=5))
  for cnt in TEAM_NUMBER_GOAL:
    task_spec.append(TaskSpec(eval_fn=ScoreHit, eval_fn_kwargs={'combat_style': style, 'N': cnt},
                              reward_to='team'))

# defeat agents of a certain level as a team
for agent_type in ['player', 'npc']: # c.AGENT_TYPE_CONSTRAINT
  for level in LEVEL_GOAL:
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=DefeatEntity,
                                  eval_fn_kwargs={'agent_type': agent_type, 'level': level,
                                                  'num_agent': num_agent},
                                  reward_to='team'))

# hoarding gold -- evaluated on the current gold
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=HoardGold, eval_fn_kwargs={'amount': amount},
                            sampling_weight=3))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=HoardGold, eval_fn_kwargs={'amount': amount},
                            reward_to='team'))

# earning gold -- evaluated on the total gold earned by selling items
# does NOT include looted gold
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={'amount': amount},
                            sampling_weight=3))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=EarnGold, eval_fn_kwargs={'amount': amount},
                            reward_to='team'))

# spending gold, by buying items
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=SpendGold, eval_fn_kwargs={'amount': amount},
                            sampling_weight=3))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=SpendGold, eval_fn_kwargs={'amount': amount},
                            reward_to='team'))

# making profits by trading -- only buying and selling are counted
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=MakeProfit, eval_fn_kwargs={'amount': amount},
                            sampling_weight=3))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(TaskSpec(eval_fn=MakeProfit, eval_fn_kwargs={'amount': amount},
                            reward_to='team'))

# managing inventory space
def PracticeInventoryManagement(gs, subject, space, num_tick):
  return InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)
for space in [2, 4, 8]:
  task_spec += [TaskSpec(eval_fn=PracticeInventoryManagement,
                         eval_fn_kwargs={'space': space, 'num_tick': num_tick})
                for num_tick in STAY_ALIVE_GOAL]

# own item, evaluated on the current inventory
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=OwnItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  sampling_weight=4-level if level < 4 else 1))
    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=OwnItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  reward_to='team'))

# equip item, evaluated on the current inventory and equipment status
for item in EQUIP_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    task_spec.append(TaskSpec(eval_fn=EquipItem,
                              eval_fn_kwargs={'item': item, 'level': level, 'num_agent': 1},
                              sampling_weight=4-level if level < 4 else 1))
    # team task
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=EquipItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'num_agent': num_agent},
                                  reward_to='team'))

# consume items (ration, potion), evaluated based on the event log
for item in c.consumables:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=ConsumeItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  sampling_weight=4-level if level < 4 else 1))
    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=ConsumeItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  reward_to='team'))

# harvest items, evaluated based on the event log
for item in HARVEST_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=HarvestItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  sampling_weight=4-level if level < 4 else 1))
    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=HarvestItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  reward_to='team'))

# list items, evaluated based on the event log
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=ListItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  sampling_weight=4-level if level < 4 else 1))
    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=ListItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  reward_to='team'))

# buy items, evaluated based on the event log
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=BuyItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  sampling_weight=4-level if level < 4 else 1))
    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=BuyItem,
                                  eval_fn_kwargs={'item': item, 'level': level,
                                                  'quantity': quantity},
                                  reward_to='team'))

# fully armed, evaluated based on the current player/inventory status
for style in COMBAT_STYLE:
  for level in LEVEL_GOAL:
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(TaskSpec(eval_fn=FullyArmed,
                                  eval_fn_kwargs={'combat_style': style, 'level': level,
                                                  'num_agent': num_agent},
                                  reward_to='team'))


if __name__ == '__main__':
  import psutil
  from contextlib import contextmanager
  import multiprocessing as mp
  import numpy as np
  import dill

  @contextmanager
  def create_pool(num_proc):
    pool = mp.Pool(processes=num_proc)
    yield pool
    pool.close()
    pool.join()

  # 3495 task specs: divide the specs into chunks
  num_workers = round(psutil.cpu_count(logical=False)*0.7)
  spec_chunks = np.array_split(task_spec, num_workers)
  with create_pool(num_workers) as pool:
    chunk_results = pool.map(check_task_spec, spec_chunks)

  num_error = 0
  for results in chunk_results:
    for result in results:
      if result["runnable"] is False:
        print("ERROR: ", result["spec_name"])
        num_error += 1
  print("Total number of errors: ", num_error)

  # test if the task spec is pickalable
  with open('sample_curriculum.pkl', 'wb') as f:
    dill.dump(task_spec, f)
