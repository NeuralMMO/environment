'''Manual test for creating learning curriculum manually'''
# pylint: disable=invalid-name,redefined-outer-name,bad-builtin

import nmmo
import nmmo.lib.material as Material
from nmmo.task import base_predicates as bp
from nmmo.task.task_api import OngoingTask, make_team_tasks
from nmmo.task import constraint as c


EVENT_NUMBER_GOAL = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 30, 50]
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

""" task_spec is a list of tuple (reward_to, predicate class, kwargs)

    each tuple in the task_spec will create tasks for a team in teams

    reward_to: must be in ['team', 'agent']
      * 'team' create a single team task, in which all team members get rewarded
      * 'agent' create a task for each agent, in which only the agent gets rewarded

    predicate class from the base predicates or custom predicates like above

    kwargs are the additional args that go into predicate. There are also special keys
      * 'target' must be ['left_team', 'right_team', 'left_team_leader', 'right_team_leader']
          these str will be translated into the actual agent ids
      * 'task_cls' is optional. If not provided, the standard Task is used. """
task_spec = []

# explore, eat, drink, attack any agent, harvest any item, level up any skill
#   which can happen frequently
essential_skills = ['GO_FARTHEST', 'EAT_FOOD', 'DRINK_WATER',
                    'SCORE_HIT', 'HARVEST_ITEM', 'LEVEL_UP']
for event_code in essential_skills:
  task_spec += [('agent', bp.CountEvent, {'event': event_code, 'N': cnt})
                for cnt in EVENT_NUMBER_GOAL]

# item/market skills, which happen less frequently or should not do too much
item_skills = ['CONSUME_ITEM', 'GIVE_ITEM', 'DESTROY_ITEM', 'EQUIP_ITEM',
               'GIVE_GOLD', 'LIST_ITEM', 'EARN_GOLD', 'BUY_ITEM']
for event_code in item_skills:
  task_spec += [('agent', bp.CountEvent, {'event': event_code, 'N': cnt})
                for cnt in INFREQUENT_GOAL] # less than 10

# find resource tiles
for resource in Material.Harvestable:
  for reward_to in ['agent', 'team']:
    task_spec.append((reward_to, bp.CanSeeTile, {'tile_type': resource}))

# stay alive ... like ... for 300 ticks
# i.e., getting incremental reward for each tick alive as an individual or a team
for reward_to in ['agent', 'team']:
  for num_tick in STAY_ALIVE_GOAL:
    task_spec.append((reward_to, bp.TickGE, {'num_tick': num_tick}))

# protect the leader: get reward for each tick the leader is alive
task_spec.append(('team', bp.StayAlive, {'target': 'my_team_leader', 'task_cls': OngoingTask}))

# want the other team or team leader to die
for target in ['left_team', 'left_team_leader', 'right_team', 'right_team_leader']:
  task_spec.append(('team', bp.AllDead, {'target': target}))

# occupy the center tile, assuming the Medium map size
# TODO: it'd be better to have some intermediate targets toward the center
for reward_to in ['agent', 'team']:
  task_spec.append((reward_to, bp.OccupyTile, {'row': 80, 'col': 80})) # TODO: get config

# form a tight formation, for a certain number of ticks
def PracticeFormation(gs, subject, dist, num_tick):
  return bp.AllMembersWithinRange(gs, subject, dist) * bp.TickGE(gs, subject, num_tick)
for dist in [1, 3, 5, 10]:
  task_spec += [('team', PracticeFormation, {'dist': dist, 'num_tick': num_tick})
                for num_tick in STAY_ALIVE_GOAL]

# find the other team leader
for reward_to in ['agent', 'team']:
  for target in ['left_team_leader', 'right_team_leader']:
    task_spec.append((reward_to, bp.CanSeeAgent, {'target': target}))

# find the other team (any agent)
for reward_to in ['agent']: #, 'team']:
  for target in ['left_team', 'right_team']:
    task_spec.append((reward_to, bp.CanSeeGroup, {'target': target}))

# explore the map -- sum the l-inf distance traveled by all subjects
for dist in [10, 20, 30, 50, 100]: # each agent
  task_spec.append(('agent', bp.DistanceTraveled, {'dist': dist}))
for dist in [30, 50, 70, 100, 150, 200, 300, 500]: # summed over all team members
  task_spec.append(('team', bp.DistanceTraveled, {'dist': dist}))

# level up a skill
for skill in SKILLS:
  for level in LEVEL_GOAL:
    # since this is an agent task, num_agent must be 1
    task_spec.append(('agent', bp.AttainSkill, {'skill': skill, 'level': level, 'num_agent': 1}))

# make attain skill a team task by varying the number of agents
for skill in SKILLS:
  for level in LEVEL_GOAL:
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(('team', bp.AttainSkill,
                          {'skill': skill, 'level': level,'num_agent': num_agent}))

# practice specific combat style
for style in COMBAT_STYLE:
  for cnt in EVENT_NUMBER_GOAL:
    task_spec.append(('agent', bp.ScoreHit, {'combat_style': style, 'N': cnt}))
  for cnt in TEAM_NUMBER_GOAL:
    task_spec.append(('team', bp.ScoreHit, {'combat_style': style, 'N': cnt}))

# defeat agents of a certain level as a team
for agent_type in ['player', 'npc']: # c.AGENT_TYPE_CONSTRAINT
  for level in LEVEL_GOAL:
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(('team', bp.DefeatEntity,
                          {'agent_type': agent_type, 'level': level, 'num_agent': num_agent}))

# hoarding gold -- evaluated on the current gold
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(('agent', bp.HoardGold, {'amount': amount}))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(('team', bp.HoardGold, {'amount': amount}))

# earning gold -- evaluated on the total gold earned by selling items
# does NOT include looted gold
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(('agent', bp.EarnGold, {'amount': amount}))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(('team', bp.EarnGold, {'amount': amount}))

# spending gold, by buying items
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(('agent', bp.SpendGold, {'amount': amount}))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(('team', bp.SpendGold, {'amount': amount}))

# making profits by trading -- only buying and selling are counted
for amount in EVENT_NUMBER_GOAL:
  task_spec.append(('agent', bp.MakeProfit, {'amount': amount}))
for amount in TEAM_NUMBER_GOAL:
  task_spec.append(('team', bp.MakeProfit, {'amount': amount}))

# managing inventory space
def PracticeInventoryManagement(gs, subject, space, num_tick):
  return bp.InventorySpaceGE(gs, subject, space) * bp.TickGE(gs, subject, num_tick)
for space in [2, 4, 8]:
  task_spec += [('agent', PracticeInventoryManagement, {'space': space, 'num_tick': num_tick})
                for num_tick in STAY_ALIVE_GOAL]

# own item, evaluated on the current inventory
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(('agent', bp.OwnItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(('team', bp.OwnItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

# equip item, evaluated on the current inventory and equipment status
for item in EQUIP_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    task_spec.append(('agent', bp.EquipItem,
                      {'item': item, 'level': level, 'num_agent': 1}))

    # team task
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(('team', bp.EquipItem,
                          {'item': item, 'level': level, 'num_agent': num_agent}))

# consume items (ration, potion), evaluated based on the event log
for item in c.consumables:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(('agent', bp.ConsumeItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(('team', bp.ConsumeItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

# harvest items, evaluated based on the event log
for item in HARVEST_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(('agent', bp.HarvestItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(('team', bp.HarvestItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

# list items, evaluated based on the event log
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(('agent', bp.ListItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(('team', bp.ListItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

# buy items, evaluated based on the event log
for item in ALL_ITEM:
  for level in LEVEL_GOAL:
    # agent task
    for quantity in ITEM_NUM_GOAL:
      if level + quantity <= 6 or quantity == 1: # heuristic prune
        task_spec.append(('agent', bp.BuyItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

    # team task
    for quantity in TEAM_ITEM_GOAL:
      if level + quantity <= 10 or quantity == 1: # heuristic prune
        task_spec.append(('team', bp.BuyItem,
                          {'item': item, 'level': level, 'quantity': quantity}))

# fully armed, evaluated based on the current player/inventory status
for style in COMBAT_STYLE:
  for level in LEVEL_GOAL:
    for num_agent in AGENT_NUM_GOAL:
      if level + num_agent <= 6 or num_agent == 1: # heuristic prune
        task_spec.append(('team', bp.FullyArmed,
                          {'combat_style': style, 'level': level, 'num_agent': num_agent}))


if __name__ == '__main__':
  # pylint: disable=bare-except
  import psutil
  from contextlib import contextmanager
  import multiprocessing as mp
  import numpy as np
  import pickle

  @contextmanager
  def create_pool(num_proc):
    pool = mp.Pool(processes=num_proc)
    yield pool
    pool.close()
    pool.join()

  def check_task_spec(spec_list):
    teams = {0:[1,2,3], 1:[4,5], 2:[6,7], 3:[8,9], 4:[10,11]}
    config = nmmo.config.Default()
    env = nmmo.Env(config)
    for idx, single_spec in enumerate(spec_list):
      # pylint: disable=cell-var-from-loop
      test_task = make_team_tasks(teams, [single_spec])
      try:
        env.reset(make_task_fn=lambda: test_task)
        for _ in range(3):
          env.step({})
      except:
        print('invalid task spec:', single_spec)

      if idx > 0 and idx % 50 == 0:
        print(idx, 'task specs checked.')

  # 3590 task specs: divide the specs into chunks
  num_cores = psutil.cpu_count(logical=False)
  spec_chunks = np.array_split(task_spec, num_cores)
  with create_pool(num_cores) as pool:
    pool.map(check_task_spec, spec_chunks)

  # print(sample_task[0].name)
  # if len(sample_task) > 1:
  #   print(sample_task[-1].name)

  # test if the task spec is pickalable
  with open('manual_curriculum.pkl', 'wb') as f:
    pickle.dump(task_spec, f)
