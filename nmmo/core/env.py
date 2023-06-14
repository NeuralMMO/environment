import functools
import random
from typing import Any, Dict, List, Callable
from collections import defaultdict
from ordered_set import OrderedSet

import gym
import numpy as np
from pettingzoo.utils.env import AgentID, ParallelEnv

import nmmo
from nmmo.core import realm
from nmmo.core.config import Default
from nmmo.core.observation import Observation
from nmmo.core.tile import Tile
from nmmo.entity.entity import Entity
from nmmo.systems.item import Item
from nmmo.task import task_api
from nmmo.task.game_state import GameStateGenerator
from scripted.baselines import Scripted

class Env(ParallelEnv):
  # Environment wrapper for Neural MMO using the Parallel PettingZoo API

  #pylint: disable=no-value-for-parameter
  def __init__(self,
               config: Default = nmmo.config.Default(),
               seed = None):
    self._init_random(seed)

    super().__init__()

    self.config = config
    self.realm = realm.Realm(config)
    self.obs = None

    self.possible_agents = list(range(1, config.PLAYER_N + 1))
    self._dead_agents = set()
    self._episode_stats = defaultdict(lambda: defaultdict(float))
    self._dead_this_tick = None
    self.scripted_agents = OrderedSet()

    self._gamestate_generator = GameStateGenerator(self.realm, self.config)
    self.game_state = None
    # Default task: rewards 1 each turn agent is alive
    self.tasks = task_api.nmmo_default_task(self.possible_agents)
    self.agent_task_map = None

  # pylint: disable=method-cache-max-size-none
  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent: int):
    '''Neural MMO Observation Space

    Args:
        agent: Agent ID

    Returns:
        observation: gym.spaces object contained the structured observation
        for the specified agent. Each visible object is represented by
        continuous and discrete vectors of attributes. A 2-layer attentional
        encoder can be used to convert this structured observation into
        a flat vector embedding.'''

    def box(rows, cols):
      return gym.spaces.Box(
          low=-2**20, high=2**20,
          shape=(rows, cols),
          dtype=np.float32)

    obs_space = {
      "CurrentTick": gym.spaces.Discrete(1),
      "AgentId": gym.spaces.Discrete(1),
      "Tile": box(self.config.MAP_N_OBS, Tile.State.num_attributes),
      "Entity": box(self.config.PLAYER_N_OBS, Entity.State.num_attributes),
    }

    if self.config.ITEM_SYSTEM_ENABLED:
      obs_space["Inventory"] = box(self.config.INVENTORY_N_OBS, Item.State.num_attributes)

    if self.config.EXCHANGE_SYSTEM_ENABLED:
      obs_space["Market"] = box(self.config.MARKET_N_OBS, Item.State.num_attributes)

    if self.config.PROVIDE_ACTION_TARGETS:
      obs_space['ActionTargets'] = self.action_space(None)

    return gym.spaces.Dict(obs_space)

  def _init_random(self, seed):
    if seed is not None:
      np.random.seed(seed)
      random.seed(seed)

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    '''Neural MMO Action Space

    Args:
        agent: Agent ID

    Returns:
        actions: gym.spaces object contained the structured actions
        for the specified agent. Each action is parameterized by a list
        of discrete-valued arguments. These consist of both fixed, k-way
        choices (such as movement direction) and selections from the
        observation space (such as targeting)'''

    actions = {}
    for atn in sorted(nmmo.Action.edges(self.config)):
      if atn.enabled(self.config):

        actions[atn] = {}
        for arg in sorted(atn.edges):
          n = arg.N(self.config)
          actions[atn][arg] = gym.spaces.Discrete(n)

        actions[atn] = gym.spaces.Dict(actions[atn])

    return gym.spaces.Dict(actions)

  ############################################################################
  # Core API

  # TODO: This doesn't conform to the PettingZoo API
  # pylint: disable=arguments-renamed
  def reset(self, map_id=None, seed=None, options=None,
            make_task_fn: Callable=None):
    '''OpenAI Gym API reset function

    Loads a new game map and returns initial observations

    Args:
        map_id: Map index to load. Selects a random map by default
        seed: random seed to use
        make_task_fn: A function to make tasks

    Returns:
        observations, as documented by _compute_observations()

    Notes:
        Neural MMO simulates a persistent world. Ideally, you should reset
        the environment only once, upon creation. In practice, this approach
        limits the number of parallel environment simulations to the number
        of CPU cores available. At small and medium hardware scale, we
        therefore recommend the standard approach of resetting after a long
        but finite horizon: ~1000 timesteps for small maps and
        5000+ timesteps for large maps
    '''

    self._init_random(seed)
    self.realm.reset(map_id)
    self._dead_agents = set()
    self._episode_stats.clear()
    self._dead_this_tick = {}

    # check if there are scripted agents
    for eid, ent in self.realm.players.items():
      if isinstance(ent.agent, Scripted):
        self.scripted_agents.add(eid)

    self.obs = self._compute_observations()
    self._gamestate_generator = GameStateGenerator(self.realm, self.config)

    if make_task_fn is not None:
      self.tasks = make_task_fn()
    else:
      for task in self.tasks:
        task.reset()
    self.agent_task_map = self._map_task_to_agent()

    return {a: o.to_gym() for a,o in self.obs.items()}

  def _map_task_to_agent(self):
    agent_task_map: Dict[int, List[task_api.Task]] = {}
    for task in self.tasks:
      for agent_id in task.assignee:
        if agent_id in agent_task_map:
          agent_task_map[agent_id].append(task)
        else:
          agent_task_map[agent_id] = [task]
    return agent_task_map

  def step(self, actions: Dict[int, Dict[str, Dict[str, Any]]]):
    '''Simulates one game tick or timestep

    Args:
        actions: A dictionary of agent decisions of format::

              {
                agent_1: {
                    action_1: [arg_1, arg_2],
                    action_2: [...],
                    ...
                },
                agent_2: {
                    ...
                },
                ...
              }

          Where agent_i is the integer index of the i\'th agent

          The environment only evaluates provided actions for provided
          gents. Unprovided action types are interpreted as no-ops and
          illegal actions are ignored

          It is also possible to specify invalid combinations of valid
          actions, such as two movements or two attacks. In this case,
          one will be selected arbitrarily from each incompatible sets.

          A well-formed algorithm should do none of the above. We only
          Perform this conditional processing to make batched action
          computation easier.

    Returns:
        (dict, dict, dict, None):

        observations:
          A dictionary of agent observations of format::

              {
                agent_1: obs_1,
                agent_2: obs_2,
                ...
              }

          Where agent_i is the integer index of the i\'th agent and
          obs_i is specified by the observation_space function.

        rewards:
          A dictionary of agent rewards of format::

              {
                agent_1: reward_1,
                agent_2: reward_2,
                ...
              }

          Where agent_i is the integer index of the i\'th agent and
          reward_i is the reward of the i\'th' agent.

          By default, agents receive -1 reward for dying and 0 reward for
          all other circumstances. Override Env.reward to specify
          custom reward functions

        dones:
          A dictionary of agent done booleans of format::

              {
                agent_1: done_1,
                agent_2: done_2,
                ...
              }

          Where agent_i is the integer index of the i\'th agent and
          done_i is a boolean denoting whether the i\'th agent has died.

          Note that obs_i will be a garbage placeholder if done_i is true.
          This is provided only for conformity with PettingZoo. Your
          algorithm should not attempt to leverage observations outside of
          trajectory bounds. You can omit garbage obs_i values by setting
          omitDead=True.

        infos:
          A dictionary of agent infos of format:

              {
                agent_1: None,
                agent_2: None,
                ...
              }

          Provided for conformity with PettingZoo
    '''
    assert self.obs is not None, 'step() called before reset'
    # Add in scripted agents' actions, if any
    if self.scripted_agents:
      actions = self._compute_scripted_agent_actions(actions)

    # Drop invalid actions of BOTH neural and scripted agents
    #   we don't need _deserialize_scripted_actions() anymore
    actions = self._validate_actions(actions)
    # Execute actions
    self._dead_this_tick = self.realm.step(actions)
    dones = {}
    for agent_id in self.agents:
      if agent_id in self._dead_this_tick or self.realm.tick >= self.config.HORIZON:
        self._dead_agents.add(agent_id)
        self._episode_stats[agent_id]["death_tick"] = self.realm.tick
        dones[agent_id] = True
      else:
        dones[agent_id] = False

    # Generate obs for each agent in self.agents
    self.obs = self._compute_observations()
    gym_obs = {a: o.to_gym() for a,o in self.obs.items()}

    # Generate rewards, infos for each agent in self.agents
    rewards, infos = self._compute_rewards()
    for k,r in rewards.items():
      self._episode_stats[k]['reward'] += r

    # When the episode ends, add the episode stats to the info of the last agents
    if len(self._dead_agents) == len(self.possible_agents):
      for agent_id, stats in self._episode_stats.items():
        if agent_id not in infos:
          infos[agent_id] = {}
        infos[agent_id]["episode_stats"] = stats

    # NOTE: all obs, rewards, dones, infos have data for each agent in self.agents
    return gym_obs, rewards, dones, infos

  def _validate_actions(self, actions: Dict[int, Dict[str, Dict[str, Any]]]):
    '''Deserialize action arg values and validate actions
       For now, it does a basic validation (e.g., value is not none).
    '''
    validated_actions = {}

    for ent_id, atns in actions.items():
      if ent_id not in self.realm.players:
        #assert ent_id in self.realm.players, f'Entity {ent_id} not in realm'
        continue # Entity not in the realm -- invalid actions

      entity = self.realm.players[ent_id]
      if not entity.alive:
        #assert entity.alive, f'Entity {ent_id} is dead'
        continue # Entity is dead -- invalid actions

      validated_actions[ent_id] = {}

      for atn, args in sorted(atns.items()):
        action_valid = True
        deserialized_action = {}

        if not atn.enabled(self.config):
          action_valid = False
          break

        for arg, val in sorted(args.items()):
          obj = arg.deserialize(self.realm, entity, val)
          if obj is None:
            action_valid = False
            break
          deserialized_action[arg] = obj

        if action_valid:
          validated_actions[ent_id][atn] = deserialized_action

    return validated_actions

  def _compute_scripted_agent_actions(self, actions: Dict[int, Dict[str, Dict[str, Any]]]):
    '''Compute actions for scripted agents and add them into the action dict'''
    for eid in self.scripted_agents:
      # remove the dead scripted agent from the list
      if eid in self._dead_agents or eid not in self.realm.players:
        self.scripted_agents.discard(eid)
        continue

      # override the provided scripted agents' actions
      actions[eid] = self.realm.players[eid].agent(self.obs[eid])

    return actions

  def _compute_observations(self):
    '''Create an Observation object for each agent in self.agents'''
    obs = {}
    market = Item.Query.for_sale(self.realm.datastore) # the same for all agents

    # dummy obs
    dummy_tiles = np.zeros((1, len(Tile.State.attr_name_to_col)))
    dummy_entities = np.zeros((1, len(Entity.State.attr_name_to_col)))
    dummy_inventory = np.zeros((1, len(Item.State.attr_name_to_col)))
    dummy_market = np.zeros((1, len(Item.State.attr_name_to_col)))

    for agent_id in self.agents:
      if agent_id not in self.realm.players:
        # return dummy obs for the agents in dead_this_tick
        obs[agent_id] = Observation(self.config, self.realm.tick, agent_id,
                                    dummy_tiles, dummy_entities, dummy_inventory, dummy_market)
      else:
        agent = self.realm.players.get(agent_id)
        agent_r = agent.row.val
        agent_c = agent.col.val

        visible_entities = Entity.Query.window(
            self.realm.datastore,
            agent_r, agent_c,
            self.config.PLAYER_VISION_RADIUS
        )
        visible_tiles = Tile.Query.window(
            self.realm.datastore,
            agent_r, agent_c,
            self.config.PLAYER_VISION_RADIUS)

        inventory = Item.Query.owned_by(self.realm.datastore, agent_id)

        # NOTE: the tasks for each agent is in self.agent_task_map, and task embeddings are
        #   available in each task instance, via task.embedding
        # CHECK ME: do we want to pass in self.agent_task_map[agent_id],
        #   so that we can include task embedding in the obs?
        obs[agent_id] = Observation(self.config, self.realm.tick, agent_id,
                                    visible_tiles, visible_entities, inventory, market)
    return obs

  def _compute_rewards(self):
    '''Computes the reward for the specified agent

    Override this method to create custom reward functions. You have full
    access to the environment state via self.realm. Our baselines do not
    modify this method; specify any changes when comparing to baselines

    Args:
        player: player object

    Returns:
        reward:
          The reward for the actions on the previous timestep of the
          entity identified by ent_id.
    '''
    # Initialization
    agents = set(self.agents)
    infos = {agent_id: {'task': {}} for agent_id in agents}
    rewards = defaultdict(int)

    # Compute Rewards and infos
    self.game_state = self._gamestate_generator.generate(self.realm, self.obs)
    for task in self.tasks:
      task_rewards, task_infos = task.compute_rewards(self.game_state)
      for agent_id, reward in task_rewards.items():
        if agent_id in agents:
          rewards[agent_id] = rewards.get(agent_id,0) + reward
          infos[agent_id]['task'][task.name] = task_infos[agent_id] # progress

    # Make sure the dead agents return the rewards of -1
    for agent_id in self._dead_this_tick:
      rewards[agent_id] = -1

    return rewards, infos

  ############################################################################
  # PettingZoo API
  ############################################################################

  def render(self, mode='human'):
    '''For conformity with the PettingZoo API only; rendering is external'''

  @property
  def agents(self) -> List[AgentID]:
    '''For conformity with the PettingZoo API only; rendering is external'''
    # "current" agents, which return obs: both alive and dead_this_tick
    agents = set(list(self.realm.players.keys()) + list(self._dead_this_tick.keys()))
    return list(agents)

  def close(self):
    '''For conformity with the PettingZoo API only; rendering is external'''

  def seed(self, seed=None):
    return self._init_random(seed)

  def state(self) -> np.ndarray:
    raise NotImplementedError

  metadata = {'render.modes': ['human'], 'name': 'neural-mmo'}
