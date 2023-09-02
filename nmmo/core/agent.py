class Agent:
  policy   = 'Neural'

  def __init__(self, config, idx):
    '''Base class for agents

    Args:
      config: A Config object
      idx: Unique AgentID int
    '''
    self.config = config
    self.iden   = idx
    self._np_random = None

  def __call__(self, obs):
    '''Used by scripted agents to compute actions. Override in subclasses.

    Args:
        obs: Agent observation provided by the environment
    '''

  def set_rng(self, np_random):
    '''Set the random number generator for the agent for reproducibility

    Args:
        np_random: A numpy random.Generator object
    '''
    self._np_random = np_random

class Scripted(Agent):
  '''Base class for scripted agents'''
  policy   = 'Scripted'
