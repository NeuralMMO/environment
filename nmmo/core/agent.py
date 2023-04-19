
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

  def __call__(self, obs):
    '''Used by scripted agents to compute actions. Override in subclasses.

    Args:
        obs: Agent observation provided by the environment
    '''
