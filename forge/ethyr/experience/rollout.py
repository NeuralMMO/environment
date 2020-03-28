from pdb import set_trace as TT

from collections import defaultdict
import numpy as np

from forge.blade.lib.log import Blob

class Output:
   def __init__(self, atnArgKey, atnLogits, atnIdx, value):
      '''Data structure specifying a chosen action

      Args:
         atnArgKey : Action-Argument formatted string                           
         atnLogits : Action logits                                              
         atnsIdx   : Argument indices sampled from logits                       
         value     : Value function prediction  
      '''
      self.atnArgKey = atnArgKey
      self.atnLogits = atnLogits
      self.atnIdx    = atnIdx      
      self.value     = value
      self.reward    = 0 #Probably being assigned wrong

class Rollout:
   def __init__(self, config):
      '''Rollout object used internally by RolloutManager

      Args:
         config: A configuration object
      '''
      self.actions = defaultdict(list)
      self.rewards = []

      self.done = False
      self.time = -1

      #Logger
      self.config  = config
      self.blob    = None

   def __len__(self):
      '''Length of a rollout

      Returns:
         lifetime: Number of timesteps the agent has survived
      '''
      return self.blob.lifetime

   def inputs(self, reward, key):
      '''Collects input data to internal buffers

      Args:
         reward : The reward received by the agent for its last action
         key    : The ID associated with the agent
      '''
      #Also check if blob is not none. This prevents
      #recording the first reward of a partial trajectory
      if reward is not None:
         self.rewards.append(reward)

      self.time += 1

   def outputs(self, atnArgKey, atnLogits, atnIdx, value):
      '''Collects output data to internal buffers

      Args:
         atnArgKey : Action-Argument formatted string                           
         atnLogits : Action logits                                              
         atnsIdx   : Argument indices sampled from logits                       
         value     : Value function prediction  
      '''
      output = Output(atnArgKey, atnLogits, atnIdx, value)
      self.actions[self.time].append(output)

   def finish(self):
      '''Called internally once the full rollout has been collected'''
      self.rewards.append(-1)
      #self.returns = self.gae(
      #      self.config.GAMMA,
      #      self.config.LAMBDA,
      #      self.config.HORIZON)

      self.returns = self.discount(self.config.GAMMA)
      for k, atns in self.actions.items():
         for e, ret in zip(atns, self.rewards):
            e.reward = ret
      
      self.lifespan = len(self.rewards)

   def gae(self, gamma, lamb, H):
      '''Applies generalized advantage estimation to the given trajectory
      
      Args:
         gamma: Reward discount factor
         gamma: GAE discount factor

      Returns:
         rewards: Discounted list of rewards
      '''
      r = self.rewards
      V = self.values

      L = len(r)
      returns = []
      for t in range(L):
         At, T = 0, min(L-t-1, H)
         for i in range(T):
            tt      = t + i
            deltaT  =  r[tt] + gamma*V[tt+1] - V[tt]
            At      += deltaT * (gamma*lamb)**i

         for out in self.actions[t]:
            out.returns = At
            
         returns.append(At)

      return returns

   def discount(self, gamma):
      '''Applies standard gamma discounting to the given trajectory
      
      Args:
         gamma: Reward discount factor

      Returns:
         rewards: Discounted list of rewards
      '''
      rets, N   = [], len(self.rewards)
      discounts = np.array([gamma**i for i in range(N)])
      rewards   = np.array(self.rewards)

      for idx in range(N):
         R_i = sum(rewards[idx:]*discounts[:N-idx])
         for out in self.actions[idx]:
            out.returns = R_i 
         
         rets.append(R_i)

      return rets
