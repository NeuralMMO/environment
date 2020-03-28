from pdb import set_trace as T
import numpy as np

import ray
import time

from collections import defaultdict

from forge.blade import core
from forge.blade.lib.log import BlobSummary 

from forge.trinity.ascend import Ascend, runtime, waittime, Log
from forge.blade import IO

import projekt

class Realm(core.Realm):
   '''Example environment overrides'''
   def spawn(self):
      '''Example override of the spawn function'''
      return super().spawn()

      #Example override
      ent    = self.desciples[entID]
      packets[entID].reward = 0.05 * min(
          ent.resources.health.val,
          ent.resources.water.val,
          ent.resources.food.val)

   def reward(self, ent):
      '''Example override of the reward function'''
      return super().reward(ent)

@ray.remote
class God(Ascend):
   '''Server level infrastructure layer 

   This environment server runs a persistent game instance that distributes 
   agents observations and collects action decisions from client nodes.

   This infrastructure configuration is the direct opposite of the typical 
   MPI broadcast/recv paradigm, which operates an optimizer server over a 
   bank of environments. The MPI approach does not scale well to massively
   multiagent environments, as it assumes a many-to-one (as opposed to a 
   one-to-many) ratio of envs to optimizers.

   OpenAI Rapid style optimization does not fix this issue. Rapid solves 
   bandwidth constraints by collocating the agent policies with the 
   environment, but it still makes the same many-to-one assumption. 

   Note that you could probably get away with either MPI or Rapid style
   optimization on current instances of the environment because we only have
   128 agents, not thousands. However, we built with future scale in mind
   and invested in infrastructure early.''' 

   def __init__(self, config, idx):
      '''Initializes an environment and logging utilities

      Args:
         trinity : A Trinity object as shown in __main__
         config  : A Config object as shown in __main__
         idx     : Hardware index used to specify the game map
      '''
      super().__init__(config, idx)
      self.nPop, self.ent       = config.NPOP, 0
      self.config, self.idx     = config, idx
      self.nUpdates, self.grads = 0, []

      self.env   = Realm(config, idx)
      self.blobs = BlobSummary()

      self.obs, self.rewards, self.dones, _ = self.env.reset()
      self.workerName = 'God+Realm {}'.format(self.idxStr)

   def getEnv(self):
      '''Ray does not allow direct access to remote attributes

      Returns:
         env: A copy of the environment (returns a reference in local mode)
      '''
      return self.env

   def clientHash(self, entID):
      '''Hash function for mapping entID->client

      Returns:
         clientID: A client membership ID
      '''
      return self.idx % self.config.NSWORD
      return entID % self.config.NSWORD

   def distribute(self):
      '''Shards input data across clients using the Ascend async API

      Args:
         weights: A parameter vector to replace the current model weights

      Returns:
         rets: List of (data, gradients, blobs) tuples from the clients
      '''

      #Preprocess obs
      clientData, deserialize, nUpdates = IO.inputs(
         self.obs, self.rewards, self.dones, 
         self.config, self.clientHash)

      #Shard entities across clients
      return Ascend.distribute(self.trinity.sword,
            clientData, shard=[True]), deserialize

   @waittime
   def syn(dat):
      time.sleep(0.1)
      return ray.get(dat)

   def synchronize(self, rets, deserialize):
      '''Aggregates output data across shards with the Ascend async API

      Args:
         rets: List of (data, gradients, blobs) tuples from self.distrib()

      Returns:
         atnDict: Dictionary of actions to be submitted to the environment
      '''
      atnDict, gradList, blobList = None, [], []
      #for obs in self.syn(rets):
      for obs in super().synchronize(rets):
         #Process outputs
         atnDict = IO.outputs(obs, deserialize, atnDict)

         #Collect update
         if False and self.backward:
            self.grads.append(grads)
            self.blobs.add([blobs])

      return atnDict

   def init(self, trinity):
      self.trinity = trinity
      return self.workerName, 'Initialized'

   def step(self):
      '''Sync weights and compute a model update by collecting
      a batch of trajectories from remote clients.

      Args:
         recv: Upstream data from the cluster (in this case, a param vector)

      Returns:
         grads   : A vector of gradients aggregated across clients
         summary : A BlobSummary object logging agent statistics
         log     : Logging object for infrastructure timings
      '''
      Ascend.send(self.trinity.quill, self.logs(), 'God_Utilization')
      entLog = self.env.entLog()
      if len(entLog) > 0:
         Ascend.send(self.trinity.quill, entLog, 'Realm_Logs')
      self.tick()

   @runtime
   def tick(self):
      '''Simulate a single server tick and all remote clients.
      The optional data packet specifies a new model parameter vector

      Args:
         recv: Upstream data from the cluster (in this case, a param vector)
      '''
      #Make decisions
      packet, deserialize = self.distribute()
      actions             = self.synchronize(packet, deserialize)

      #Step the environment and all agents at once.
      #The environment handles action priotization etc.
      self.obs, self.rewards, self.dones, _ = self.env.step(actions)
