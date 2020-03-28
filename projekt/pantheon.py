from pdb import set_trace as T

import ray
import time

from collections import defaultdict

from forge.blade.lib.log import BlobSummary
from forge.blade.lib.utils import printf

from forge.ethyr.torch import Model
from forge.trinity.ascend import Ascend, runtime, waittime

from forge.ethyr.experience import Rollout, RolloutManager

from forge.ethyr.torch import optim
from forge.ethyr.torch.param import getParameters, setParameters

import projekt

@ray.remote(num_gpus=0)
class Pantheon(Ascend):
   '''Cluster level infrastructure layer

  This module aggregates gradients across all server level 
   environments and updates model weights using Adam.

   It also demonstrates logging and snapshotting functionality 
   through the Quill and Model libraries, respectively.'''

   def __init__(self, config, idx):
      '''Initializes a copy of the model, which keeps
      track of the weights for the optimizer.

      Args:
         trinity : A Trinity object as shown in __main__
         config  : A Config object as shown in __main__
         idx     : Unused hardware index
      '''
      super().__init__(config, idx)
      self.config   = config
      self.rollouts = {}                                                      
      self.n        = 0
      self.nPkt     = 0

      self.uninit   = True 
      config.DEVICE = 'cpu'
      device        = config.DEVICE
      self.net      = projekt.Policy(config).to(device)
      self.manager  = RolloutManager(config)

      self.workerName = 'Pantheon {}'.format(self.idxStr)
      self.first = True

   def recvModel(self):
      packets = self.recv('Model')
      packets = [e for e in packets]
      if len(packets) > 0:
         weights = packets[-1]
         setParameters(self.net, weights)

         if self.uninit:
            self.uninit = False
            printf(self.workerName, 'Received Model')

   @waittime
   def recvExperience(self):
      packets = self.recv('Experience')
      returns = []
      for pkt in packets:
         #print('Packet: {}'.format(pkt))
         returns.append(pkt)

      self.nPkt += len(returns)
      return returns

   def init(self, trinity):
      self.trinity = trinity
      return self.workerName, 'Initialized'

   @runtime
   def step(self):
      '''Broadcasts updated weights to server level God optimizer nodes.
      Performs an Adam step once optimizers return a batch of gradients.

      Returns:
         perf  : Log message describing agent performance
         stats : Log message describing data collected
         log   : Dictionary of logs containing infrastructure usage data
      ''' 
      self.recvModel()

      trinity = self.trinity
      packets = self.recvExperience()
      for packet in packets:
         self.manager.collectInputs(packet)
         self.net(packet, self.manager)
         rollouts, _ = self.manager.step()

         for k, rollout in rollouts.items():
            assert k not in self.rollouts
            self.rollouts[k] = rollout
            self.n += rollout.time

      if len(packets) == 0:
         time.sleep(0.1)

      if self.n > self.config.SERVER_UPDATES:
         rollouts      = self.rollouts
         self.rollouts = {}

         if not self.config.TEST:
            optim.backward(rollouts, self.config)
            grads = self.net.grads()
            Ascend.send(trinity.cluster, grads, 'Gradients')

         update = (len(rollouts), self.n, self.nPkt)
         Ascend.send(trinity.quill, update, 'Pantheon_Updates')
         Ascend.send(trinity.quill, self.logs(), 'Pantheon_Utilization')
         self.nPkt = 0
         self.n    = 0



