from pdb import set_trace as T
from copy import deepcopy

import ray
import ray.experimental.signal as signal

from forge.blade.lib.utils import printf
from forge.trinity.ascend import Ascend, runtime, waittime

from forge.ethyr.experience import RolloutManager
from forge.ethyr.torch.param import setParameters

import projekt

@ray.remote#(num_gpus=1)
class Sword(Ascend):
   '''Client level infrastructure demo

   This environment server runs a subset of the
   agents associated with a single server and
   computes model updates over collected rollouts

   At small scale, each server is collocated with a
   single client on the same CPU core. For larger
   experiments with multiple clients, decorate this
   class with @ray.remote to enable sharding.'''

   def __init__(self, config, idx):
      '''Initializes a model and relevent utilities
                                                                              
      Args:                                                                   
         trinity : A Trinity object as shown in __main__                      
         config  : A Config object as shown in __main__                       
         idx     : Unused hardware index                                      
      '''
      super().__init__(config, idx)
      #config.DEVICE = 'cuda'
      config        = deepcopy(config)
      device        = config.DEVICE
      self.config   = config 

      self.net = projekt.Policy(config).to(device).eval()
      self.uninit = True

      self.workerName = 'Sword {}'.format(self.idxStr)

   def init(self, trinity):
      self.trinity = trinity
      return self.workerName, 'Initialized'

   def recvModel(self):
      #Receive weight packets
      packet = self.recv('Model')
      packet = [e for e in packet]

      #Sync model weights; batch obs; compute forward pass
      if len(packet) > 0:
         setParameters(self.net, packet[-1])
         if self.uninit:
            self.uninit = False
            printf(self.workerName, 'Received Model')

      return packet

   def optimHash(self):
      return self.idx % self.config.NPANTHEON

   @waittime
   def sync(self, packet):
      dst = self.optimHash()
      Ascend.send(self.trinity.pantheon[dst], packet, 'Experience')

   @runtime
   def step(self, packet):
      '''Synchronizes weights from upstream; computes
      agent decisions; computes policy updates.
                                                                              
      Args:                                                                   
         packet   : An IO object specifying observations
         weights  : An optional parameter vector to replace model weights
         backward : (bool) Whether of not a backward pass should be performed  
      Returns:                                                                   
         data    : The same IO object populated with action decisions
         grads   : A vector of gradients aggregated across trajectories
         summary : A BlobSummary object logging agent statistics
      '''   
      #Sync model
      self.recvModel()

      #Compute forward pass
      self.net(packet, None)

      #Send experience and logs
      self.sync(packet)

      Ascend.send(self.trinity.quill, self.logs(), 'Sword_Utilization')

      return packet

