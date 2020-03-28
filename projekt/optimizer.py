from pdb import set_trace as T
import ray

from forge.trinity.ascend import Ascend, runtime
from forge.ethyr.experience import RolloutManager

@ray.remote
class Optimizer(Ascend):
   def __init__(self, config, idx):
      super().__init__(config, idx)
      self.rollouts = {}
      self.blobs    = {}

   @runtime
   def step(self, rollouts={}, blobs=[], backward=False):
      for k, rollout in rollouts.items():
         assert k not in self.rollouts
         self.rollouts[k] = rollout 

      self.blobs += blobs
         
      if backward:
         optim.backward(rollouts, self.config)                                

      self.manager.clearInputs()                                           
      grads = self.net.grads()

