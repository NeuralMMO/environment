from pdb import set_trace as T
from copy import deepcopy

import ray
import ray.experimental.signal as signal

from forge.blade.lib.utils import printf
from forge.blade import IO

from forge.ethyr.torch.param import setParameters
from forge.ethyr.torch import Model

import projekt
from projekt.god import Realm

class Evaluator:
   def __init__(self, config, envIdx):
      self.env                              = Realm(config, envIdx)
      self.obs, self.rewards, self.dones, _ = self.env.reset()

      self.net    = Model(projekt.Policy, config).net
      self.config = config

   def tick(self):
      obs, deserialize, nUpdates = IO.inputs(
               self.obs, self.rewards, self.dones,
               self.config, None)
      self.net(obs, None)
      actions = IO.outputs(obs, deserialize)
      self.obs, self.rewards, self.dones, _ = self.env.step(actions)
