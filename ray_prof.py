from pdb import set_trace as T
import numpy as np
import pickle
import json
import ray
import time

#Core environment and configuration
from experiments import Experiment, Config
from forge.blade.core import Realm
from forge.blade import IO

def timeit(f, n=100):
   actor = Remote.remote()
   data  = Data()

   envTime = 0
   ioTime  = 0
   fTime   = 0

   for i in range(n):
      x, envt, iot    =  data.step()
      envTime += envt/n
      ioTime  += iot/n
      
      t = time.time()
      f(actor, x)
      fTime += (time.time() - t)/n

   print('Func: {}, Env: {:.4f}, IO: {:.4f}, Func: {:.4f}'.format(f.__name__, envTime, ioTime, fTime))


@ray.remote
class Remote:
   def f(self, x):
      return

class Data:
   def __init__(self):
      #Define an experiment configuration
      config = Experiment('demo', Config).init(TEST=True)
      self.config = config

      #Initialize the environment and policy
      env                        = Realm(config)
      obs, rewards, dones, infos = env.reset()

      self.env = env

      #Warmup
      for i in range(15):
         env.step({})

   def step(self):
      #Submit actions
      t = time.time()
      nxtObs, rewards, dones, info = self.env.step({})
      envt = time.time() - t
      t = time.time()
      inp, deserialize, n = IO.inputs(nxtObs, rewards, dones, self.config)
      iot = time.time() - t
      return inp, envt, iot
      
def ray_default(actor, x):
   actor.f.remote(x)

if __name__ == '__main__':
   ray.init()
   timeit(ray_default)
