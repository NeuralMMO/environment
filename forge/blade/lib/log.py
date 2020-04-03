from pdb import set_trace as T
from collections import defaultdict
from collections import deque

from tqdm import tqdm
import numpy as np
import json, pickle
import time
import ray
import os

from forge.blade.lib.utils import EDA
from forge.blade.lib.enums import Material
from forge.blade.lib import enums
from copy import deepcopy

from forge.trinity.ascend import Ascend
from forge.blade.systems import visualizer

def Test(filename, msg='test'):
      file = open(filename,"w")
      print(msg,file=file)
      file.close()

class TimePacket:
   def __init__(self, x):
      self.time = time.time()
      self.val  = x

class TimeQueue:
   def __init__(self, tMinutes=10):
      self.histLen = 60*tMinutes
      self.data    = deque()

   def update(self, x):
      t = time.time()
      packet = TimePacket(x)
      self.data.append(packet)
      
      #Remove elements that are too old
      while len(self.data) > 0:
         pkt = self.data.popleft()
         if t - pkt.time < self.histLen:
            self.data.appendleft(pkt)
            break

   @property
   def val(self):
      if len(self.data) < 2:
         return 0

      t   = self.data[-1].time - self.data[0].time
      val = sum([e.val for e in self.data])
      return val / t 

class Stat:
   def __init__(self, k=0.99):
      self.data = []
      self.min  = np.inf
      self.max  = -np.inf
      self.eda  = EDA(k)
      self.val  = 0

   def update(self, x, add=False):
      if add:
         x = self.val + x

      self.val = x
      self.eda.update(x)
      #self.data.append(x)
      if x < self.min:
         self.min = x
      if x > self.max:
         self.max = x

   @property
   def summary(self):
      return self.eda.eda

class Bar(tqdm):
   def __init__(self, position=0, title='title', form=None):
      lbar = '{desc}: {percentage:3.0f}%|'
      bar = '{bar}'
      rbar  = '| [' '{elapsed}{postfix}]'
      fmt = ''.join([lbar, bar, rbar])

      if form is not None:
         fmt = form

      super().__init__(
            total=100,
            position=position,
            bar_format=fmt)

      self.title(title)

   def percent(self, val):
      self.update(100*val - self.n)

   def title(self, txt):
      self.desc = txt
      self.refresh()

# Not in use
class Logger:                                                                 
   def __init__(self, middleman):                                             
      self.items     = 'reward lifetime value'.split()                              
      self.middleman = middleman                                              
      self.tick      = 0                                                      

                                                                              
   def update(self, lifetime_mean, reward_mean, value_mean,
              lifetime_std, reward_std, value_std):
      data = {}                                                               
      data['lifetime'] = max(0.0,lifetime_mean)
      data['lifetimeupper'] = max(0.0,lifetime_std)
      data['lifetimelower'] = max(0.0,lifetime_std)
      data[visualizer.Config.XAXIS]     = self.tick                                            
      # data['reward']   = reward_mean
      # data['value']    = value_mean
      # data['lifetime_std']  = lifetime_std
      # data['reward_std']    = reward_std
      # data['value_std']     = value_std
                                                                              
      self.tick += 1                                                          
      self.middleman.setData.remote(data)

class BlobSummary:
   def __init__(self):
      self.nRollouts = 0
      self.nUpdates  = 0

      self.lifetime = []
      self.reward   = [] 
      self.value    = []

   def add(self, blobs):
      for blob in blobs:
         self.nRollouts += blob.nRollouts
         self.nUpdates  += blob.nUpdates

         self.lifetime += blob.lifetime
         self.reward   += blob.reward
         self.value    += blob.value

      return self

#Agent logger
class Blob:
   def __init__(self, entID, annID, lifetime, exploration): 
      self.exploration = exploration
      self.lifetime    = lifetime

      self.entID = entID 
      self.annID = annID

#Static blob analytics
# Used in Quill
#   Inkwell.step() is called when Quill.step() is called
#   Inkwell handles the data of quill
class InkWell:
   def __init__(self):#, middleman=None):
      self.util = defaultdict(lambda: defaultdict(Stat))
      self.stat = defaultdict(lambda: defaultdict(Stat))
      # self.middleman = middleman                                              
      # self.xaxis = 'Training Epochs'
      # self.x = 0

   def summary(self):
      return

   def step(self, utilization, statistics):
      '''Calls utilization and statistics'''
      self.utilization(utilization)
      self.statistics(statistics)

   def statistics(self, logs):
      for rollouts, updates, nPkt in logs['Pantheon_Updates']:
         performance = self.stat['Performance']
         performance['Epochs'].update(1, add=True)
         performance['Rollouts'].update(rollouts, add=True)
         performance['Packets'].update(nPkt)
         performance['Updates'].update(updates, add=True)

         t = 'Time'
         if t not in performance:
            performance[t] = TimeQueue()
         performance[t].update(updates)


      # TODO
      # Rewards are passed around in InkWell in 
      # the form of Lifetime (statistics 
      # function). Values would be added there as well
      # likely. You can see that generally, I'm aggregating
      # summary statistics there. There is also a commented 
      # out .append(blob) line -- we can inject whatever
      # data we like into blobs for more detailed visualizations
      # as needed
      for blobs, nEnt in logs['Realm_Logs']:
         self.stat['Agent']['Population'].update(nEnt)
         for blob in blobs:
            #self.stat['Blobs'].append(blob)
            self.stat['Agent']['Lifetime'].update(blob.lifetime)
         #    data = {'lifetime': self.stat['Agent']['Lifetime'].val,
         #            self.xaxis: self.x}
         #    self.x += 1
         #    if self.middleman: self.middleman.setData.remote(data)

            for tile, count in blob.exploration.items():
               self.stat['Agent'][tile].update(count)
            

   def utilization(self, logs):
      for k, vList in logs.items():
         for v in vList:
            self.util[k]['run'].update(v.run)
            self.util[k]['wait'].update(v.wait)

   def summary(self):
      summary = defaultdict(dict)
      for log, vDict in self.stat.items():
         for k, stat in vDict.items():
            if log not in self.stat:
               continue
            summary[log][k] = stat
     
      for log, vDict in self.util.items():
         for k, stat in vDict.items():
            if log not in self.util:
               continue
            summary[log][k] = stat
      return summary
            
   def unique(blobs):
      tiles = defaultdict(list)
      for blob in blobs:
          for t, v in blob.unique.items():
             tiles['unique_'+t.tex].append(v)
      return tiles

   def counts(blobs):
      tiles = defaultdict(list)
      for blob in blobs:
          for t, v in blob.counts.items():
             tiles['counts_'+t.tex].append(v)
      return tiles

   def explore(blobs):
      tiles = defaultdict(list)
      for blob in blobs:
          for t in blob.counts.keys():
             counts = blob.counts[t]
             unique = blob.unique[t]
             if counts != 0:
                tiles['explore_'+t.tex].append(unique / counts)
      return tiles

   def lifetime(blobs):
      return {'lifetime':[blob.lifetime for blob in blobs]}
 
   def reward(blobs):
      return {'reward':[blob.reward for blob in blobs]}
  
   def value(blobs):
      return {'value': [blob.value for blob in blobs]}

@ray.remote
# Quill is an Ascend class, which means it's a remote instance with send/recv functions. It's a data hub, aggregates logs from all other workers
class Quill(Ascend):
   def __init__(self, config, idx):

      super().__init__(config, 0)
      self.config     = config
      self.middleman = None
      if self.config.LOG:
         self.middleman   = visualizer.Middleman.remote()
         self.logger = Logger(self.middleman)
         self.vis    = visualizer.BokehServer.remote(self.middleman, config)
         self.vis.update.remote()
      elif self.config.LOAD_EXP:
         self.middleman  = visualizer.Middleman.remote()
         self.logger     = Logger(self.middleman)
         self.vis        = visualizer.BokehServer.remote(self.middleman, config)
         self.vis.update.remote()


      self.inkwell = InkWell()
      
      self.stats      = defaultdict(Stat)
      self.epochs     = 0
      self.rollouts   = 0
      self.updates    = 0


   def init(self, trinity):
      self.trinity = trinity
      return 'Quill', 'Initialized'

   def step(self):
      utilization, statistics = {}, {}

      #Utilization
      for key in 'Pantheon God Sword'.split():
         utilization[key] = self.recv(key + '_Utilization')

      #Statistics
      for key in 'Pantheon_Updates God_Logs Realm_Logs'.split():
         statistics[key] = self.recv(key)
 
      time.sleep(0.1)
      self.inkwell.step(utilization, statistics)

      def mean(lst):
          return sum(lst)/max(len(lst),1)
      lifetimes = []
      for blobs, _ in statistics['Realm_Logs']:
          for blob in blobs:
              lifetimes.append(blob.lifetime)
      if self.config.LOG:
          self.logger.update(mean(lifetimes), 0, 0, np.std(np.array(lifetimes)), 0, 0)
          

      return self.inkwell.summary()

#Log wrapper and benchmarker
# Not in use
class Benchmarker:
   def __init__(self, logdir):
      self.benchmarks = {}

   def wrap(self, func):
      self.benchmarks[func] = Utils.BenchmarkTimer()
      def wrapped(*args):
         self.benchmarks[func].startRecord()
         ret = func(*args)
         self.benchmarks[func].stopRecord()
         return ret
      return wrapped

   def bench(self, tick):
      if tick % 100 == 0:
         for k, benchmark in self.benchmarks.items():
            bench = benchmark.benchmark()
            print(k.__func__.__name__, 'Tick: ', tick,
                  ', Benchmark: ', bench, ', FPS: ', 1/bench)
 

