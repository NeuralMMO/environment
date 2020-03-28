from pdb import set_trace as T
import numpy as np
import time

from collections import defaultdict

from forge.blade.io.stimulus.static import Stimulus as Static
from forge.blade.io.serial import Serial

def camel(string):
   '''Convert a string to camel case'''
   return string[0].lower() + string[1:]

class Stimulus:
   '''Static IO class used for interacting with game observations

   The environment returns game objects in observations.
   This class assembles them into usable data packets'''
   def process(config, inp, env, ent, serialize=True):
      '''Utility for preprocessing game observations

      Built to be semi-automatic and only require small updates
      for large new classes of observations

      Args:
         config    : An environment configuration object
         inp       : An IO object specifying observations
         env       : Local environment observation
         ent       : Local entity observation
         serialize : (bool) Whether to serialize the IO object data
      '''

      #Static handles
      Stimulus.funcNames = 'Entity Tile'.split()
      Stimulus.functions = [Stimulus.entity, Stimulus.tile]

      for key, f in zip(Stimulus.funcNames, Stimulus.functions):
         #t = time.time()
         f(inp, env, ent, key, serialize)
         #print('{}: {:.4f}'.format(key, time.time() - t))

   def add(inp, obs, lookup, obj, *args, key, serialize=False):
      '''Pull attributes from game and serialize names'''
      for name, attr in obj.nodes:
         val = attr.get(*args)
         obs.attributes[name].append(val)

      #Serialize names
      lookupKey = obj
      if serialize:
         lookupKey = (key.serial, obj.serial)

      idx = lookup.add(lookupKey, orig=obj)
      inp.obs.names[key].append(idx)

   def tile(inp, env, ent, key, serialize=False):
      '''Internal processor for tile objects'''
      for r, row in enumerate(env):
         for c, tile in enumerate(row):
            Stimulus.add(inp, inp.obs.entities[key], inp.lookup,
               tile, tile, r, c, key=ent, serialize=serialize)

   def entity(inp, env, ent, key, serialize=False):
      '''Internal processor for player objects. Always returns self first'''
      ents = []
      for tile in env.ravel():
         for e in tile.ents.values():
            ents.append(e)
     
      ents = sorted(ents, key=lambda e: e is ent, reverse=True)

      for e in ents:
         Stimulus.add(inp, inp.obs.entities[key], inp.lookup,
            e, ent, e, key=ent, serialize=serialize)

      '''
      ents = []
      static = Stimulus.static[key]
      for tile in env.ravel():
         for e in tile.ents.values():
            Stimulus.add(inp, inp.obs.entities[key], inp.lookup,
               static, e, ent, e, key=ent, serialize=serialize)
      '''



