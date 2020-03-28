from pdb import set_trace as T
import numpy as np

import inspect

from forge.blade.io.stimulus import node

class Stimulus(node.Flat):
   class Entity(node.Flat):
      #Base data
      class Base(node.Flat):
         class Self(node.Discrete):
            def init(self, config):
               self.default = 0
               self.max = 1

            def get(self, ent, ref):
               val = int(ent is ref)
               return self.asserts(val)

         class Population(node.Discrete):
            def init(self, config):
               self.default = None
               self.max = config.NPOP

         class R(node.Discrete):
            def init(self, config):
               self.min = -config.STIM
               self.max = config.STIM

            def get(self, ent, ref):
               val = self.val - ref.base.r.val
               return self.asserts(val)
    
         class C(node.Discrete):
            def init(self, config):
               self.min = -config.STIM
               self.max = config.STIM

            def get(self, ent, ref):
               val = self.val - ref.base.c.val
               return self.asserts(val)

      #Historical stats
      class History(node.Flat):
         class Damage(node.Continuous):
            def init(self, config):
               self.default = None
               self.scale = 0.01

         class TimeAlive(node.Continuous):
            def init(self, config):
               self.default = 0
               self.scale = 0.01

      #Resources
      class Resources(node.Flat):
         class Food(node.Continuous):
            def init(self, config):
               self.default = config.RESOURCE
               self.max     = config.RESOURCE

         class Water(node.Continuous):
            def init(self, config):
               self.default = config.RESOURCE
               self.max     = config.RESOURCE

         class Health(node.Continuous):
            def init(self, config):
               self.default = config.HEALTH 
               self.max     = config.HEALTH

      #Status effects
      class Status(node.Flat):
         class Freeze(node.Continuous):
            def init(self, config):
               self.default = 0
               self.max     = 3

         class Immune(node.Continuous):
            def init(self, config):
               self.default = config.IMMUNE
               self.max     = config.IMMUNE

         class Wilderness(node.Continuous):
            def init(self, config):
               self.default = -1
               self.min     = -1
               self.max     = 126

   class Tile(node.Flat):
      class NEnts(node.Continuous):
         def init(self, config):
            self.max = config.NENT

         def get(self, tile, r, c):
            return len(tile.ents)

      class Index(node.Discrete):
         def init(self, config):
            self.max = config.NTILE

         def get(self, tile, r, c):
            return tile.state.index

      class RRel(node.Discrete):
         def init(self, config):
            self.max = config.WINDOW

         def get(self, tile, r, c):
            return r
 
      class CRel(node.Discrete):
         def init(self, config):
            self.max = config.WINDOW

         def get(self, tile, r, c):
            return c

