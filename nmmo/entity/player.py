import numpy as np
from pdb import set_trace as T

import nmmo
from nmmo.systems import ai, equipment, inventory
from nmmo.lib import material

from nmmo.systems.skill import Skills
from nmmo.systems.achievement import Diary
from nmmo.entity import entity

class Player(entity.Entity):
   def __init__(self, realm, pos, agent, color, pop):
      super().__init__(realm, pos, agent.iden, agent.policy, color, pop)
      self.agent  = agent
      self.pop    = pop

      # Scripted hooks
      self.target = None
      self.food   = None
      self.water  = None
      self.vision = 7

      # Logs
      self.buys                     = 0
      self.sells                    = 0 
      self.ration_consumed          = 0
      self.poultice_consumed        = 0
      self.ration_level_consumed    = 0
      self.poultice_level_consumed = 0

      # Submodules
      self.skills = Skills(realm, self)

      self.diary  = None
      tasks = realm.config.TASKS
      if tasks:
          self.diary = Diary(tasks)

      self.dataframe.init(nmmo.Serialized.Entity, self.entID, self.pos)

   @property
   def serial(self):
      return self.population, self.entID

   @property
   def isPlayer(self) -> bool:
      return True

   @property
   def population(self):
      if __debug__:
          assert self.base.population.val == self.pop
      return self.pop

   def applyDamage(self, dmg, style):
      self.resources.food.increment(dmg)
      self.resources.water.increment(dmg)
      self.skills.applyDamage(dmg, style)
      
   def receiveDamage(self, source, dmg):
      if not super().receiveDamage(source, dmg):
         if source:
            source.history.playerKills += 1
         return 

      self.resources.food.decrement(dmg)
      self.resources.water.decrement(dmg)
      self.skills.receiveDamage(dmg)

   def packet(self):
      data = super().packet()

      data['entID']     = self.entID
      data['annID']     = self.population

      data['base']      = self.base.packet()
      data['resource']  = self.resources.packet()
      data['skills']    = self.skills.packet()
      data['inventory'] = self.inventory.packet()

      return data
  
   def update(self, realm, actions):
      '''Post-action update. Do not include history'''
      super().update(realm, actions)

      if not self.alive:
         return

      self.resources.update(realm, self, actions)
      self.skills.update(realm, self)

      if self.diary:
         self.diary.update(realm, self)
