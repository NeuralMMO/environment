from pdb import set_trace as T
import numpy as np

from itertools import chain 
from collections import defaultdict

class Serial:
   '''Internal serialization class for communication across machines

   Mainly wraps Stimulus.serialize and Action.serialize. Also provides
   keying functionality for converting game objects to unique IDs.

   Format: World, Tick, key.serial[0], key.serial[1], key type'''

   #Length of serial key tuple
   KEYLEN = 4

   def key(key):
      '''Convert a game object to a unique key'''
      if key is None:
         return tuple([-1]*Serial.KEYLEN)

      #Concat object key with class key
      T()
      n = Serial.KEYLEN - len(key.serial) - 1
      ret = [-1]*Serial.KEYLEN
      ret[-1] = key.SERIAL
      ret[n:-1] = key.serial
      return tuple(ret)

   def nontemporal(key):
      '''Get the time independent part of a key'''
      T()
      return tuple(key[:1]) + tuple(key[2:])

   def population(key):
      '''Get the population component of a nontemporal entity key'''
      T()
      return key[1]

