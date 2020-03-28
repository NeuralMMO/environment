from pdb import set_trace as T
from forge.blade.io.stimulus.node import ClassIterable

class StimHook(metaclass=ClassIterable):
   def __init__(self, meta, config, inputs=True):
      self.meta = meta
      self.config = config
      self.nodes = []
                                                                              
      if inputs:
         self.inputs(meta, config)

      self.leaves()

   def leaves(self):
      stack = [([], self)]
      while len(stack) > 0:
         name, node = stack.pop()
         if not isinstance(node, StimHook):
            name = '-'.join(name)
            self.nodes.append((name, node))
            continue
         for lve in node.meta:
            lveName = lve.__name__
            tag     = name + [lveName]
            lveName = lveName[0].lower() + lveName[1:]

            lve  = node.__dict__[lveName]
            stack.append((tag, lve))
                                                                              
   def inputs(self, cls, config):                                             
      for c in cls:                                                     
         self.__dict__[c.name] = c(config)                                    
                                                                              
   def outputs(self, config):                                                 
      data = {}                                                               
      for name, cls in self.nodes:
         name       = name.lower()
         attr       = self.__dict__[cls.name]
         data[name] = attr.packet()

      return data

   def packet(self):                                                          
      return self.outputs(self.config)
