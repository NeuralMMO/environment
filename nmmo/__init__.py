import logging

from .version import __version__

from .lib import material, spawn
from .render.overlay import Overlay, OverlayRegistry
from .core import config, agent, action
from .core.action import Action
from .core.agent import Agent, Scripted
from .core.env import Env
from .core.terrain import MapGenerator, Terrain

MOTD = rf'''      ___           ___           ___           ___
     /__/\         /__/\         /__/\         /  /\      Version {__version__:<8}
     \  \:\       |  |::\       |  |::\       /  /::\
      \  \:\      |  |:|:\      |  |:|:\     /  /:/\:\    An open source
  _____\__\:\   __|__|:|\:\   __|__|:|\:\   /  /:/  \:\   project originally
 /__/::::::::\ /__/::::| \:\ /__/::::| \:\ /__/:/ \__\:\  founded by Joseph Suarez
 \  \:\~~\~~\/ \  \:\~~\__\/ \  \:\~~\__\/ \  \:\ /  /:/  and formalized at OpenAI
  \  \:\  ~~~   \  \:\        \  \:\        \  \:\  /:/
   \  \:\        \  \:\        \  \:\        \  \:\/:/    Now developed and
    \  \:\        \  \:\        \  \:\        \  \::/     maintained at MIT in
     \__\/         \__\/         \__\/         \__\/      Phillip Isola's lab '''

__all__ = ['Env', 'config', 'agent', 'Agent', 'Scripted', 'MapGenerator', 'Terrain',
        'action', 'Action', 'material', 'spawn',
        'Overlay', 'OverlayRegistry']

try:
  __all__.append('OpenSkillRating')
except RuntimeError:
  logging.error('Warning: OpenSkill not installed. Ignore if you do not need this feature')
