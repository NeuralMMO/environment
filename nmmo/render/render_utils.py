import numpy as np
from scipy import signal

from nmmo.lib.colors import Neon

# NOTE: added to fix json.dumps() cannot serialize numpy objects
# pylint: disable=inconsistent-return-statements
def np_encoder(obj):
  if isinstance(obj, np.generic):
    return obj.item()

def normalize(ary: np.ndarray, norm_std=2):
  R, C         = ary.shape
  preprocessed = np.zeros_like(ary)
  nonzero      = ary[ary!= 0]
  mean         = np.mean(nonzero)
  std          = np.std(nonzero)
  if std == 0:
    std = 1
  for r in range(R):
    for c in range(C):
      val = ary[r, c]
      if val != 0:
        val = (val - mean) / (norm_std * std)
        val = np.clip(val+1, 0, 2)/2
        preprocessed[r, c] = val
  return preprocessed

def clip(ary: np.ndarray):
  R, C         = ary.shape
  preprocessed = np.zeros_like(ary)
  nonzero      = ary[ary!= 0]
  mmin         = np.min(nonzero)
  mmag         = np.max(nonzero) - mmin
  for r in range(R):
    for c in range(C):
      val = ary[r, c]
      val = (val - mmin) / mmag
      preprocessed[r, c] = val
  return preprocessed

def make_two_tone(ary, norm_std=2, preprocess='norm', invert=False, periods=1):
  if preprocess == 'norm':
    ary   = normalize(ary, norm_std)
  elif preprocess == 'clip':
    ary   = clip(ary)

  # if preprocess not in ['norm', 'clip'], assume no preprocessing
  R, C      = ary.shape

  colorized = np.zeros((R, C, 3))
  if periods != 1:
    ary = np.abs(signal.sawtooth(periods*3.14159*ary))
  if invert:
    colorized[:, :, 0] = ary
    colorized[:, :, 1] = 1-ary
  else:
    colorized[:, :, 0] = 1-ary
    colorized[:, :, 1] = ary

  colorized *= (ary != 0)[:, :, None]

  return colorized

# TODO: this is a hack to make the client work
#   by adding color, population, self to the packet
#   integrating with team helper could make this neat
def patch_packet(packet, realm):
  for ent_id in packet['player']:
    packet['player'][ent_id]['base']['color'] = Neon.GREEN.packet()
    # EntityAttr: population was changed to npc_type
    packet['player'][ent_id]['base']['population'] = 0
    # old code: nmmo.Serialized.Entity.Self, no longer being used
    packet['player'][ent_id]['base']['self'] = 1

  npc_colors = {
    1: Neon.YELLOW.packet(), # passive npcs
    2: Neon.MAGENTA.packet(), # neutral npcs
    3: Neon.BLOOD.packet() } # aggressive npcs
  for ent_id in packet['npc']:
    npc = realm.npcs.corporeal[ent_id]
    packet['npc'][ent_id]['base']['color'] = npc_colors[int(npc.npc_type.val)]
    packet['npc'][ent_id]['base']['population'] = -int(npc.npc_type.val) # note negative
    packet['npc'][ent_id]['base']['self'] = 1

  return packet
