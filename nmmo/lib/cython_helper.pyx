#cython: boundscheck=False
#cython: wraparound=True
#cython: nonecheck=False

from types import SimpleNamespace
import numpy as np
cimport numpy as cnp

# for array indexing
cnp.import_array()

def make_move_mask(cnp.ndarray[cnp.int8_t] mask,
                   cnp.ndarray[cnp.int8_t, ndim=2] habitable_tiles,
                   short row, short col,
                   cnp.ndarray[cnp.int64_t] row_delta,
                   cnp.ndarray[cnp.int64_t] col_delta):
  for i in range(4):
    mask[i] = habitable_tiles[row_delta[i] + row, col_delta[i] + col]

# NOTE: assume that incoming mask are all zeros
def make_attack_mask(cnp.ndarray[cnp.int8_t] mask,
                     cnp.ndarray[cnp.int16_t, ndim=2] entities,
                     dict entity_attr,
                     dict my_info):
  cdef short idx
  cdef short num_valid_target = 0
  cdef short attr_id = entity_attr["id"]
  cdef short attr_time_alive = entity_attr["time_alive"]
  cdef short attr_npc_type = entity_attr["npc_type"]
  cdef short attr_row = entity_attr["row"]
  cdef short attr_col = entity_attr["col"]

  for idx in range(len(entities)):
    # skip empty row
    if entities[idx, attr_id] == 0:
      continue
    # out of range
    if abs(entities[idx, attr_row] - my_info["row"]) > my_info["attack_range"] or \
       abs(entities[idx, attr_col] - my_info["col"]) > my_info["attack_range"]:
      continue
    # cannot attack during immunity
    if entities[idx, attr_id] > 0 and \
       entities[idx, attr_time_alive] < my_info["immunity"]:
      continue
    # cannot attack self
    if entities[idx, attr_id] == my_info["agent_id"]:
      continue
    # npc_type must be 0, 1, 2, 3
    if entities[idx, attr_npc_type] > 3:
      continue
    mask[idx] = 1
    num_valid_target += 1

  # cython: wraparound need to be True
  # if any valid target, set the no-op to 0
  mask[-1] = 0 if num_valid_target > 0 else 1

def parse_array(short[:] data, dict attr_name_to_col):
  cdef short col
  cdef str attr
  cdef dict result = {}
  for attr, col in attr_name_to_col.items():
    result[attr] = data[col]
  return SimpleNamespace(**result)
