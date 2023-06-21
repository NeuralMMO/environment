from typing import List

import numpy as np

from nmmo.datastore.datastore import Datastore, DataTable


class NumpyTable(DataTable):
  def __init__(self, num_columns: int, initial_size: int, dtype=np.int16):
    super().__init__(num_columns)
    self._dtype  = dtype
    self._initial_size = initial_size
    self._max_rows = 0
    self._data = np.zeros((0, self._num_columns), dtype=self._dtype)
    self._expand(self._initial_size)

  def reset(self):
    super().reset() # resetting _id_allocator
    self._max_rows = 0
    self._data = np.zeros((0, self._num_columns), dtype=self._dtype)
    self._expand(self._initial_size)

  def update(self, row_id: int, col: int, value):
    self._data[row_id, col] = value

  def get(self, ids: List[int]):
    return self._data[ids]

  def where_eq(self, col: int, value):
    return self._data[self._data[:,col] == value]

  def where_neq(self, col: int, value):
    return self._data[self._data[:,col] != value]

  def where_in(self, col: int, values: List):
    return self._data[np.isin(self._data[:,col], values)]

  def window(self, row_idx: int, col_idx: int, row: int, col: int, radius: int):
    return self._data[(
      (np.abs(self._data[:,row_idx] - row) <= radius) &
      (np.abs(self._data[:,col_idx] - col) <= radius)
    ).ravel()]

  def add_row(self) -> int:
    if self._id_allocator.full():
      self._expand(self._max_rows * 2)
    row_id = self._id_allocator.allocate()
    return row_id

  def remove_row(self, row_id: int) -> int:
    self._id_allocator.remove(row_id)
    self._data[row_id] = 0

  def _expand(self, max_rows: int):
    assert max_rows > self._max_rows
    data = np.zeros((max_rows, self._num_columns), dtype=self._dtype)
    data[:self._max_rows] = self._data
    self._max_rows = max_rows
    self._id_allocator.expand(max_rows)
    self._data = data

  def is_empty(self) -> bool:
    all_data_zero = np.sum(self._data)==0
    # 0th row is reserved as padding, so # of free ids is _max_rows-1
    all_id_free = len(self._id_allocator.free) == self._max_rows-1
    return all_data_zero and all_id_free

class NumpyDatastore(Datastore):
  def _create_table(self, num_columns: int) -> DataTable:
    return NumpyTable(num_columns, 100)
