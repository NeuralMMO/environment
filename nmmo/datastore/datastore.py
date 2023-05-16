from __future__ import annotations
from typing import Dict, List
from nmmo.datastore.id_allocator import IdAllocator

"""
This code defines a data storage system that allows for the
creation, manipulation, and querying of records.

The DataTable class serves as the foundation for the data
storage, providing methods for updating and retrieving data,
as well as filtering and querying records.

The DatastoreRecord class represents a single record within
a table and provides a simple interface for interacting with
the data. The Datastore class serves as the main entry point
for the data storage system, allowing for the creation and
management of tables and records.

The implementation of the DataTable class is left to the
developer, but the DatastoreRecord and Datastore classes
should be sufficient for most use cases.

See numpy_datastore.py for an implementation.
"""
class DataTable:
  def __init__(self, num_columns: int):
    self._num_columns = num_columns
    self._id_allocator = IdAllocator(100)

  def reset(self):
    self._id_allocator = IdAllocator(100)

  def update(self, row_id: int, col: int, value):
    raise NotImplementedError

  def get(self, ids: List[id]):
    raise NotImplementedError

  def where_in(self, col: int, values: List):
    raise NotImplementedError

  def where_eq(self, col: str, value):
    raise NotImplementedError

  def where_neq(self, col: str, value):
    raise NotImplementedError

  def window(self, row_idx: int, col_idx: int, row: int, col: int, radius: int):
    raise NotImplementedError

  def remove_row(self, row_id: int):
    raise NotImplementedError

  def add_row(self) -> int:
    raise NotImplementedError

  def is_empty(self) -> bool:
    raise NotImplementedError

class DatastoreRecord:
  def __init__(self, datastore, table: DataTable, row_id: int) -> None:
    self.datastore = datastore
    self.table = table
    self.id = row_id

  def update(self, col: int, value):
    self.table.update(self.id, col, value)

  def get(self, col: int):
    return self.table.get(self.id)[col]

  def delete(self):
    self.table.remove_row(self.id)

class Datastore:
  def __init__(self) -> None:
    self._tables: Dict[str, DataTable] = {}

  def register_object_type(self, object_type: str, num_colums: int):
    if object_type not in self._tables:
      self._tables[object_type] = self._create_table(num_colums)

  def create_record(self, object_type: str) -> DatastoreRecord:
    table = self._tables[object_type]
    row_id = table.add_row()
    return DatastoreRecord(self, table, row_id)

  def table(self, object_type: str) -> DataTable:
    return self._tables[object_type]

  def _create_table(self, num_columns: int) -> DataTable:
    raise NotImplementedError
