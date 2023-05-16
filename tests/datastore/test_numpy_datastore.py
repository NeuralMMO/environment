import unittest

import numpy as np

from nmmo.datastore.numpy_datastore import NumpyTable

# pylint: disable=protected-access
class TestNumpyTable(unittest.TestCase):
  def test_continous_table(self):
    table = NumpyTable(3, 10, np.float32)
    table.update(2, 0, 2.1)
    table.update(2, 1, 2.2)
    table.update(5, 0, 5.1)
    table.update(5, 2, 5.3)
    np.testing.assert_array_equal(
      table.get([1,2,5]),
      np.array([[0, 0, 0], [2.1, 2.2, 0], [5.1, 0, 5.3]], dtype=np.float32)
    )

  def test_discrete_table(self):
    table = NumpyTable(3, 10, np.int32)
    table.update(2, 0, 11)
    table.update(2, 1, 12)
    table.update(5, 0, 51)
    table.update(5, 2, 53)
    np.testing.assert_array_equal(
      table.get([1,2,5]),
      np.array([[0, 0, 0], [11, 12, 0], [51, 0, 53]], dtype=np.int32)
    )

  def test_expand(self):
    table = NumpyTable(3, 10, np.float32)

    table.update(2, 0, 2.1)
    with self.assertRaises(IndexError):
      table.update(10, 0, 10.1)

    table._expand(11)
    table.update(10, 0, 10.1)

    np.testing.assert_array_equal(
      table.get([10, 2]),
      np.array([[10.1, 0, 0], [2.1, 0, 0]], dtype=np.float32)
    )

if __name__ == '__main__':
  unittest.main()
