import unittest

import numpy as np

from nmmo.datastore.numpy_datastore import NumpyDatastore


class TestDatastore(unittest.TestCase):

  def testdatastore_record(self):
    datastore = NumpyDatastore()
    datastore.register_object_type("TestObject", 2)
    c1 = 0
    c2 = 1

    o = datastore.create_record("TestObject")
    self.assertEqual([o.get(c1), o.get(c2)], [0, 0])

    o.update(c1, 1)
    o.update(c2, 2)
    self.assertEqual([o.get(c1), o.get(c2)], [1, 2])

    np.testing.assert_array_equal(
      datastore.table("TestObject").get([o.id]),
      np.array([[1, 2]]))

    o2 = datastore.create_record("TestObject")
    o2.update(c2, 2)
    np.testing.assert_array_equal(
      datastore.table("TestObject").get([o.id, o2.id]),
      np.array([[1, 2], [0, 2]]))

    np.testing.assert_array_equal(
      datastore.table("TestObject").where_eq(c2, 2),
      np.array([[1, 2], [0, 2]]))

    o.delete()
    np.testing.assert_array_equal(
      datastore.table("TestObject").where_eq(c2, 2),
      np.array([[0, 2]]))

if __name__ == '__main__':
  unittest.main()
