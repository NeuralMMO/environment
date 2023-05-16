from collections import defaultdict
import unittest

from nmmo.datastore.serialized import SerializedState

# pylint: disable=no-member,unused-argument,unsubscriptable-object

FooState = SerializedState.subclass("FooState", [
  "a", "b", "col"
])

FooState.Limits = {
  "a": (-10, 10),
}

class MockDatastoreRecord():
  def __init__(self):
    self._data = defaultdict(lambda: 0)

  def get(self, name):
    return self._data[name]

  def update(self, name, value):
    self._data[name] = value

class MockDatastore():
  def create_record(self, name):
    return MockDatastoreRecord()

  def register_object_type(self, name, attributes):
    assert name == "FooState"
    assert attributes == ["a", "b", "col"]

class TestSerialized(unittest.TestCase):

  def test_serialized(self):
    state = FooState(MockDatastore(), FooState.Limits)

    # initial value = 0
    self.assertEqual(state.a.val, 0)

    # if given value is within the range, set to the value
    state.a.update(1)
    self.assertEqual(state.a.val, 1)

    # if given a lower value than the min, set to min
    a_min, a_max = FooState.Limits["a"]
    state.a.update(a_min - 100)
    self.assertEqual(state.a.val, a_min)

    # if given a higher value than the max, set to max
    state.a.update(a_max + 100)
    self.assertEqual(state.a.val, a_max)

if __name__ == '__main__':
  unittest.main()
