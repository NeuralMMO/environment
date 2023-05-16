import unittest

from nmmo.datastore.id_allocator import IdAllocator

class TestIdAllocator(unittest.TestCase):
  def test_id_allocator(self):
    id_allocator = IdAllocator(10)

    for i in range(1, 10):
      row_id = id_allocator.allocate()
      self.assertEqual(i, row_id)
    self.assertTrue(id_allocator.full())

    id_allocator.remove(5)
    id_allocator.remove(6)
    id_allocator.remove(1)
    self.assertFalse(id_allocator.full())

    self.assertSetEqual(
      set(id_allocator.allocate() for i in range(3)),
      set([5, 6, 1])
    )
    self.assertTrue(id_allocator.full())

    id_allocator.expand(11)
    self.assertFalse(id_allocator.full())

    self.assertEqual(id_allocator.allocate(), 10)

    with self.assertRaises(KeyError):
      id_allocator.allocate()

  def test_id_reuse(self):
    id_allocator = IdAllocator(10)

    for i in range(1, 10):
      row_id = id_allocator.allocate()
      self.assertEqual(i, row_id)
    self.assertTrue(id_allocator.full())

    id_allocator.remove(5)
    id_allocator.remove(6)
    id_allocator.remove(1)
    self.assertFalse(id_allocator.full())

    self.assertSetEqual(
      set(id_allocator.allocate() for i in range(3)),
      set([5, 6, 1])
    )
    self.assertTrue(id_allocator.full())

    id_allocator.expand(11)
    self.assertFalse(id_allocator.full())

    self.assertEqual(id_allocator.allocate(), 10)

    with self.assertRaises(KeyError):
      id_allocator.allocate()

    id_allocator.remove(10)
    self.assertEqual(id_allocator.allocate(), 10)

if __name__ == '__main__':
  unittest.main()
