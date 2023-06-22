from ordered_set import OrderedSet

class IdAllocator:
  def __init__(self, max_id):
    # Key 0 is reserved as padding
    self.max_id = 1
    self.free = OrderedSet()
    self.expand(max_id)

  def full(self):
    return len(self.free) == 0

  def remove(self, row_id):
    self.free.add(row_id)

  def allocate(self):
    return self.free.pop(0)

  def expand(self, max_id):
    self.free.update(range(self.max_id, max_id))
    self.max_id = max_id
