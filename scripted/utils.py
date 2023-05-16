

def l1(start, goal):
   sr, sc = start
   gr, gc = goal
   return abs(gr - sr) + abs(gc - sc)

def l2(start, goal):
   sr, sc = start
   gr, gc = goal
   return 0.5*((gr - sr)**2 + (gc - sc)**2)**0.5

def lInfty(start, goal):
   sr, sc = start
   gr, gc = goal
   return max(abs(gr - sr), abs(gc - sc))

def adjacentPos(pos):
   r, c = pos
   return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]

def adjacentDeltas():
   return [(-1, 0), (1, 0), (0, 1), (0, -1)]

def inSight(dr, dc, vision):
    return (
          dr >= -vision and
          dc >= -vision and
          dr <= vision and
          dc <= vision)