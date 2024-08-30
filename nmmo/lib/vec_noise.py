import numpy as np

# The noise2() was ported from https://github.com/zbenjamin/vec_noise

# https://github.com/zbenjamin/vec_noise/blob/master/_noise.h#L13
GRAD3 = np.array([
    [1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
    [1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
    [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1],
    [1,0,-1], [-1,0,-1], [0,-1,1], [0,1,1]
])

# https://github.com/zbenjamin/vec_noise/blob/master/_noise.h#L31
PERM = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140,
    36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120,
    234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
    88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71,
    134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
    230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161,
    1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130,
    116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250,
    124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227,
    47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44,
    154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98,
    108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
    242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14,
    239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121,
    50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243,
    141, 128, 195, 78, 66, 215, 61, 156, 180
], dtype=np.int32)
PERM = np.concatenate((PERM, PERM))

# 2D simplex skew factors
F2 = 0.5 * (np.sqrt(3.0) - 1.0)
G2 = (3.0 - np.sqrt(3.0)) / 6.0

# https://github.com/zbenjamin/vec_noise/blob/master/_simplex.c#L46
def snoise2(x, y):
  """Generate 2D simplex noise for given coordinates."""
  s = (x + y) * F2
  i = np.floor(x + s).astype(int)
  j = np.floor(y + s).astype(int)
  t = (i + j) * G2

  x0 = x - (i - t)
  y0 = y - (j - t)

  # Determine which simplex we're in
  i1 = (x0 > y0).astype(int)
  j1 = 1 - i1

  x1 = x0 - i1 + G2
  y1 = y0 - j1 + G2
  x2 = x0 - 1 + 2 * G2
  y2 = y0 - 1 + 2 * G2

  # Hash coordinates of the three simplex corners
  ii = i & 255
  jj = j & 255
  gi0 = PERM[ii + PERM[jj]] % 12
  gi1 = PERM[ii + i1 + PERM[jj + j1]] % 12
  gi2 = PERM[ii + 1 + PERM[jj + 1]] % 12

  # Calculate contribution from three corners
  t0 = 0.5 - x0**2 - y0**2
  t1 = 0.5 - x1**2 - y1**2
  t2 = 0.5 - x2**2 - y2**2

  mask0 = (t0 >= 0).astype(float)
  mask1 = (t1 >= 0).astype(float)
  mask2 = (t2 >= 0).astype(float)

  n0 = mask0 * t0**4 * (GRAD3[gi0, 0] * x0 + GRAD3[gi0, 1] * y0)
  n1 = mask1 * t1**4 * (GRAD3[gi1, 0] * x1 + GRAD3[gi1, 1] * y1)
  n2 = mask2 * t2**4 * (GRAD3[gi2, 0] * x2 + GRAD3[gi2, 1] * y2)

  # Sum up and scale the result
  return 70 * (n0 + n1 + n2)
