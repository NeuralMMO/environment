cimport numpy as np
import numpy as np
import math

cpdef double[:,:] compute_values(tiles, entity):
   cdef double[:,:] reward_matrix = map_to_rewards(tiles, entity)
   cdef int nb_lines = len(reward_matrix[0]), nb_columns = len(reward_matrix[1])

   cdef double gamma_factor = 0.8  # look ahead ∈ [0, 1]
   cdef double max_delta = 0.01  # maximum allowed approximation

   cdef double[:,:] value_matrix = np.zeros((nb_lines, nb_columns), dtype=np.float64)
   cdef double[:,:] old_value_matrix

   cdef double delta = float('inf')

   cdef int line, column
   cdef double max_value, current_value
   while delta > max_delta:
      old_value_matrix = np.array(value_matrix, copy=True)
      for line in range(nb_lines):
         for column in range(nb_columns):
            max_value = float('-inf')

            if line - 1 >= 0:
               max_value = value_matrix[line - 1][column]

            if line + 1 < nb_lines:
               current_value = value_matrix[line + 1][column]
               if current_value > max_value:
                  max_value = current_value

            if column - 1 >= 0:
               current_value = value_matrix[line][column - 1]
               if current_value > max_value:
                  max_value = current_value

            if column + 1 < nb_columns:
               current_value = value_matrix[line][column + 1]
               if current_value > max_value:
                  max_value = current_value

            value_matrix[line][
               column] = reward_matrix[line][column] + gamma_factor * max_value

      delta = np.nanmax(
         np.abs(np.subtract(old_value_matrix, value_matrix)))
   return value_matrix

cpdef double[:,:] map_to_rewards(tiles, entity):
   cdef double lava_reward, stone_reward, water_reward, enemy_reward, forest_reward, scrub_reward, around_water_reward
   cdef int line, column
   cdef str tile_val

   lava_reward = stone_reward = water_reward = enemy_reward = float('-inf')
   forest_reward = 1.0 + math.pow(
      (1 - entity.resources.food.val / entity.resources.food.max) * 15.0,
      1.25)
   scrub_reward = 1.0
   around_water_reward = 1.0 + math.pow(
      (1 - entity.resources.water.val / entity.resources.water.max) * 15.0,
      1.25)

   cdef double[:,:] reward_matrix = np.full((len(tiles), len(tiles[0])), 0.0, dtype=np.float64)

   for line in range(len(tiles)):
      tile_line = tiles[line]
      for column in range(len(tile_line)):
         tile_val = tile_line[column].state.tex
         if tile_val == 'lava':
            reward_matrix[line][column] += lava_reward

         if tile_val == 'stone':
            reward_matrix[line][column] += stone_reward

         if tile_val == 'forest':
            reward_matrix[line][column] += forest_reward

         if tile_val == 'water':
            reward_matrix[line][column] += water_reward


         if has_water_around(tiles, line, column):
            reward_matrix[line][column] += around_water_reward

         if tile_val == 'scrub':
            reward_matrix[line][column] += scrub_reward

         if len(tile_line[column].ents.values()) > 0:
            reward_matrix[line][column] += enemy_reward

   return reward_matrix

cpdef tuple max_value_direction_around(int line, int column, double[:,:] value_matrix):
   cdef double max_value = float('-inf')
   cdef int horizontal_dir, vertical_dir

   if line - 1 >= 0 and value_matrix[line - 1][column] > max_value :
      max_value = value_matrix[line - 1][column]
      horizontal_dir = -1
      vertical_dir = 0

   if line + 1 < len(value_matrix) and value_matrix[line + 1][column] > max_value :
      max_value = value_matrix[line + 1][column]
      horizontal_dir = 1
      vertical_dir = 0


   if column - 1 >= 0 and value_matrix[line][column-1] > max_value :
      max_value = value_matrix[line][column-1]
      horizontal_dir = 0
      vertical_dir = -1

   if column + 1<len(value_matrix[0]) and value_matrix[line][column+1] > max_value :
      horizontal_dir = 0
      vertical_dir = 1

   return horizontal_dir, vertical_dir


cdef has_water_around(tiles, int line, int column):
   return (inBounds(line-1, column, tiles.shape) and tiles[line-1, column].state.tex is 'water') or \
          (inBounds(line, column-1, tiles.shape) and tiles[line, column-1].state.tex is 'water') or \
          (inBounds(line+1, column, tiles.shape) and tiles[line+1, column].state.tex is 'water') or \
          (inBounds(line, column+1, tiles.shape) and tiles[line, column+1].state.tex is 'water')

cdef inBounds(int r, int c, tuple shape):
   cdef int R, C
   R, C = shape
   return (
         r > 0 and
         c > 0 and
         r < R and
         c < C
         )
