# pylint: disable=protected-access, no-member
import unittest
import nmmo


class TestDeathFog(unittest.TestCase):
  def test_death_fog(self):
    config = nmmo.config.Default()
    config.set("PLAYER_DEATH_FOG", 3)
    config.set("PLAYER_DEATH_FOG_SPEED", 1/2)
    config.set("PLAYER_DEATH_FOG_FINAL_SIZE", 16)
    config.set("PROVIDE_DEATH_FOG_OBS", True)

    env = nmmo.Env(config)
    env.reset()

    # check the initial fog map
    border = config.MAP_BORDER
    other_border = config.MAP_SIZE - config.MAP_BORDER - 1
    center = config.MAP_SIZE // 2
    safe = config.PLAYER_DEATH_FOG_FINAL_SIZE
    self.assertEqual(env.realm.fog_map[border,border], 0)
    self.assertEqual(env.realm.fog_map[other_border,other_border], 0)
    self.assertEqual(env.realm.fog_map[border+1,border+1], -1)

    # Safe area should be marked with the negative map size
    self.assertEqual(env.realm.fog_map[center-safe,center-safe], -config.MAP_SIZE)
    self.assertEqual(env.realm.fog_map[center+safe-1,center+safe-1], -config.MAP_SIZE)

    for _ in range(config.PLAYER_DEATH_FOG):
      env.step({})

    # check the fog map after the death fog onset
    self.assertEqual(env.realm.fog_map[border,border], config.PLAYER_DEATH_FOG_SPEED)
    self.assertEqual(env.realm.fog_map[border+1,border+1], -1 + config.PLAYER_DEATH_FOG_SPEED)

    for _ in range(3):
      env.step({})

    # check the fog map after 3 ticks after the death fog onset
    self.assertEqual(env.realm.fog_map[border,border], config.PLAYER_DEATH_FOG_SPEED*4)
    self.assertEqual(env.realm.fog_map[border+1,border+1], -1 + config.PLAYER_DEATH_FOG_SPEED*4)

if __name__ == '__main__':
  unittest.main()
