import unittest

import nmmo
from nmmo.lib.team_helper import TeamHelper

class TestTeamSpawn(unittest.TestCase):
  def test_team_spawn(self):
    config = nmmo.config.Small()
    num_teams = 16
    team_size = 8

    team_helper = TeamHelper({
      i: [i*team_size+j+1 for j in range(team_size)]
      for i in range(num_teams)}
    )

    env = nmmo.Env(config, team_helper=team_helper)
    env.reset()

    # agents in the same team should spawn together
    team_locs = {}
    for team_id, team_members in team_helper.teams.items():
      team_locs[team_id] = env.realm.players[team_members[0]].pos
      for agent_id in team_members:
        self.assertEqual(team_locs[team_id], env.realm.players[agent_id].pos)

    # teams should be apart from each other
    for i in range(num_teams):
      for j in range(i+1, num_teams):
        self.assertNotEqual(team_locs[i], team_locs[j])


if __name__ == '__main__':
  unittest.main()
