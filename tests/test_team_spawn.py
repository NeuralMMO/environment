import unittest

import nmmo
from nmmo.core.agent import Agent
from nmmo.lib.team_helper import TeamHelper
from nmmo.lib import spawn


class TeamLoader(spawn.SequentialLoader):
  def __init__(self, config, np_random, team_helper: TeamHelper):
    assert config.PLAYERS == [Agent], \
      "TeamLoader only supports config.PLAYERS == [Agent]"
    super().__init__(config, np_random)
    self.team_helper = team_helper

    self.candidate_spawn_pos = \
      spawn.get_team_spawn_positions(config, team_helper.num_teams)

  def get_spawn_position(self, agent_id):
    team_id, _ = self.team_helper.team_and_position_for_agent[agent_id]
    return self.candidate_spawn_pos[team_id]


class TestTeamSpawn(unittest.TestCase):
  def test_team_spawn(self):
    num_teams = 16
    team_size = 8
    team_helper = TeamHelper({
      i: [i*team_size+j+1 for j in range(team_size)]
      for i in range(num_teams)}
    )

    config = nmmo.config.Small()
    config.PLAYER_N = num_teams * team_size
    config.PLAYER_LOADER =\
      lambda config, np_random: TeamLoader(config, np_random, team_helper)

    assert config.PLAYER_N == num_teams * team_size,\
      "config.PLAYER_N must be num_teams * team_size"
    env = nmmo.Env(config)
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
