import argparse
import nmmo.lib.task as task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--num_teams", type=int, default=10)
    parser.add_argument("--team_size", type=int, default=1)
    parser.add_argument("--min_clauses", type=int, default=1)
    parser.add_argument("--max_clauses", type=int, default=1)
    parser.add_argument("--min_clause_size", type=int, default=1)
    parser.add_argument("--max_clause_size", type=int, default=1)
    parser.add_argument("--not_p", type=float, default=0.5)
    
    flags = parser.parse_args()

    team_helper = task.TeamHelper(range(flags.team_size * flags.num_teams), flags.num_teams)
    sampler = task.TaskSampler.create_default_task_sampler(team_helper, 0)
    for i in range(flags.tasks):
      task = sampler.sample(
        flags.min_clauses, flags.max_clauses, 
        flags.min_clause_size, flags.max_clause_size, flags.not_p)
      print(task.to_string())



