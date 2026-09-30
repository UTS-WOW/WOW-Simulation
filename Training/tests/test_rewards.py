import numpy as np

from naval_rl.mock_env import mock_spec
from naval_rl.rewards import RewardFunction


def test_team_reward_is_zero_sum_and_objective_survives_annealing():
    spec = mock_spec({"max_team": 2, "max_allies": 1, "max_contacts": 2, "max_zones": 1})
    names = spec.team_reward_components
    fn = RewardFunction(names, spec.agent_reward_components, anneal_updates=10, floor=0.0)
    comp = {k: i for i, k in enumerate(names)}
    team = np.zeros((2, len(names)), dtype=np.float32)
    # player removes 30% of the enemy fleet, enemy removes 10% of the player's; player sinks one of two
    team[0, comp["damage_dealt"]] = 0.3; team[1, comp["damage_taken"]] = 0.3
    team[1, comp["damage_dealt"]] = 0.1; team[0, comp["damage_taken"]] = 0.1
    team[0, comp["kills"]] = 0.5; team[1, comp["losses"]] = 0.5
    team[0, comp["score_delta"]] = 0.02; team[1, comp["score_delta"]] = -0.02
    agent = np.zeros((2, 2, len(spec.agent_reward_components)), dtype=np.float32)
    r = fn(team, agent, update=0)
    assert abs(r[0, 0] + r[1, 0]) < 1e-6
    # once shaping has fully annealed only the objective (score margin) remains
    r_late = fn(team, agent, update=10)
    assert abs(r_late[0, 0] - 2.0 * 0.02) < 1e-6
    team[:, :] = 0; team[0, comp["win"]] = 1; team[1, comp["loss"]] = 1
    r_end = fn(team, agent, update=10)
    assert r_end[0, 0] == 1.0 and r_end[1, 0] == -1.0
