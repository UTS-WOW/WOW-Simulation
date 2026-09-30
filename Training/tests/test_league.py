"""The procedural stages really do vary the battlefield, so the policy cannot memorise one map."""

from naval_rl.config import DEFAULT_STAGES, RANDOM_MODES, Config
from naval_rl.league import League


def _plans(stage: int, n: int = 300):
    cfg = Config(start_stage=stage)
    league = League(cfg, ".", seed=3)
    return [league.sample(worker=i % 8).reset for i in range(n)]


def test_full_game_stage_draws_every_battlefield():
    resets = _plans(3)
    assert {r["preset"] for r in resets} == {0, 1, 2}              # archipelago, open sea, strait
    assert {r["density"] for r in resets} == {0, 1, 2}
    assert len({r["weather"] for r in resets}) == 4
    assert {r["mode"] for r in resets} == {0}                       # stage 3 stays domination
    assert len({r["seed"] for r in resets}) > 8                      # a fresh map every few battles


def test_open_stage_draws_modes_cap_sizes_and_fleet_sizes():
    resets = _plans(4)
    assert {r["mode"] for r in resets} == set(RANDOM_MODES)
    radii = [r["capture_radius"] for r in resets]
    lo, hi = DEFAULT_STAGES[4]["procedural"]["capture_radius"]
    assert lo <= min(radii) and max(radii) <= hi and max(radii) - min(radii) > (hi - lo) * 0.5
    sizes = {r["player_ships"] for r in resets}
    assert min(sizes) == 3 and max(sizes) == 8


def test_fixed_values_and_lists_are_respected():
    cfg = Config(stages=[{"name": "x", "procedural": {"mode": [1, 2], "preset": 1, "density": [0],
                                                     "weather": 3, "ships": [4, 4]}, "difficulty": 2}])
    league = League(cfg, ".", seed=0)
    resets = [league.sample(0).reset for _ in range(50)]
    assert {r["mode"] for r in resets} == {1, 2}
    assert {r["preset"] for r in resets} == {1}
    assert {r["density"] for r in resets} == {0}
    assert {r["weather"] for r in resets} == {3}
    assert "capture_radius" not in resets[0]
