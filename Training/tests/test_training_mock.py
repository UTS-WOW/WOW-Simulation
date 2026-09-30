"""End-to-end: the trainer, league and mock environment together learn the mock task.

The mock pays for shooting the contact whose first feature is largest (a pointer-head decision) and
for steering the compass leg its self features hint at (a fixed-head decision). Within the test's
budget the pointer decision is learned to high accuracy; the compass mapping (eight separate
associations starting at 1-in-16 odds) takes longer, so the test only asks that it improves.
"""

import numpy as np
import torch

from naval_rl.config import Config
from naval_rl.env import init_message, make_workers
from naval_rl.league import League
from naval_rl.mappo import MAPPO


def _rollout_accuracy(algo):
    b = algo._buf
    m = (b["active"][..., None] * b["alive"]) > 0
    hint = np.argmax(b["self"][..., :8], -1)
    move = (b["actions"][..., 0] == 1 + hint)[m].mean()
    best = 1 + np.argmax(b["contacts"][..., 0], -1)
    target = (b["actions"][..., 2] == best)[m].mean()
    return move, target


def test_mappo_learns_the_mock_task(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config(workers=4, mock=True, max_team=3, max_allies=2, max_contacts=3, max_zones=3,
                 rollout=64, chunk=16, epochs=4, minibatches=2, d_model=64, hidden=64, layers=1,
                 lr_actor=1e-3, lr_critic=1e-3, total_updates=60, ent_coef=0.003, ent_coef_final=0.001,
                 gamma=0.9, lam=0.9,
                 stages=[{"name": "mock", "procedural": {"ships": [3, 3]}, "difficulty": 2, "promote_winrate": 2}],
                 selfplay_from_stage=0, mix_latest=0.5, mix_snapshot=0.0, mix_rule=0.5)
    init = init_message(3, 2, 3, 3, "intent", 1.0, 0.02, True)
    workers = make_workers(4, init, None, 0, mock=True)
    algo = MAPPO(cfg, workers[0].spec, workers, League(cfg, str(tmp_path)), torch.device("cpu"))
    algo.reset_all()

    algo.collect()
    move0, target0 = _rollout_accuracy(algo)
    for _ in range(cfg.total_updates):
        stats = algo.learn()
        assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
        algo.collect()
    move1, target1 = _rollout_accuracy(algo)

    assert target0 < 0.4 and target1 > 0.75, f"target accuracy {target0:.2f} -> {target1:.2f}"
    assert move1 > 1.6 * move0, f"move accuracy {move0:.2f} -> {move1:.2f}"
    # the league saw both self-play and rule-AI episodes, and recorded results for the latter
    assert len(algo.league.window[0]) > 0
