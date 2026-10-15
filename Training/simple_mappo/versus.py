"""MAPPO against Double DQN: the two fleets fight each other, and (optionally) train against each other.

The stage notebooks compare the methods side by side - each against the rule AI. This is the other
test: put one method's fleet on each side of the same battle. Used by Versus-MAPPO-vs-DDQN.ipynb:

    stages  = versus_stages(4)                          # King of the Hill: [head to head, training]
    battles = NavalEnv(stages=stages, n_envs=8)
    env_m, env_d = Monitor(battles, ".../mappo"), Monitor(battles, ".../ddqn")
    mappo, dqn = MAPPO.load(..., env=env_m), FleetDDQN.load(..., env=env_d)

    head_to_head(mappo, dqn, env_m, n_battles=40)      # MAPPO's fleet (blue) vs Double DQN's (red)
    head_to_head(dqn, mappo, env_d, n_battles=40)      # the other way round: the same map from the other side

The enemy seat is the "rival" of league.py: blue.set_rival(red) puts red's fleet into blue's league,
and the stage's opponent mix sends every battle's enemy to it. Each fleet
plays the way it plays in the game: MAPPO samples its options, Double DQN takes its best one.
"""

from __future__ import annotations

import math

import numpy as np

from .curriculum import DEFAULT_CURRICULUM

FIGHT, TRAIN = 0, 1                     # the two stages versus_stages() returns


def versus_stages(stage: int, rival_share: float = 0.7) -> list[dict]:
    """Two stages made from curriculum stage `stage` (its battle, fleets, commander and team spirit):

        0  head to head: every enemy fleet is the rival (the other method's fleet)
        1  training each other: the rival in rival_share of the battles, the stage's rule AI in the
           rest - the anchor that keeps both honest (two fleets that only ever meet each other can
           learn tricks that only work on each other)

    No promotion and no rehearsal: the battle stays the same all the way."""
    base = dict(DEFAULT_CURRICULUM[stage])
    for key in ("promote_win_rate", "window", "stop_win_rate", "stop_window", "eval_battles", "eval_every"):
        base.pop(key, None)
    base.update(opponent="league", rehearsal=0.0)
    if base.get("pass_if") in ("sunk", "captured"):     # a passive-target stage: fight for the win instead
        base.pop("pass_if"), base.pop("within", None)
    fight = dict(base, name=f"{base['name']} - head to head", opponents={"rival": 1.0})
    train = dict(base, name=f"{base['name']} - training each other",
                 opponents={"rival": rival_share, "rule": round(1.0 - rival_share, 6)} if rival_share < 1 else {"rival": 1.0})
    return [fight, train]


def head_to_head(blue, red, env, n_battles: int = 40, deterministic: bool = False) -> dict:
    """blue's fleet (our side) against red's fleet (the enemy side), on env's head-to-head stage.

    Returns blue's results: battles, win_rate (+ its standard error), enemy ships sunk and own ships
    lost per battle, points for and against. Red's results are the mirror image."""
    if not blue.set_rival(red, champion=False):          # red's fleet as it is now
        raise ValueError("the two fleets have different networks - they cannot share a battle")
    previous = env.stage
    env.set_stage(FIGHT)
    env.set_opponent_policy(blue._opponent_act, blue.league.sample)    # red flies the enemy
    blue._play_memory = None
    try:
        obs = env.reset()
        E = env.n_envs
        quota = math.ceil(n_battles / E)
        done_per_env = np.zeros(E, dtype=int)
        results = []
        while done_per_env.min() < quota:
            actions = blue.predict(obs, deterministic)
            obs, _, dones, infos = env.step(actions, blue.last_orders)
            for e in np.flatnonzero(dones[:E]):
                if done_per_env[e] < quota and infos[e].get("opponent") == "rival":
                    results.append(infos[e])
                    done_per_env[e] += 1
    finally:
        env.set_stage(previous)
    won = np.array([r["won"] for r in results], dtype=float)
    stats = lambda k: float(np.mean([r.get("stats", {}).get(k, np.nan) for r in results]))
    return {"battles": len(results), "win_rate": float(won.mean()),
            "win_rate_stderr": float(won.std(ddof=1) / np.sqrt(len(won))) if len(won) > 1 else 0.0,
            "enemy_ships_sunk": stats("player_kills"), "own_ships_lost": stats("enemy_kills"),
            "points": stats("player_score"), "enemy_points": stats("enemy_score"),
            "mean_battle_time": float(np.mean([r["battle_time"] for r in results]))}


def both_sides(mappo, dqn, env_mappo, env_dqn, n_battles: int = 40, deterministic: bool = False) -> dict:
    """The head-to-head from both seats (each fleet once as blue), and MAPPO's overall win rate."""
    a = head_to_head(mappo, dqn, env_mappo, n_battles, deterministic)
    b = head_to_head(dqn, mappo, env_dqn, n_battles, deterministic)
    n = a["battles"] + b["battles"]
    mappo_wins = a["win_rate"] * a["battles"] + (1.0 - b["win_rate"]) * b["battles"]
    return {"MAPPO as blue": a, "Double DQN as blue": b,
            "mappo_win_rate": mappo_wins / max(1, n), "battles": n}


def elo_gap(win_rate: float) -> float:
    """The Elo difference that predicts this win rate (400 points = 10:1 odds; capped at +-1200)."""
    p = min(max(win_rate, 1e-3), 1 - 1e-3)
    return 400.0 * math.log10(p / (1.0 - p))
