"""Curriculum and opponent league.

Stages run from a 1v1 battleship duel up to the full game. At each stage the learner fights the
rule-based fleet AI at the stage's difficulty (Recruit / Veteran / Elite) and is promoted once its
rolling win rate clears the stage threshold. From `selfplay_from_stage` on, episodes are drawn from
a mix:

    latest    both fleets run the current policy (and both sides' experience trains it)
    snapshot  a frozen past policy, chosen with prioritised fictitious self-play: opponents the
              learner still loses to come up more often
    rule      the scripted AI, kept in the mix as an anchor so the policy never forgets how to beat
              it and cannot drift into a strategy cycle that only works against itself

The learner's side is randomised every episode. The observation frames are team-rotated, so both
sides look the same to the network, but maps and spawns are not perfectly symmetric.
"""

from __future__ import annotations

import copy
import json
import os
from collections import deque
from dataclasses import dataclass

import numpy as np

RULE, LATEST, SNAPSHOT = "rule", "latest", "snapshot"


@dataclass
class EpisodePlan:
    stage: int
    learner_team: int
    opponent: str                 # rule | latest | snapshot
    snapshot_id: int = -1
    difficulty: int = 2
    reset: dict | None = None

    def controller(self, team: int) -> str:
        """Who acts for a team: "learner", "latest", "snapshot" or "rule"."""
        if team == self.learner_team:
            return "learner"
        return self.opponent

    def trains_on(self, team: int) -> bool:
        return team == self.learner_team or (self.opponent == LATEST)

    @property
    def learned_teams(self) -> int:
        bits = 1 << self.learner_team
        if self.opponent != RULE:
            bits |= 1 << (1 - self.learner_team)
        return bits


class League:
    def __init__(self, cfg, root: str, seed: int = 0):
        self.cfg = cfg
        self.root = root
        self.rng = np.random.default_rng(seed)
        self.stage = int(cfg.start_stage)
        self.window = {i: deque(maxlen=cfg.winrate_window) for i in range(len(cfg.stages))}
        self.snapshots: list[dict] = []           # {"id", "path", "update", "games", "score"}
        self._scenarios: dict[str, dict] = {}
        self._world: dict[int, tuple[int, int]] = {}   # worker -> (terrain seed, episodes on it)
        self._next_snapshot = 0

    # ------------------------------------------------------------------ sampling

    def sample(self, worker: int) -> EpisodePlan:
        cfg = self.cfg
        st = cfg.stages[self.stage]
        learner = int(self.rng.integers(2))
        opponent, snap, difficulty = RULE, -1, int(st.get("difficulty", 2))

        if self.stage >= cfg.selfplay_from_stage:
            roll = self.rng.random()
            total = cfg.mix_latest + cfg.mix_snapshot + cfg.mix_rule
            if roll < cfg.mix_latest / total:
                opponent = LATEST
            elif roll < (cfg.mix_latest + cfg.mix_snapshot) / total and self.snapshots:
                opponent, snap = SNAPSHOT, self._pfsp()
            else:
                opponent = RULE
                # mostly the stage difficulty, sometimes easier tiers so they stay beaten
                difficulty = int(self.rng.choice([0, 1, 2], p=[0.15, 0.25, 0.6]))

        plan = EpisodePlan(self.stage, learner, opponent, snap, difficulty)
        plan.reset = self._reset_message(st, worker, plan)
        return plan

    def _pfsp(self) -> int:
        """Prioritised fictitious self-play: weight (1 - learner win rate)^2 against each snapshot."""
        w = []
        for s in self.snapshots:
            rate = (s["score"] + 1.0) / (s["games"] + 2.0)       # prior of one win, one loss
            w.append((1.0 - rate) ** 2 + 0.02)
        w = np.array(w) / np.sum(w)
        return self.snapshots[int(self.rng.choice(len(self.snapshots), p=w))]["id"]

    def _reset_message(self, st: dict, worker: int, plan: EpisodePlan) -> dict:
        msg = {
            "learned_teams": plan.learned_teams,
            "opponent_difficulty": plan.difficulty,
            "episode_seed": int(self.rng.integers(1, 2**31 - 1)),
        }
        if "scenario" in st:
            sc = copy.deepcopy(self._scenario(st["scenario"]))
            j = float(self.cfg.jitter)
            for ship in sc.get("ships", []):
                ship["x"] = float(ship["x"]) + float(self.rng.uniform(-j, j))
                ship["y"] = float(ship["y"]) + float(self.rng.uniform(-j, j))
                ship["heading"] = float(ship.get("heading", 0.0)) + float(self.rng.uniform(-15, 15))
            sc["aiDifficulty"] = plan.difficulty
            msg.update({"use_scenario": True, "scenario": sc})
            if st.get("time_limit"):
                msg["time_limit"] = float(st["time_limit"])
            return msg

        p = dict(st["procedural"])
        seed, used = self._world.get(worker, (0, 10**9))
        if used >= int(p.get("world_reuse", 10)):
            seed, used = int(self.rng.integers(1, 999_999)), 0
        self._world[worker] = (seed, used + 1)

        def pick(v, choices):
            return int(self.rng.choice(choices)) if v == "random" else int(v)

        ships = p.get("ships", [6, 6])
        n = int(self.rng.integers(int(ships[0]), int(ships[1]) + 1))
        n = min(n, self.cfg.max_team)
        msg.update({
            "use_scenario": False,
            "mode": int(p.get("mode", 0)),
            "preset": pick(p.get("preset", 0), [0, 1, 2]),
            "density": pick(p.get("density", 1), [0, 1, 2]),
            "weather": pick(p.get("weather", 0), [0, 1, 2, 3]),
            "player_ships": n,
            "enemy_ships": n,
            "seed": seed,
            "time_limit": float(p.get("time_limit", 0)),
        })
        if "capture_radius" in p:
            msg["capture_radius"] = float(p["capture_radius"])
        return msg

    def _scenario(self, path: str) -> dict:
        if path not in self._scenarios:
            full = path if os.path.isabs(path) else os.path.join(self.root, path)
            with open(full) as f:
                self._scenarios[path] = json.load(f)
        return self._scenarios[path]

    # ------------------------------------------------------------------ results

    def record(self, plan: EpisodePlan, learner_score: float) -> None:
        """learner_score: 1 win, 0 loss, 0.5 draw."""
        if plan.opponent == RULE and plan.difficulty >= int(self.cfg.stages[plan.stage].get("difficulty", 2)):
            self.window[plan.stage].append(learner_score)
        elif plan.opponent == SNAPSHOT:
            for s in self.snapshots:
                if s["id"] == plan.snapshot_id:
                    s["games"] += 1
                    s["score"] += learner_score

    def winrate(self, stage: int | None = None) -> float:
        w = self.window[self.stage if stage is None else stage]
        return float(np.mean(w)) if w else float("nan")

    def maybe_promote(self) -> bool:
        st = self.cfg.stages[self.stage]
        w = self.window[self.stage]
        if self.stage + 1 >= len(self.cfg.stages):
            return False
        if len(w) >= max(int(st.get("min_episodes", 50)), 10) and np.mean(w) >= float(st.get("promote_winrate", 0.7)):
            self.stage += 1
            return True
        return False

    def add_snapshot(self, path: str, update: int) -> int:
        sid = self._next_snapshot
        self._next_snapshot += 1
        self.snapshots.append({"id": sid, "path": path, "update": update, "games": 0, "score": 0.0})
        if len(self.snapshots) > self.cfg.pool_size:
            # keep the very first snapshot as a fixed reference point, drop the next oldest
            self.snapshots.pop(1)
        return sid

    def state_dict(self) -> dict:
        return {"stage": self.stage, "window": {k: list(v) for k, v in self.window.items()},
                "snapshots": self.snapshots, "next_snapshot": self._next_snapshot}

    def load_state_dict(self, d: dict) -> None:
        self.stage = d["stage"]
        for k, v in d["window"].items():
            self.window[int(k)] = deque(v, maxlen=self.cfg.winrate_window)
        self.snapshots = d["snapshots"]
        self._next_snapshot = d["next_snapshot"]
