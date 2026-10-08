"""The league: past copies of our own fleet to practise against, and Elo ratings for everyone.

Against one fixed opponent a policy learns to beat THAT opponent - and against only its latest self
it can go round in circles (rock beats scissors beats paper beats rock...). The fix used by OpenAI
Five and AlphaStar is a league of opponents:

    latest  a frozen copy of the current fleet, refreshed every few updates (OpenAI Five: 80 %)
    past    older copies kept in a pool (OpenAI Five: 20 %), picked by prioritised fictitious
            self-play (PFSP, AlphaStar): an opponent we lose to comes up more often,

                weight = (1 - our win rate against it) ** 2

    rule    the game's rule-based AI - the anchor every stage is judged against

Every player has an Elo rating (chess style): after each battle the winner takes points from the
loser, more for an upset. The rule AI's three difficulties are players too, so the ratings show how
the fleet compares with Recruit, Veteran and Elite over time.

The pool keeps only the actor networks (FleetPolicy weights): that is all an opponent needs.
"""

from __future__ import annotations

import copy

import numpy as np

RULE_NAMES = ["rule-recruit", "rule-veteran", "rule-elite"]


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo: the chance that a player rated rating_a beats one rated rating_b."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class OpponentPool:
    def __init__(self, size: int = 20, seed: int = 0, elo_k: float = 16.0):
        self.size, self.elo_k = size, elo_k
        self.rng = np.random.default_rng(seed)
        self.snapshots: list[dict] = []      # {"id", "update", "weights", "games", "wins"}
        self.next_id = 0
        self.ratings = {"learner": 1000.0, **{name: 1000.0 for name in RULE_NAMES}}

    # ------------------------------------------------------------------ the pool

    def add(self, policy, update: int) -> int:
        """Keeps a frozen copy of the policy's weights; the oldest copy goes when the pool is full
        (the very first one stays, as a fixed reference point)."""
        sid = self.next_id
        self.next_id += 1
        weights = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        self.snapshots.append({"id": sid, "update": update, "weights": weights, "games": 0, "wins": 0.0})
        self.ratings[f"past-{sid}"] = self.ratings["learner"]
        if len(self.snapshots) > self.size:
            gone = self.snapshots.pop(1)
            self.ratings.pop(f"past-{gone['id']}", None)
        return sid

    def sample(self) -> int | None:
        """PFSP: a past snapshot, the ones we still lose to most likely. None while the pool is empty."""
        if not self.snapshots:
            return None
        win_rate = np.array([(s["wins"] + 1.0) / (s["games"] + 2.0) for s in self.snapshots])   # prior: 1 win, 1 loss
        weight = (1.0 - win_rate) ** 2 + 0.02
        return int(self.snapshots[int(self.rng.choice(len(weight), p=weight / weight.sum()))]["id"])

    def weights(self, sid: int) -> dict | None:
        for s in self.snapshots:
            if s["id"] == sid:
                return s["weights"]
        return None

    # ------------------------------------------------------------------ results

    def record(self, info: dict, difficulty: int | None = None) -> None:
        """One finished training battle of the learner (an env info dict)."""
        opponent, won = info.get("opponent"), float(info["won"])
        if opponent == "past":
            for s in self.snapshots:
                if s["id"] == info.get("opponent_id"):
                    s["games"] += 1
                    s["wins"] += won
            self._elo("learner", f"past-{info.get('opponent_id')}", won)
        elif opponent == "rule" and difficulty is not None:
            self._elo("learner", RULE_NAMES[int(np.clip(difficulty, 0, 2))], won)

    def _elo(self, a: str, b: str, score_a: float) -> None:
        if b not in self.ratings:
            return
        ea = expected_score(self.ratings[a], self.ratings[b])
        self.ratings[a] += self.elo_k * (score_a - ea)
        self.ratings[b] -= self.elo_k * (score_a - ea)

    def table(self) -> list[tuple[str, float, str]]:
        """(player, Elo, record against it) best first - for the notebook."""
        games = {f"past-{s['id']}": f"{int(s['wins'])}/{s['games']} won" for s in self.snapshots}
        return sorted(((k, round(v), games.get(k, "")) for k, v in self.ratings.items()), key=lambda r: -r[1])

    # ------------------------------------------------------------------ saving

    def state_dict(self) -> dict:
        return {"snapshots": copy.deepcopy(self.snapshots), "next_id": self.next_id, "ratings": dict(self.ratings)}

    def load_state_dict(self, d: dict) -> None:
        self.snapshots, self.next_id, self.ratings = d["snapshots"], d["next_id"], d["ratings"]
