"""The league: copies of our own fleet - and of the other method's - to practise against, and Elo ratings.

Against one fixed opponent a policy learns to beat THAT opponent - and against only its latest self
it can go round in circles (rock beats scissors beats paper beats rock...). The fix used by OpenAI
Five, AlphaGo Zero and AlphaStar is a league of opponents (the shares are set per curriculum stage):

    mirror    the current fleet on both sides, both learning (env.py)
    latest    a frozen copy of the current fleet, refreshed every few updates (OpenAI Five: 80 %)
    past      older copies kept in a pool (OpenAI Five: 20 %), picked by prioritised fictitious
              self-play (PFSP, AlphaStar): an opponent we lose to comes up more often,

                  weight = (1 - our win rate against it) ** 2            ("hard", the default)
                  weight = win rate * (1 - win rate)                     ("variance": the ones we
                                                                          beat about half the time)

    champion  the strongest version of the fleet so far. It is only replaced when the learner has
              beaten it - at least champion_win_rate of the last champion_games battles against it
              (AlphaGo Zero's gating: 55 %). The learner then becomes the new champion, and the old
              one stays in the pool as a past copy. So the fleet always has its best self to beat.
    rival     the OTHER method's fleet (MAPPO's for the Double DQN run, and the other way round) -
              a differently trained opponent with different habits, the way AlphaStar's league mixes
              different kinds of agents so no single strategy dominates (cross-play)
    rule      the game's rule-based AI - the anchor every stage is judged against

Every player has an Elo rating (chess style): after each battle the winner takes points from the
loser, more for an upset. The rule AI's three difficulties are players too, so the ratings show how
the fleet compares with Recruit, Veteran and Elite over time.

The pool keeps only the acting networks (FleetPolicy weights): that is all an opponent needs. A
copy marked greedy (a Double DQN fleet) takes its best action; the others sample (MAPPO).
"""

from __future__ import annotations

import copy
import os

import numpy as np

RULE_NAMES = ["rule-recruit", "rule-veteran", "rule-elite"]


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo: the chance that a player rated rating_a beats one rated rating_b."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _weights(policy) -> dict:
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


class OpponentPool:
    def __init__(self, size: int = 20, seed: int = 0, elo_k: float = 16.0, pfsp: str = "hard",
                 champion_games: int = 40, champion_win_rate: float = 0.55):
        self.size, self.elo_k, self.pfsp = size, elo_k, pfsp
        self.champion_games, self.champion_win_rate = champion_games, champion_win_rate
        self.rng = np.random.default_rng(seed)
        self.snapshots: list[dict] = []      # {"id", "kind", "name", "update", "weights", "greedy", "games", "wins"}
        self.next_id = 0
        self.ratings = {"learner": 1000.0, **{name: 1000.0 for name in RULE_NAMES}}
        self.champion: int | None = None     # the snapshot id of the champion
        self.vs_champion: list[float] = []   # the learner's results against the current champion

    # ------------------------------------------------------------------ the pool

    def _new(self, kind: str, name: str, weights: dict, update: int, greedy: bool) -> int:
        sid = self.next_id
        self.next_id += 1
        self.snapshots.append({"id": sid, "kind": kind, "name": name, "update": update, "weights": weights,
                               "greedy": greedy, "games": 0, "wins": 0.0})
        self.ratings.setdefault(name, self.ratings["learner"])
        return sid

    def add(self, policy, update: int, greedy: bool = False) -> int:
        """Keeps a frozen copy of the policy's weights as a past copy; the oldest past copy goes when the
        pool is full (the very first one stays, as a fixed reference point)."""
        sid = self._new("past", f"past-{self.next_id}", _weights(policy), update, greedy)
        past = [s for s in self.snapshots if s["kind"] == "past"]
        if len(past) > self.size:
            gone = past[1]
            self.snapshots.remove(gone)
            self.ratings.pop(gone["name"], None)
        return sid

    def crown(self, policy, update: int, greedy: bool = False) -> int:
        """The learner becomes the champion; the previous champion stays on as a past copy."""
        for s in self.snapshots:
            if s["id"] == self.champion:
                s["kind"] = "past"
        sid = self._new("champion", f"champion-{self.next_id}", _weights(policy), update, greedy)
        self.ratings[f"champion-{sid}"] = self.ratings["learner"]
        self.champion, self.vs_champion = sid, []
        return sid

    def champion_beaten(self) -> bool:
        """The learner beat the champion often enough to take its place (AlphaGo Zero's gating)."""
        return (len(self.vs_champion) >= self.champion_games
                and float(np.mean(self.vs_champion[-self.champion_games:])) >= self.champion_win_rate)

    def set_rival(self, weights: dict, name: str, update: int, greedy: bool) -> int:
        """The other method's fleet (cross-play). Replaces the previous rival; its rating is kept."""
        self.snapshots = [s for s in self.snapshots if s["kind"] != "rival"]
        return self._new("rival", f"rival-{name}", weights, update, greedy)

    def sample(self, kind: str = "past") -> int | None:
        """An opponent of this kind: the champion, the rival, or a past copy picked by PFSP (the ones we
        still lose to most likely). None when there is none yet."""
        if kind == "champion":
            return self.champion
        pool = [s for s in self.snapshots if s["kind"] == kind or (kind == "past" and s["kind"] == "champion")]
        if not pool:
            return None
        win_rate = np.array([(s["wins"] + 1.0) / (s["games"] + 2.0) for s in pool])     # prior: 1 win, 1 loss
        weight = (win_rate * (1.0 - win_rate) if self.pfsp == "variance" else (1.0 - win_rate) ** 2) + 0.02
        return int(pool[int(self.rng.choice(len(weight), p=weight / weight.sum()))]["id"])

    def weights(self, sid: int) -> dict | None:
        s = self._get(sid)
        return s["weights"] if s else None

    def greedy(self, sid: int) -> bool:
        s = self._get(sid)
        return bool(s and s.get("greedy", False))

    def _get(self, sid: int) -> dict | None:
        for s in self.snapshots:
            if s["id"] == sid:
                return s
        return None

    # ------------------------------------------------------------------ results

    def record(self, info: dict, difficulty: int | None = None) -> None:
        """One finished training battle of the learner (an env info dict)."""
        opponent, won = info.get("opponent"), float(info["won"])
        if opponent in ("past", "champion", "rival"):
            s = self._get(info.get("opponent_id"))
            if s is None:
                return
            s["games"] += 1
            s["wins"] += won
            if s["id"] == self.champion:
                self.vs_champion.append(won)
            self._elo("learner", s["name"], won)
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
        games = {s["name"]: f"{int(s['wins'])}/{s['games']} won" + (" (champion)" if s["id"] == self.champion else "")
                 for s in self.snapshots}
        live = set(games) | {"learner", *RULE_NAMES}
        return sorted(((k, round(v), games.get(k, "")) for k, v in self.ratings.items() if k in live),
                      key=lambda r: -r[1])

    # ------------------------------------------------------------------ saving

    def state_dict(self) -> dict:
        return {"snapshots": copy.deepcopy(self.snapshots), "next_id": self.next_id, "ratings": dict(self.ratings),
                "champion": self.champion, "vs_champion": list(self.vs_champion)}

    def load_state_dict(self, d: dict) -> None:
        self.snapshots, self.next_id, self.ratings = d["snapshots"], d["next_id"], d["ratings"]
        for s in self.snapshots:                       # pools saved before champions and rivals existed
            s.setdefault("kind", "past")
            s.setdefault("name", f"past-{s['id']}")
            s.setdefault("greedy", False)
        self.champion, self.vs_champion = d.get("champion"), list(d.get("vs_champion", []))


def fleet_snapshot(source, champion: bool = True):
    """A fleet to fight, from a MAPPO / Double DQN model or a checkpoint path: (weights, method name,
    timesteps, greedy). Its champion if it has one (and champion=True), otherwise its current network."""
    if isinstance(source, str):
        import torch
        ck = torch.load(source, map_location="cpu", weights_only=False)
        greedy = ck.get("method") == "fleet-ddqn"
        league = ck.get("league") or {}
        best = league.get("champion") if champion else None
        weights = next((s["weights"] for s in league.get("snapshots", []) if s["id"] == best), None)
        return weights or ck["policy"], "Double DQN" if greedy else "MAPPO", ck.get("num_timesteps", 0), greedy
    greedy = hasattr(source, "value_head")
    best = source.league.weights(source.league.champion) if champion and source.league.champion is not None else None
    return best or _weights(source.policy), "Double DQN" if greedy else "MAPPO", source.num_timesteps, greedy


class LeagueMixin:
    """What MAPPO and FleetDDQN share about the league: the enemy fleet's actions in league battles,
    past copies, the champion, and the other method's fleet. GREEDY: this method's own copies take
    their best action (Double DQN) instead of sampling (MAPPO)."""

    GREEDY = False
    OPPONENT_EPSILON = 0.02

    def _opponent_act(self, obs: dict, envs: list[int], ids: list[int]) -> np.ndarray:
        """The enemy fleet's actions in league battles: the frozen latest copy (id -1) or a snapshot."""
        E = self.env.n_envs
        if self._opp_memory is None or self._opp_memory.h.shape[0] != E:
            self._opp_memory = self.policy.new_memory(E, self.env.n_agents)
        tens = self._tensors(obs)
        envs, ids = np.asarray(envs), np.asarray(ids)
        actions = np.zeros((len(envs), self.env.n_agents, len(self.head_sizes)), dtype=np.int32)
        for sid in np.unique(ids):
            rows = np.flatnonzero(ids == sid)
            policy = self.opponent if sid < 0 else self._past_policy(int(sid))
            greedy = self.GREEDY if sid < 0 or self.league.weights(int(sid)) is None else self.league.greedy(int(sid))
            eps = self.hp.get("opponent_epsilon", self.OPPONENT_EPSILON) if greedy else None
            # a Double DQN copy with Boltzmann exploration plays the way it trained
            temp = self.hp.get("temperature") if greedy and self.hp.get("exploration") == "softmax" else None
            memory = self._opp_memory.take(envs[rows])
            out = policy.act({k: v[rows] for k, v in tens.items()}, memory, epsilon=eps, commander_epsilon=eps,
                             temperature=temp)
            self._opp_memory.put(envs[rows], memory)
            actions[rows] = out["actions"].cpu().numpy()
        return actions

    def _past_policy(self, sid: int):
        if sid not in self._past:
            weights = self.league.weights(sid)
            if weights is None:
                return self.opponent
            policy = copy.deepcopy(self.opponent)
            policy.load_state_dict(weights)
            self._past = {k: v for k, v in self._past.items() if self.league.weights(k) is not None}
            self._past[sid] = policy
        return self._past[sid]

    def _league_step(self) -> None:
        """After an update: a past copy every snapshot_every updates, and a new champion when the
        learner has beaten the old one (or the first champion, when a stage starts using one)."""
        mix = self.env.stages[self.env.stage].get("opponents") or {}
        if "past" in mix and self.n_updates % self.hp["snapshot_every"] == 0:
            self.league.add(self.policy, self.n_updates, greedy=self.GREEDY)
        if "champion" in mix and (self.league.champion is None or self.league.champion_beaten()):
            beaten = self.league.champion is not None
            rate = float(np.mean(self.league.vs_champion[-self.league.champion_games:])) if beaten else 0.0
            sid = self.league.crown(self.policy, self.n_updates, greedy=self.GREEDY)
            if self.verbose:
                print(f"*** new champion (champion-{sid}) at {self.num_timesteps:,} timesteps"
                      + (f": the fleet beat the old champion in {rate:.0%} of its last "
                         f"{self.league.champion_games} battles against it ***" if beaten else " - the first one ***"))

    def set_rival(self, other, champion: bool = True) -> bool:
        """Cross-play: the other method's fleet (a model, or a checkpoint path) joins this run's league as
        the "rival" - its champion if it has one (champion=False: its current network). Call it again
        to refresh it (e.g. before each learn)."""
        if other is None or (isinstance(other, str) and not os.path.exists(other)):
            return False
        weights, name, steps, greedy = fleet_snapshot(other, champion)
        own = self.policy.state_dict()
        if set(weights) != set(own) or any(tuple(weights[k].shape) != tuple(own[k].shape) for k in own):
            print(f"the {name} fleet has a different network (settings differ) - not added as a rival")
            return False
        self.league.set_rival(weights, name, steps, greedy)
        if self.verbose:
            print(f"rival: the {name} fleet ({steps:,} timesteps) joins the league for cross-play")
        return True
