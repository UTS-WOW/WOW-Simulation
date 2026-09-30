"""Reward weights and shaping schedule.

The environment reports raw components (see RLRewardTracker.cs); the weights live here so reward
design can change without rebuilding the game.

Per agent slot and step:
    r = sum_k team_w[k] * team_component[k] * s_k  +  shaping * sum_k agent_w[k] * agent_component[k]

Team terms are shared by every ship on the side and are zero-sum between the sides (damage dealt
by one side is damage taken by the other, the score margin is antisymmetric), which is what makes
self-play well posed. The terminal win/loss and the score margin are the actual objective and are
never annealed; everything else is shaping and decays towards `floor` so the final policy
optimises for winning rather than for the proxies.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TEAM_WEIGHTS = {
    "score_delta": 2.0,           # (my - their score) / 1000 per step: zone income + kill points
    "damage_dealt": 1.0,          # fraction of the enemy fleet's HP removed
    "damage_taken": -1.0,
    "zones_captured": 0.1,
    "zones_lost": -0.1,
    "kills": 0.3,                 # fraction of the enemy fleet sunk
    "losses": -0.3,
    "friendly_fire_taken": -0.5,
    "win": 1.0,
    "loss": -1.0,
    "draw": 0.0,
}

DEFAULT_AGENT_WEIGHTS = {
    "damage_dealt": 0.2,          # hull fractions of the victims
    "spotting_damage": 0.2,       # damage team mates did to targets this ship was spotting
    "friendly_fire_dealt": -0.5,
    "damage_taken": 0.0,
    "sunk": -0.1,
}

OBJECTIVE_TERMS = {"score_delta", "win", "loss", "draw"}


class RewardFunction:
    def __init__(self, team_components: list[str], agent_components: list[str],
                 team_weights: dict | None = None, agent_weights: dict | None = None,
                 anneal_updates: int = 1500, floor: float = 0.1, spotting: bool = True):
        tw = dict(DEFAULT_TEAM_WEIGHTS, **(team_weights or {}))
        aw = dict(DEFAULT_AGENT_WEIGHTS, **(agent_weights or {}))
        if not spotting:
            aw["spotting_damage"] = 0.0
        self.team_w = np.array([tw.get(k, 0.0) for k in team_components], dtype=np.float32)
        self.team_objective = np.array([k in OBJECTIVE_TERMS for k in team_components], dtype=bool)
        self.agent_w = np.array([aw.get(k, 0.0) for k in agent_components], dtype=np.float32)
        self.anneal_updates = max(1, anneal_updates)
        self.floor = floor

    def shaping(self, update: int) -> float:
        frac = min(1.0, update / self.anneal_updates)
        return 1.0 - (1.0 - self.floor) * frac

    def __call__(self, team_reward: np.ndarray, agent_reward: np.ndarray, update: int) -> np.ndarray:
        """team_reward [..., K_team], agent_reward [..., N, K_agent] -> reward per slot [..., N]."""
        s = self.shaping(update)
        scale = np.where(self.team_objective, 1.0, s).astype(np.float32)
        team = (team_reward * self.team_w * scale).sum(-1)
        agent = (agent_reward * self.agent_w).sum(-1) * s
        return team[..., None] + agent
