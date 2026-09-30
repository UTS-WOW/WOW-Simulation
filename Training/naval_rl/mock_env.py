"""A stand-in for the Unity environment with the same interface, shapes and message semantics.

It exists so the trainer can be tested without Unity: every array the real environment sends is
produced here with the right shape and masks, and the reward carries a learnable signal. Each
agent's self features hold a one-hot "hint" in columns 0..7; choosing the matching compass leg on
the move head (option 1 + hint) and shooting the contact whose first feature is largest pays off.
A trainer that is wired correctly learns both within a few thousand agent-steps.
"""

from __future__ import annotations

import numpy as np

from .env import Obs
from .spec import Spec


def mock_spec(init: dict) -> Spec:
    Z, C = init["max_zones"], init["max_contacts"]
    intent = init.get("action_mode", "intent") == "intent"
    move_fixed = 13 if intent else 5
    heads = [
        {"name": "move", "fixed": move_fixed, "pointer": "zones" if intent else "", "size": move_fixed + (Z if intent else 0)},
        {"name": "speed", "fixed": 5, "pointer": "", "size": 5},
        {"name": "target", "fixed": 1, "pointer": "contacts", "size": 1 + C},
        {"name": "fire", "fixed": 2, "pointer": "", "size": 2},
        {"name": "torpedo", "fixed": 2, "pointer": "", "size": 2},
        {"name": "ability", "fixed": 15, "pointer": "", "size": 15},
    ]
    dims = {"self": 16, "ally": 8, "contact": 10, "zone": 6, "critic_own": 17, "critic_enemy": 12,
            "critic_zone": 6, "critic_match": 8}
    team_rc = ["score_delta", "damage_dealt", "damage_taken", "zones_captured", "zones_lost", "kills", "losses",
               "friendly_fire_taken", "win", "loss", "draw"]
    agent_rc = ["damage_dealt", "spotting_damage", "friendly_fire_dealt", "damage_taken", "sunk"]
    return Spec.from_json({
        "max_team": init["max_team"], "max_allies": init["max_allies"], "max_contacts": C, "max_zones": Z,
        "action_mode": init.get("action_mode", "intent"), "decision_period": init.get("decision_period", 1.0),
        "sim_dt": init.get("sim_dt", 0.02), "dims": dims, "heads": heads, "enemy_privileged": [1, 6],
        "features": {"critic_enemy": ["alive", "t1", "t2", "t3", "t4", "t5", "b0", "b1", "b2", "belief_none", "b4", "b5"]},
        "team_reward_components": team_rc, "agent_reward_components": agent_rc,
    })


class MockWorker:
    def __init__(self, init: dict, seed: int = 0, episode_length: int = 40, death_rate: float = 0.01,
                 step_signal: float = 0.5):
        self.spec = mock_spec(init)
        self.rng = np.random.default_rng(seed)
        self.episode_length = episode_length
        self.death_rate = death_rate
        self.step_signal = step_signal
        self.episode = 0
        self._pending = None
        self._t = 0
        self._ships = (0, 0)
        self._learned = [True, False]
        self._alive = None
        self._score = np.zeros(2)
        self.last_config = None

    # ------------------------------------------------------------------ interface

    def reset_async(self, config: dict) -> None:
        self.last_config = config
        lt = int(config.get("learned_teams", 1))
        self._learned = [bool(lt & 1), bool(lt & 2)]
        n = self.spec.max_team
        p = int(np.clip(config.get("player_ships", n), 1, n))
        e = int(np.clip(config.get("enemy_ships", n), 1, n))
        self._ships = (p, e)
        self._alive = np.zeros((2, n), dtype=bool)
        self._alive[0, :p] = True
        self._alive[1, :e] = True
        self._t = 0
        self._score = np.zeros(2)
        self.episode += 1
        self._make_obs()
        self._pending = self._build(terminal=False, team_rew=np.zeros((2, 11)), agent_rew=np.zeros((2, n, 5)))

    def step_async(self, actions: np.ndarray) -> None:
        s = self.spec
        n = s.max_team
        actions = np.asarray(actions).reshape(2, n, len(s.heads))
        team_rew = np.zeros((2, 11), dtype=np.float32)
        agent_rew = np.zeros((2, n, 5), dtype=np.float32)
        for t in range(2):
            alive = self._alive[t]
            if not alive.any():
                continue
            if self._learned[t]:
                correct_move = actions[t, :, 0] == 1 + self._hint[t]
                best_contact = 1 + np.argmax(self._contacts[t][:, :, 0], axis=1)
                correct_target = actions[t, :, 2] == best_contact
            else:  # "rule AI": right half the time
                correct_move = self.rng.random(n) < 0.5
                correct_target = self.rng.random(n) < 0.5
            quality = (correct_move.astype(np.float32) + correct_target.astype(np.float32)) * 0.5
            q = float(quality[alive].mean())
            team_rew[t, 0] = q * self.step_signal
            agent_rew[t, alive, 0] = quality[alive] * self.step_signal
            self._score[t] += q
        self._t += 1
        # random attrition, never the last ship
        for t in range(2):
            idx = np.flatnonzero(self._alive[t])
            if len(idx) > 1 and self.rng.random() < self.death_rate * len(idx):
                self._alive[t, self.rng.choice(idx)] = False
        terminal = self._t >= self.episode_length
        winner = -1
        draw = False
        if terminal:
            if abs(self._score[0] - self._score[1]) < 1e-6:
                draw = True
                team_rew[:, 10] = 1.0
            else:
                winner = int(np.argmax(self._score))
                team_rew[winner, 8] = 1.0
                team_rew[1 - winner, 9] = 1.0
        self._make_obs()
        self._pending = self._build(terminal, team_rew, agent_rew, winner, draw)

    def recv(self) -> Obs:
        obs, self._pending = self._pending, None
        return obs

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ fake observations

    def _make_obs(self):
        s = self.spec
        n, C = s.max_team, s.max_contacts
        self._hint = self.rng.integers(0, 8, size=(2, n))
        self._contacts = self.rng.random((2, n, C, s.dims["contact"]), dtype=np.float32)

    def _build(self, terminal, team_rew, agent_rew, winner=-1, draw=False) -> Obs:
        s = self.spec
        n, A, C, Z = s.max_team, s.max_allies, s.max_contacts, s.max_zones
        d = s.dims
        rng = self.rng
        alive = self._alive.astype(np.float32)
        selfv = rng.normal(size=(2, n, d["self"])).astype(np.float32) * 0.1
        selfv[..., :8] = 0.0
        for t in range(2):
            selfv[t, np.arange(n), self._hint[t]] = 1.0
        mask = np.zeros((2, n, s.total_logits), dtype=np.float32)
        off = 0
        for h in s.heads:
            if h.pointer == "zones":
                mask[..., off:off + h.fixed] = 1.0
                mask[..., off + h.fixed:off + h.fixed + 3] = 1.0      # three zones exist
            elif h.pointer == "contacts":
                mask[..., off:off + h.size] = 1.0
            else:
                mask[..., off:off + h.size] = 1.0
            off += h.size
        ally_mask = np.zeros((2, n, A), dtype=np.float32)
        ally_mask[..., :min(A, 2)] = 1.0
        zone_mask = np.zeros((2, n, Z), dtype=np.float32)
        zone_mask[..., :3] = 1.0
        for t in range(2):
            if not self._learned[t]:
                selfv[t] = 0.0
                mask[t] = 0.0
        arrays = {
            "self": selfv * alive[..., None],
            "allies": rng.normal(size=(2, n, A, d["ally"])).astype(np.float32),
            "ally_mask": ally_mask,
            "contacts": self._contacts,
            "contact_mask": np.ones((2, n, C), dtype=np.float32),
            "zones": rng.normal(size=(2, n, Z, d["zone"])).astype(np.float32),
            "zone_mask": zone_mask,
            "action_mask": mask,
            "alive": alive,
            "critic_own": np.concatenate([rng.normal(size=(2, n, d["critic_own"] - 1)).astype(np.float32),
                                          alive[..., None]], axis=-1),
            "critic_enemy": rng.normal(size=(2, n, d["critic_enemy"])).astype(np.float32),
            "critic_enemy_mask": np.ones((2, n), dtype=np.float32),
            "critic_zones": rng.normal(size=(2, Z, d["critic_zone"])).astype(np.float32),
            "critic_zone_mask": zone_mask[:, 0, :],
            "critic_match": rng.normal(size=(2, d["critic_match"])).astype(np.float32),
            "team_reward": team_rew.astype(np.float32),
            "agent_reward": agent_rew.astype(np.float32),
            "learned": np.array(self._learned, dtype=np.float32),
        }
        stats = {}
        if terminal:
            stats = {"player_score": float(self._score[0]), "enemy_score": float(self._score[1]),
                     "focus_fire": [0.5, 0.5], "spotting_share": [0.1, 0.1]}
        return Obs(arrays=arrays, terminal=terminal, winner=winner, draw=draw, reason="mock",
                   battle_time=float(self._t), episode=self.episode, ships=self._ships, stats=stats)
