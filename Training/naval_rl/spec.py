"""The environment's self-description (the reply to "init"). Nothing about sizes is hard-coded
on the Python side; everything comes from here."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Head:
    name: str
    fixed: int          # options the network scores with a plain linear layer
    pointer: str        # "", "zones" or "contacts": extra options scored against entity tokens
    size: int           # total options on the wire (fixed + pointer slots)


@dataclass
class Spec:
    max_team: int
    max_allies: int
    max_contacts: int
    max_zones: int
    action_mode: str
    decision_period: float
    sim_dt: float
    dims: dict
    heads: list[Head]
    enemy_privileged: tuple[int, int]
    features: dict = field(default_factory=dict)
    team_reward_components: list[str] = field(default_factory=list)
    agent_reward_components: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, j: dict) -> "Spec":
        return cls(
            max_team=j["max_team"],
            max_allies=j["max_allies"],
            max_contacts=j["max_contacts"],
            max_zones=j["max_zones"],
            action_mode=j["action_mode"],
            decision_period=j["decision_period"],
            sim_dt=j["sim_dt"],
            dims=j["dims"],
            heads=[Head(**h) for h in j["heads"]],
            enemy_privileged=tuple(j["enemy_privileged"]),
            features=j.get("features", {}),
            team_reward_components=j.get("team_reward_components", []),
            agent_reward_components=j.get("agent_reward_components", []),
        )

    def to_json(self) -> dict:
        return {
            "max_team": self.max_team, "max_allies": self.max_allies, "max_contacts": self.max_contacts,
            "max_zones": self.max_zones, "action_mode": self.action_mode, "decision_period": self.decision_period,
            "sim_dt": self.sim_dt, "dims": self.dims,
            "heads": [h.__dict__ for h in self.heads], "enemy_privileged": list(self.enemy_privileged),
            "features": self.features, "team_reward_components": self.team_reward_components,
            "agent_reward_components": self.agent_reward_components,
        }

    @property
    def total_logits(self) -> int:
        return sum(h.size for h in self.heads)

    @property
    def head_sizes(self) -> list[int]:
        return [h.size for h in self.heads]

    def with_caps(self, max_team=None, max_allies=None, max_contacts=None, max_zones=None) -> "Spec":
        """The same network on a bigger battle: only the padding caps (and pointer head sizes) change."""
        s = Spec.from_json(self.to_json())
        s.max_team = max_team or s.max_team
        s.max_allies = max_allies if max_allies is not None else s.max_allies
        s.max_contacts = max_contacts or s.max_contacts
        s.max_zones = max_zones or s.max_zones
        for h in s.heads:
            if h.pointer == "zones":
                h.size = h.fixed + s.max_zones
            elif h.pointer == "contacts":
                h.size = h.fixed + s.max_contacts
        return s
