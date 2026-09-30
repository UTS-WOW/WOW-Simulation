"""Training configuration. Defaults are the starting point from the design doc; override with a
YAML/JSON file (--config) and/or --set key=value."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

DEFAULT_STAGES = [
    # 0: gunnery, target choice, angling, fire timing. A duel is dice-heavy: a simple scripted tactic
    #    (auto target, fire at will, broadside) wins 55% against Recruit, so that is the bar
    {"name": "bb_duel", "scenario": "scenarios/stage0_bb_duel.json", "difficulty": 0,
     "promote_winrate": 0.55, "min_episodes": 100},
    # 1: capture while fighting, small mixed fleets, no cover
    {"name": "koth_3v3", "scenario": "scenarios/stage1_koth_3v3.json", "difficulty": 1,
     "promote_winrate": 0.65, "min_episodes": 100},
    # 2: spotting, smoke, radar, islands
    {"name": "archipelago_3v3", "scenario": "scenarios/stage2_archipelago_3v3.json", "difficulty": 1,
     "promote_winrate": 0.65, "min_episodes": 150},
    # 3: the full game at 6v6, on any battlefield: archipelago, open sea or strait, any island density
    #    and weather, a fresh map every few battles - so the policy learns terrain, not one map
    {"name": "domination_6v6", "procedural": {"mode": 0, "preset": "random", "density": "random", "ships": [6, 6],
                                               "weather": "random", "time_limit": 900, "world_reuse": 8},
     "difficulty": 2, "promote_winrate": 0.6, "min_episodes": 200},
    # 4: generalisation: any mode with or without objectives (domination, skirmish, fleet battle,
    #    capture and control), any map, cap size and fleet size
    {"name": "open", "procedural": {"mode": "random", "preset": "random", "density": "random", "ships": [3, 8],
                                     "weather": "random", "capture_radius": [130, 200], "time_limit": 1200,
                                     "world_reuse": 4},
     "difficulty": 2, "promote_winrate": 1.01, "min_episodes": 0},
]

# GameMode values a procedural stage may draw with "mode": "random". Escort is left out: it needs
# transports and an anchorage objective the policy has no observation of.
RANDOM_MODES = [0, 1, 2, 3]


@dataclass
class Config:
    run_name: str = "mappo"
    out_dir: str = "runs"
    seed: int = 1
    device: str = "auto"

    # ---- environment ----
    workers: int = 8
    unity_binary: str | None = None
    ports: list | None = None
    base_port: int = 5005
    mock: bool = False
    max_team: int = 8
    max_allies: int = 7
    max_contacts: int = 8
    max_zones: int = 5
    max_obstacles: int = 8               # nearest islands / rocks / smoke clouds each ship sees
    action_mode: str = "intent"          # intent | lowlevel (ablation)
    decision_period: float = 1.0
    sim_dt: float = 0.02
    reflexes: bool = True

    # ---- algorithm ----
    algo: str = "mappo"                  # mappo | ippo (ablation: critic sees only the actor's observation)
    critic_mode: str = "privileged"      # privileged | belief (ablation: no true enemy state)
    # tested on the battleship duel: with 256-step rollouts, gamma 0.995 and a +-3 win bonus the
    # signal was too noisy to learn from; these settings learned (see Training/README.md, Results)
    rollout: int = 1024                  # decisions per worker per update
    epochs: int = 4
    minibatches: int = 4
    chunk: int = 32                      # GRU training chunk length
    gamma: float = 0.99                  # ~100 s horizon; consider 0.995 for the 15-20 minute stages
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    lr_actor: float = 2e-4
    lr_critic: float = 5e-4
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001
    ent_decay_updates: int = 300         # entropy bonus decays over this many updates, independent of
                                         # total_updates, so resuming with a new total does not reset it
    value_coef: float = 1.0
    win_coef: float = 0.25
    max_grad_norm: float = 1.0
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    hidden: int = 128
    total_updates: int = 3000
    actor_freeze_updates: int = 0        # train only the critic for this many updates first (use after
                                         # behaviour cloning, so an untrained critic cannot undo it)

    # ---- reward ----
    team_weights: dict = field(default_factory=dict)
    agent_weights: dict = field(default_factory=dict)
    anneal_updates: int = 1500
    shaping_floor: float = 0.1
    spotting_reward: bool = True

    # ---- curriculum and league ----
    stages: list = field(default_factory=lambda: [dict(s) for s in DEFAULT_STAGES])
    start_stage: int = 0
    selfplay_from_stage: int = 3
    mix_latest: float = 0.5
    mix_snapshot: float = 0.3
    mix_rule: float = 0.2
    snapshot_every: int = 25
    pool_size: int = 20
    winrate_window: int = 100
    jitter: float = 40.0                 # scenario ship position jitter (units) per episode

    # ---- io ----
    log_every: int = 1
    checkpoint_every: int = 25
    export_every: int = 25
    export_path: str = "../Assets/StreamingAssets/RL/naval_policy.bin"

    @classmethod
    def load(cls, path: str | None = None, overrides: list[str] | None = None, base: dict | None = None) -> "Config":
        """base: settings to start from (a resumed run's own config), before the file and overrides."""
        data = {k: v for k, v in (base or {}).items() if k in cls.__dataclass_fields__}
        if path:
            with open(path) as f:
                if path.endswith((".yaml", ".yml")):
                    import yaml
                    data.update(yaml.safe_load(f) or {})
                else:
                    data.update(json.load(f))
        cfg = cls(**data)
        for item in overrides or []:
            key, _, raw = item.partition("=")
            if not hasattr(cfg, key):
                raise KeyError(f"unknown config key '{key}'")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
