# Training Settings — Run `mappo`, Stages 0 to 2

Recorded 4 October 2026, 5:44 pm AEDT (Sydney time).

These are the exact settings the run `Training/runs/mappo` used from update 1 (stage 0) through stage 2.
At the time of writing it was still on stage 2 (`archipelago_3v3`, around update 4200).

**How this was checked:** the settings are taken from `Training/runs/mappo/config.json`. They were compared
with the config stored inside the checkpoints at updates 250 (stage 0), 1000 (stage 1) and 2750 (stage 2).
They are identical in every setting except `total_updates`, which was 3000 until update 2907 and 10,000 after.

## Code and game build

| | |
|---|---|
| Training code | `Training/` at git commit `933fe36` (`main`). The `Sagemaker` branch commit `8dab75e` does not change any training code. |
| Game build | `NavalTrainerServer/NavalTrainerServer.x86_64` (headless Linux build), 8 copies |
| Machine | SageMaker `ml.g4dn.2xlarge`: 8 vCPU, 32 GB RAM, Tesla T4 GPU |

## Commands used

All run from `Training/`:

| Updates | Command |
|---|---|
| 1–220 | `python3 train.py --unity-binary ../NavalTrainerServer/NavalTrainerServer.x86_64` (all defaults) |
| 221–2907 | `python3 train.py --unity-binary ../NavalTrainerServer/NavalTrainerServer.x86_64 --resume runs/mappo/checkpoints/latest.pt` |
| 2908– | same as above, plus `--set total_updates=10000` |

No other settings were overridden: every value below is the default from `Training/naval_rl/config.py`
and `Training/naval_rl/rewards.py`.

## Environment

| Setting | Value | Meaning |
|---|---|---|
| `workers` | 8 | Parallel game copies (ports 5005–5012) |
| `decision_period` | 1.0 | One decision per ship per second of game time |
| `sim_dt` | 0.02 | Physics step, so 50 frames per decision |
| `action_mode` | `intent` | High-level orders (move, speed, target, fire, torpedo, ability) |
| `reflexes` | true | In-game reflexes enabled |
| `max_team` / `max_allies` | 8 / 7 | Largest fleet the networks handle |
| `max_contacts` / `max_zones` / `max_obstacles` | 8 / 5 / 8 | Observation limits per ship |
| `seed` | 1 | |

## Algorithm (MAPPO)

| Setting | Value | Meaning |
|---|---|---|
| `algo` | `mappo` | Centralised critic |
| `critic_mode` | `privileged` | Critic sees the true enemy state |
| `rollout` | 1024 | Decisions per game per update, so 8192 per update |
| `epochs` / `minibatches` | 4 / 4 | 16 gradient steps per update |
| `chunk` | 32 | GRU training sequence length (steps) |
| `gamma` | 0.99 | Discount, about a 100 s horizon |
| `lam` | 0.95 | GAE lambda |
| `clip` / `value_clip` | 0.2 / 0.2 | PPO clipping |
| `lr_actor` / `lr_critic` | 2e-4 / 5e-4 | Adam learning rates |
| `ent_coef` → `ent_coef_final` | 0.01 → 0.001 | Entropy bonus, decays linearly … |
| `ent_decay_updates` | 300 | … over the first 300 updates, then stays at 0.001 |
| `value_coef` / `win_coef` | 1.0 / 0.25 | Value loss weight, auxiliary win-prediction weight |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `actor_freeze_updates` | 0 | Actor trained from the first update |
| `total_updates` | 3000, then 10,000 from update 2908 | Stop limit |

## Networks

| Setting | Value |
|---|---|
| `d_model` | 128 |
| `heads` | 4 |
| `layers` | 2 |
| `hidden` (GRU) | 128 |

- **Actor:** transformer entity encoder, then GRU, then action heads. Target and zone choices are pointer
  heads. 587,560 parameters.
- **Critic:** separate transformer encoder giving a value and a win probability. 399,234 parameters.

## Rewards

`team_weights` and `agent_weights` were empty, so the defaults from `rewards.py` applied.

**Team terms** (shared by every ship on a side, zero-sum between sides):

| Term | Weight | Annealed? |
|---|---|---|
| `score_delta` | 2.0 | No (objective) |
| `win` / `loss` / `draw` | 1.0 / −1.0 / 0.0 | No (objective) |
| `damage_dealt` / `damage_taken` | 1.0 / −1.0 | Yes |
| `kills` / `losses` | 0.3 / −0.3 | Yes |
| `zones_captured` / `zones_lost` | 0.1 / −0.1 | Yes |
| `friendly_fire_taken` | −0.5 | Yes |

**Agent terms** (per ship, all annealed):

| Term | Weight |
|---|---|
| `damage_dealt` | 0.2 |
| `spotting_damage` | 0.2 |
| `friendly_fire_dealt` | −0.5 |
| `sunk` | −0.1 |
| `damage_taken` | 0.0 |

**Shaping schedule:** annealed terms are multiplied by a factor that falls linearly from 1.0 to
`shaping_floor = 0.1` over `anneal_updates = 1500` updates, then stays at 0.1. `spotting_reward` is on.

## Effective schedule values by stage

| Stage | Updates | Entropy coefficient | Shaping factor |
|---|---|---|---|
| 0 `bb_duel` | 1–755 | 0.01 → 0.001 (floor reached at update 300) | 1.0 → 0.55 |
| 1 `koth_3v3` | 756–2730 | 0.001 | 0.55 → 0.1 (floor reached at update 1500) |
| 2 `archipelago_3v3` | 2731– | 0.001 | 0.1 |

## Curriculum and league

| Setting | Value |
|---|---|
| `start_stage` | 0 |
| `winrate_window` | 100 battles (promotion uses the rolling win rate against the rule AI at the stage difficulty) |
| `selfplay_from_stage` | 3, so stages 0–2 were 100% against the rule-based AI |
| `mix_latest` / `mix_snapshot` / `mix_rule` | 0.5 / 0.3 / 0.2 (from stage 3) |
| `snapshot_every` / `pool_size` | 25 / 20 (from stage 3) |
| `jitter` | 40 units of random start position, plus or minus 15° heading |

**Stages used so far** (ship classes: 0 Destroyer, 1 Cruiser, 2 Battleship; fog of war on everywhere):

| Stage | Scenario | Fleets per side | Map | Zones | Time limit | AI | Promote at |
|---|---|---|---|---|---|---|---|
| 0 `bb_duel` | `scenarios/stage0_bb_duel.json` | 1 Battleship | Open sea, 13 km apart | 1 (A) | 180 s | Recruit | 0.55, at least 100 battles |
| 1 `koth_3v3` | `scenarios/stage1_koth_3v3.json` | Battleship, Cruiser, Destroyer | Open sea, one central cap | 1 (A) | 360 s | Veteran | 0.65, at least 100 battles |
| 2 `archipelago_3v3` | `scenarios/stage2_archipelago_3v3.json` | Battleship, Cruiser, Destroyer | Islands (density 1) | 3 (A, B, C) | 480 s | Veteran | 0.65, at least 150 battles |

**Stages still to come:**

- **3 `domination_6v6`:** procedural 6v6, random preset, density and weather, 900 s limit, Elite AI,
  promote at 0.60 over at least 200 battles.
- **4 `open`:** random mode, 3–8 ships, 1200 s limit, Elite AI, no promotion.

## Logging and saving

| Setting | Value |
|---|---|
| `log_every` | 1 (a line and a `metrics.jsonl` entry every update) |
| `checkpoint_every` | 25 (`latest.pt`), with a numbered copy every 250 updates |
| `export_every` | 25, to `Assets/StreamingAssets/RL/naval_policy.bin` |
