# Simple MAPPO — the version to learn from

A small, readable MAPPO that trains a fleet in the Unity naval simulation, through a five-stage
curriculum from a battleship duel to random fleet battles. It is laid out like the course's Week 9
and 10 notebooks (config `OrderedDict`, `Monitor`, `SaveOnIntervalCallback`, `model.learn()`,
learning curve, beginning/middle/end replays), trains on your machine or on Amazon SageMaker, and
keeps the team's checkpoints in one shared S3 folder so teammates continue each other's training.

**Start with the notebook: [`Training/WOW-MAPPO-Simple.ipynb`](../WOW-MAPPO-Simple.ipynb).**

The original research pipeline in `naval_rl/` is unchanged; this package only borrows its TCP
connection to Unity.

---

## What the agents learn

Every ship of the Player fleet is an agent. The game's rule-based AI flies the enemy. The objective
is the game's own:

- **get to the centre of the capture circle and hold it** (it scores points every second),
- **sink enemy ships**,
- **do not lose your own**,
- and win the battle (first to 1000 points, or sink the whole enemy fleet; level at the clock goes to
  overtime, so there are no draws).

## Files

| File | What it does | Lines |
|---|---|---|
| [`rewards.py`](rewards.py) | **The reward.** Every term and its weight in `REWARD_WEIGHTS`. Edit this to change what the fleet wants. | ~105 |
| [`curriculum.py`](curriculum.py) | **The stages** (`DEFAULT_CURRICULUM`), random battle generation, and `CurriculumCallback` (promotion, stopping rule) | ~210 |
| [`env.py`](env.py) | `NavalEnv`: runs `n_envs` Unity battles in parallel behind `reset()` / `step()`; self-play | ~375 |
| [`mappo.py`](mappo.py) | The actor, the centralised critic, GAE and the PPO update; `learn` / `save` / `load` / `predict` | ~370 |
| [`team.py`](team.py) | `TeamStorage`: the shared S3 folder per run, with a lock; `TeamSyncCallback` | ~270 |
| [`callbacks.py`](callbacks.py) | `BaseCallback`, `SaveOnIntervalCallback` (as in Stable-Baselines3) | ~110 |
| [`utils.py`](utils.py) | `Monitor`, plots, `evaluate`, battle replays | ~410 |
| [`../train_simple.py`](../train_simple.py) | Command-line training; also the SageMaker training-job entry point | ~170 |
| [`../package_sagemaker.sh`](../package_sagemaker.sh) | Zips the code, scenarios and headless player for SageMaker | |

## How one update works

```
            ┌─────────────── n_envs Unity battles (headless, ~260x real time each) ──────────────┐
            │  battle 0          battle 1          ...          battle E-1                        │
            └────▲───────┬─────────▲───────┬─────────────────────────▲───────┬───────────────────┘
          actions│       │obs      │       │                         │       │   every decision =
         [E,N,6] │       ▼         │       ▼                         │       ▼   1 s of game time
            ┌────┴───────────────────────────────────────────────────────────────┐
  collect   │ actor(local obs of each ship) -> 6 masked categorical heads -> action │  x n_steps
            │ critic(global state + ship's view + ship id) -> value                 │
            │ rewards.py -> reward per ship                                         │
            └───────────────────────────────────┬──────────────────────────────────┘
                                                ▼
  GAE       advantage = how much better each ship's action was than the critic expected
                                                ▼
  update    10 epochs x 4 minibatches:  clipped policy loss (alive ships only)
                                        + vf_coef x value loss on normalised returns (every ship slot)
                                        - ent_coef x entropy
```

## How this is MAPPO (and not just PPO)

MAPPO (Yu et al. 2022, https://arxiv.org/abs/2103.01955) is PPO for a cooperative team, built on
**centralised training with decentralised execution**:

- **Parameter sharing**: one actor network for every ship.
- **Decentralised actor**: each ship acts only on what it can see (`obs["local"]`, fog of war
  included), so the trained policy can play the real game.
- **Centralised critic**: during training the critic sees the true state of the whole battle.

The paper's five implementation recommendations, and where they are in `mappo.py` / `env.py`:

| # | Recommendation | Here |
|---|---|---|
| 1 | Value normalisation | `ValueNorm`: the critic learns returns rescaled to about unit size, so a 1v1 duel and an 8v8 battle train the critic alike |
| 2 | Agent-specific global state | `env.py`: critic input = the whole battle + this ship's own view + its id |
| 3 | Few minibatches, several epochs | `n_minibatches=4`, `n_epochs=10`: 40 gradient steps per update, the same at every stage |
| 4 | PPO clip ≤ 0.2 | `clip_range=0.2` |
| 5 | Death masking | a sunk ship stops training the actor; its critic keeps learning the team's value |

`MAPPO(env, centralised_critic=False)` (or `train_simple.py --ippo`) gives the critic only the
ship's own view instead — that is **IPPO**, the baseline the paper compares MAPPO against, for
showing what the centralised critic adds. `normalize_values=False` (`--no-value-norm`) switches
recommendation 1 off, for the same kind of comparison.

**The networks** are two-layer MLPs (256 units). The game already scales every input (almost all
within −1.5…1.5), so the inputs go in as they are, and LayerNorm sits after the first layer: on the
raw input, the zeros of a small battle's empty slots (80% of the input in a duel, 70% in an 8v8)
would change the normalisation from stage to stage, and the policy would see its inputs shift at
every promotion.

## One method for every stage

Every stage — and any stage you add later — is trained by the **same model**: the same MAPPO, the
same network, the same hyperparameters and the same reward function. The observation always has
room for 8 ships a side and 5 circles (`MAX_SHIPS` / `MAX_ZONES` in `env.py`); a duel simply leaves
seven ship slots empty. Promotion only changes which battles are played; the model keeps training,
and a checkpoint from any stage continues on any other. A stage that needs more room than that is
refused with an error instead of silently building a different network.

## The observation, actions and reward

**Observation.** Each ship's actor input (`obs["local"]`) is its own state, every ally, every enemy
contact its team has spotted, every capture circle (with the distance to it and an "I am inside"
flag) and the nearest islands. The critic input (`obs["state"]`) is the true state of both fleets
and every circle, plus the ship's own view and its id. Both have the same size in every stage
(room for 8 ships a side and 5 circles: 758 and 2668 numbers); smaller battles fill the empty slots
with zeros and a mask. Unity sends the name of every feature (`env.spec.features`).

**Actions.** Six heads, one option each per decision; illegal options are masked:
move (incl. **sail to circle k**), speed, target, fire, torpedo, ability.

**Reward** (`rewards.py`):

| Term | Weight | Kind |
|---|---|---|
| `in_circle` | +0.05 per second inside a capture circle | ship |
| `approach_circle` | +1 per km closer to the nearest circle's centre (− when moving away) | ship |
| `zone_captured` / `zone_lost` | +1 / −1 | team |
| `enemy_sunk` / `ship_lost` | +1 / −1 per ship | team |
| `damage_dealt` / `damage_taken` | +0.5 / −0.25 per whole hull | ship |
| `friendly_fire` | −1 per whole hull | ship |
| `win` / `loss` | +5 / −5 | team |

`approach_circle` is a *potential-based* shaping term (the change in distance), so a ship cannot
farm it by sailing back and forth — only real progress pays.

## The curriculum

`curriculum.py`, `DEFAULT_CURRICULUM`:

| Stage | Training scenario | Fleet | Environment | Opponent | Promotion / stopping requirement |
|---|---|---|---|---|---|
| 0 – Battleship Duel | `bb_duel` | 1v1 battleships | Open-sea duel scenario | Recruit rule-based AI | ≥55% rolling win rate after at least 100 qualifying episodes |
| 1 – King of the Hill | `koth_3v3` | 3v3 mixed fleet | Single-objective combat scenario | Veteran rule-based AI | ≥65% rolling win rate after at least 100 qualifying episodes |
| 2 – Archipelago | `archipelago_3v3` | 3v3 mixed fleet | Islands / archipelago scenario | Veteran rule-based AI | ≥65% rolling win rate after at least 150 qualifying episodes |
| 3 – Full Domination | `domination_6v6` | 6v6 fleet | Procedural battlefield with random weather | Elite rule-based AI + self-play | ≥60% rolling win rate after at least 200 qualifying Elite rule-AI episodes |
| 4 – Open / Generalisation | `open` | Random 4–8 ships per side | Random map preset, terrain density and weather | Elite rule-based AI + self-play | Final stage: stop at ≥60% rolling win rate after at least 250 qualifying held-out evaluation episodes against the Elite rule-based AI |

- **Qualifying episodes** are battles that started on the current stage against the rule-based AI;
  self-play battles never count, and the window starts empty on every new stage.
- **Stage 0 has no circle reward**: in the duel the circle lies between the two battleships, so
  paying for it sends the ship bow-on into the enemy's guns (see Results). It is the only stage
  setting that changes the reward; the model is the same everywhere.
- **Self-play** (stages 3–4): half the battles (`"self_play": 0.5`) are flown on the enemy side by a
  frozen copy of our own actor, refreshed every `opponent_refresh` (10) updates.
- **Held-out evaluation** (stage 4): every 100,000 timesteps, 50 battles against the Elite AI on
  generated maps whose seeds are multiples of 10 — seeds training never uses. Training stops once
  the last 250 of them reach 60%.
- **Any number of stages**: a stage is a dict (see the docstring in `curriculum.py`); add as many as
  you like, as a list in the notebook or a JSON file (`train_simple.py --curriculum my_stages.json`).
  The last stage trains until its stopping rule is met; without `stop_win_rate` it trains forever.
- A saved model remembers its stage, so `MAPPO.load` (and the team checkpoints) carry on where the
  curriculum was.

## Team checkpoints

Everyone trains the same line of training in turns, from a notebook, a SageMaker training job or
their own computer. Each run is one folder:

```
s3://<bucket>/wow-mappo/runs/<run-name>/
    run.json                   who pushed last, when, timesteps, curriculum stage
    LOCK.json                  who is training it right now
    config.json
    logs/monitor.csv           every battle from every session: the team's learning curve
    logs/progress.csv
    models/model_latest.pt     the checkpoint to continue from (+ model_<n>.pt, model_final.pt)
```

- **Start**: pull `model_latest.pt` and the logs, continue from them (stage and timesteps included).
- **During training**: a new checkpoint and the logs are pushed every `save_interval` timesteps.
- **End** (or after stopping early): push once more and release the lock.
- **Two people at once?** `lock()` refuses while someone else holds the run and has pushed within
  the last 30 minutes, and `push()` refuses if someone else pushed after you pulled, so nobody
  overwrites a teammate's progress. Train in parallel under different run names instead (e.g. a
  reward experiment).
- Everyone must use the same bucket. SageMaker's default bucket belongs to one AWS account: if
  teammates log in to different accounts, create one bucket and give everyone access.

```bash
python train_simple.py --team-storage s3://<bucket>/wow-mappo --run-name curriculum --user Lukita
```

In the notebook, set `YOUR_NAME`, `RUN_NAME` and `TEAM_STORAGE` in the "Team checkpoints" cell.
A plain folder path also works in place of `s3://` (a shared drive, or for testing).

## Running it

### On your computer

```bash
cd Training
jupyter lab WOW-MAPPO-Simple.ipynb                       # the walkthrough
python train_simple.py --total-timesteps 5000000         # the curriculum, headless
python train_simple.py --scenario scenarios/stage1_koth_3v3.json --run-name koth   # one battle only
```

Needs Python 3.10+, PyTorch, NumPy, pandas, matplotlib, Pillow, and the headless player
(`Training/build_player.sh` builds it into `Builds/NavalTrainer/`).

### On Amazon SageMaker

1. Locally: `Training/build_player.sh`, then `Training/package_sagemaker.sh` → `wow_sagemaker.zip`
   (≈40 MB: code, scenarios, the Linux player).
2. Upload the zip to SageMaker Studio, open `Training/WOW-MAPPO-Simple.ipynb` from it, run the
   first cell (it unzips to `~/wow`), and fill in the "Team checkpoints" cell.
3. Use a JupyterLab space with many CPUs and memory (e.g. `ml.c5.4xlarge`: 16 vCPU, 32 GB). No GPU
   is needed. The default `ml.t3.medium` (2 vCPU, 4 GB) can only run one battle and is too small for
   the 8-ship stages.
4. The whole curriculum takes millions of timesteps: use the notebook's last section, which starts
   a **SageMaker training job** (`train_simple.py` as the entry point, the player as the `unity`
   input channel) that trains the same team run — it pulls the latest checkpoint, pushes every
   `save-interval` timesteps, and a teammate can continue it from a notebook afterwards.

**Requirements the code checks for you:**

- **glibc 2.35+ (Ubuntu 22.04 or newer).** The Unity 6 player needs it. For training jobs use a
  PyTorch image **2.4 or newer** (`framework_version="2.5"`, `py_version="py311"`, as in the
  notebook); the 2.3 and older images are Ubuntu 20.04 and the player cannot start there. The code
  stops with this explanation if the system is too old.
- **S3 access.** The SageMaker execution role needs `s3:GetObject`, `s3:PutObject` and
  `s3:ListBucket` on the team bucket. The default `sagemaker-<region>-<account>` bucket already has
  them; for your own bucket, name it with "sagemaker" in it or add the permissions.
- The player's other libraries are only libc, libm and libgcc (`ldd`), and the execute permission
  that zip files and S3 drop is restored automatically.

Tested here without AWS: the bundle unzipped into an empty folder and trained with the SageMaker
environment variables (`SM_MODEL_DIR`, `SM_CHANNEL_UNITY`, ...) set and the player's execute bit
stripped; and two "teammates" taking turns on one shared run (a local folder standing in for S3),
including the lock and the stale-push refusal. The S3 calls themselves (boto3) and a real SageMaker
job could not be run here.

### Speed

Training never runs at ×1. Each frame is a fixed `sim_dt` of game time and frames run as fast as
the CPU allows. Measured on one battle (King of the Hill 3v3, random actions):

| `sim_dt` | Speed of one battle | 24 battles on 8 cores | Battle results |
|---|---|---|---|
| 0.02 (the game's normal step) | ×90 real time | 60 s | win 13%, damage dealt / taken 2.40 / 3.21 hulls |
| **0.08 (default)** | **×260 real time** | **31 s** | win 4%, damage dealt / taken 2.19 / 3.30 hulls |
| 0.10 (the environment's maximum) | ×265 real time | — | — |

0.08 is the physics step the game itself uses at ×8 time compression; the battle statistics match
the normal step within their error bars, and larger steps buy almost nothing more. Training runs at
about 200 timesteps per second with 8 parallel battles.

## Results

All numbers are win rates against the rule-based AI with sampled actions, measured on this machine
(16 CPU cores, no GPU) with the current game rules (no draws).

### King of the Hill 3v3 vs Recruit — the reward matters

| Policy | Win rate | Battles |
|---|---|---|
| Random legal actions | 0.10 ± 0.05 | 48 |
| Scripted: every ship sails to the circle and fires at will | 0.67 ± 0.10, 0.88 ± 0.08 | 24, 16 |
| MAPPO, first reward weights (circle 0.02 / 0.2, kills ±2), 90k timesteps | 0.14 in training; never entered the circle | 253 |
| **MAPPO, circle-focused weights (the defaults), 600k timesteps, 67 min** | **0.33 ± 0.07** | 48 |

With the circle-focused weights the ships went from about 2 seconds inside the circle per battle
(first 150 battles) to about half a minute each (last 300), captured it in about one battle in eight,
and sank twice as many enemies as random play (1.9 vs 0.9 per battle). It still loses the circle to the AI on
points more often than not — an hour of training is not enough to beat the simple scripted tactic.
A 2-second `decision_period` ran twice as many battles per hour but learned less per battle; 1 s
stays the default.

### Battleship duel (curriculum stage 0) vs Recruit — what made it learn

**The shipped configuration wins 0.59 ± 0.06 after 300k timesteps** (random play 0.16 ± 0.05 in the
same evaluation, 63 duels each; it sinks the enemy in 35% of duels vs 8%, and loses its own ship in
38% vs 60%). Its training win rate per 50k timesteps: 0.23 → 0.32 → 0.38 → 0.44 → 0.43 → 0.46.

How it got there — each row changes one thing:

| Version | Duel win rate |
|---|---|
| Random legal actions | 0.16 – 0.28 |
| Scripted: sail to the circle / broadside and fire at will | 0.33 / 0.70 |
| 1 minibatch per epoch (5 gradient steps per update), circle reward on | 0.21 – 0.31 for 600k timesteps: no learning |
| 4 minibatches × 10 epochs, circle reward off (duel-sized network) | 0.25 → 0.44, 0.41 ± 0.06 at 400k |
| Same, one network for all stages, LayerNorm on the raw input, fast value normalisation | 0.16 – 0.24 for 250k: stalled |
| LayerNorm after the first layer, no value normalisation | 0.28 → 0.31 at 150k |
| **LayerNorm after the first layer, MAPPO's value normalisation (beta 0.99999)** | **0.23 → 0.46; 0.59 ± 0.06 at 300k** |

What came out of it, all now in the code:

1. **A fixed number of minibatches** (4 × 10 epochs = 40 gradient steps per update). With a fixed
   minibatch *size*, a duel's single ship gave 5 steps per update and an 8v8 battle over 100.
2. **No circle reward in the duel.** Its circle lies between the two battleships; paying for it
   pulls the ship bow-on into the enemy's guns. The only stage setting that changes the reward.
3. **LayerNorm after the first layer**, not on the raw input: the empty slots of a padded battle
   (80% of the input in a duel) would otherwise change the normalisation from stage to stage.
4. **MAPPO's value normalisation with the paper's slow statistics** (beta 0.99999). A fast version
   (0.99) moved the critic's targets every update and stalled learning.

### What was checked end to end

- **Multi-agent learning, 3v3 King of the Hill vs Veteran (stage 1), shipped configuration, 400k
  timesteps from scratch**: 0.25 ± 0.06 vs random 0.18 ± 0.05 (63 battles each); training win rate
  0.18 → 0.28, more enemies sunk (1.7 vs 1.25 a battle) and fewer ships lost (1.6 vs 1.8). It learns,
  but beating the Veteran AI needs far more training — on SageMaker, and starting from the duel as
  the curriculum does.
- **Multi-agent mechanics, in real 3v3 and 6v6 battles**: every ship gets its own observation and
  makes its own choices from the one shared actor; the critic's input is the same battle for every
  ship plus that ship's own view and id; ships earn different ship rewards (7.7 / 10.2 / 8.5 in one
  battle) and the same team rewards (3 enemies sunk, 1 ship lost, the win); a ship that sinks leaves
  actor training while its teammates carry on; and a 6v6 battle runs on the same network.
- Every curriculum stage runs in the real game: the duel, King of the Hill, the archipelago,
  generated 6v6 domination maps and the open stage (all modes, 4–8 ships, up to 5 circles), with
  promotion between them, self-play battles (both fleets flown from Python) in stages 3–4 and the
  held-out evaluation and stopping rule in stage 4.
- The notebook runs top to bottom (training, checkpoints, all plots, per-stage evaluation, replays),
  and two "teammates" can take turns on one shared run from the notebook and from `train_simple.py`.
- A SageMaker training job was emulated with the unzipped bundle.

## Why a simpler version? A review of the original method (`naval_rl/`)

The original pipeline is a serious research setup:

- a **transformer** over every ship, contact, zone and island token, a **GRU** memory, and
  **pointer heads** that pick targets and zones by attention;
- a centralised critic with a second **win-probability** head, running value normalisation,
  death masking and recurrent chunked training;
- **behaviour cloning** from the rule AI before RL, a **5-stage curriculum** with promotion
  thresholds, and **league self-play** with prioritised snapshot sampling;
- 11 team + 5 ship reward components with **annealed shaping**;
- the whole network re-implemented in C# (`RLPolicy.cs`) for in-game play, with a parity test.

Its measured results (100 battles each unless noted, sampled actions):

| Stage | Original MAPPO | Behaviour cloning | Random | Scripted |
|---|---|---|---|---|
| 0 — battleship duel vs Recruit | 0.47 ± 0.05 (100 updates) | 0.37 ± 0.05 | 0.39 ± 0.05 | 0.63 ± 0.05 |
| 1 — King of the Hill 3v3 vs Veteran (60) | 0.20 ± 0.05 | 0.58 ± 0.06 | 0.03 ± 0.02 | 0.68 ± 0.06 |
| 2 — Archipelago 3v3 vs Veteran (60) | 0.27 ± 0.06 | 0.45 ± 0.06 | 0.22 ± 0.05 | 0.43 ± 0.06 |

Reading those numbers, the likely reasons it struggled:

1. **Too many parts at once.** When it does not learn, the cause could be any of a dozen
   components; it is hard for a team to debug or explain. The simple version keeps MAPPO's three
   essentials and drops the rest.
2. **A large model for little training.** The transformer + GRU got 40–100 updates per stage. That
   is a lot of model to fit in so few updates; two small MLPs are a better match for the budget.
3. **The objective barely paid.** A zone capture was worth ±0.1 and the score margin 2 × points / 1000
   (holding the circle for a whole minute: about +0.2), while damage and kill shaping paid up to
   ±1.3 a battle on top of ±1 for the result. `Training/README.md` notes that the cloned policy
   "loses mostly on objective points". Nothing paid a ship for sailing towards the circle; the new
   reward does, and pays every second it holds it.
4. **Slower than needed.** Training simulated at the 0.02 s step; 0.08 runs about 2× faster end to
   end with the same battle results.

Kept from the original: the Unity environment and its TCP protocol, the full observation (the whole
battle — nothing was removed), the six action heads, and MAPPO's centralised critic, parameter
sharing and death masking. The advanced pipeline is still there for anyone who wants it.

| | Original (`naval_rl/`) | Simple (`simple_mappo/`) |
|---|---|---|
| Actor | transformer + GRU + pointer heads | 2-layer MLP |
| Critic | transformer + value & win heads, value normalisation | 2-layer MLP |
| Training | BC → curriculum → league self-play | MAPPO against the rule AI |
| Reward | 16 components, annealed | 11 terms, one readable function |
| Code to read | ~2,700 lines of Python + ~4,000 of C# RL code | ~2,000 lines of Python (curriculum, self-play and team storage included) |
| Plays inside the game | yes (C# re-implementation) | through the trainer (e.g. the editor on port 5005) |

## Tips

- **A model is tied to its battle size.** The MLP's input size depends on the number of ships and
  circles, so a model trained on 3v3 cannot run a 6v6; train a new one.
- **"port already in use"**: an earlier run's players are still closing; wait a few seconds or use
  another `base_port`.
- **Watch it live**: in the Unity editor tick `GameBootstrap.rlTrainingServer`, press Play, and
  create the environment with `NavalEnv(..., ports=[5005])` (see the end of the notebook).
