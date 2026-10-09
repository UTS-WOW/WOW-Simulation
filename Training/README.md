# Training — MAPPO for the naval simulation

A recurrent MAPPO trainer for the fleets in this Unity project. The Unity side lives in
`Assets/Scripts/RL/`; the Python side lives here.

This page covers **how to run it**. For **how it works** — observations, actions, rewards, networks,
the algorithm, curriculum and export, with diagrams — see [../RL_README.md](../RL_README.md).

> **New to the project? Start with the simple version.** [`simple_mappo/`](simple_mappo/README.md)
> is a small MAPPO (two MLPs, one readable reward function, the five-stage curriculum and shared
> S3 team checkpoints, ~2,000 lines) with a course-style
> notebook, [`WOW-MAPPO-Simple.ipynb`](WOW-MAPPO-Simple.ipynb), that also runs on Amazon SageMaker.
> It uses the same Unity environment and the same observation; everything below is the full
> research pipeline.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Training as a team](#training-as-a-team)
- [Configuration](#configuration)
- [Tools for checking and watching](#tools-for-checking-and-watching)
- [Playing against the trained policy](#playing-against-the-trained-policy)
- [What a run writes to disk](#what-a-run-writes-to-disk)
- [Results so far](#results-so-far)

---

## Requirements

- **Python 3.10+**, **PyTorch ≥ 2.1** and **NumPy ≥ 1.24**:

  ```bash
  pip install -r requirements.txt
  ```

- **Optional:** TensorBoard (used automatically if installed) and PyYAML (only needed for
  `--config file.yaml`).
- **A Linux build of the game** — see step 1 below.
- **`dotnet`**, only for the C# ⇄ PyTorch parity test.

The commands below call `python`; use the interpreter that has PyTorch (for example after
`conda activate`). The shell scripts (`check_paths.sh`, `pipeline_bc_mappo.sh`) take it from the
`PYTHON` environment variable when set, e.g. `PYTHON=~/anaconda3/bin/python ./check_paths.sh`.

---

## Quick start

Unless noted otherwise, run commands from the `Training/` folder.

### 1. Build the headless training player

From the repository root:

```bash
Training/build_player.sh
```

This builds `Builds/NavalTrainer/NavalTrainer.x86_64` from a synced copy of the project, so it works
even while the editor is open. The same build is available in the editor under
**Naval > RL > Build Training Player (Linux)**.

### 2. Train

Train from scratch with 8 parallel battles:

```bash
python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --workers 8 --run-name first
```

### 3. (Optional) Watch it learn in the editor

In the editor, tick **GameBootstrap → Rl Training Server** and press Play. Then:

```bash
python train.py --ports 5005 --run-name watch
```

### Recommended recipe for the full game

Learning the full game from random play is slow, so first imitate the rule-based AI (behaviour
cloning), then fine-tune with MAPPO:

```bash
# 1. record rule AI vs rule AI battles, with every decision labelled (~7 min)
python bc.py collect --battles 120

# 2. behaviour cloning (~3 min on a GPU)
python bc.py train

# 3. MAPPO fine-tuning from the cloned policy
python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --run-name bc_mappo \
    --init-from runs/bc/bc.pt --set start_stage=3 --set actor_freeze_updates=10 --set lr_actor=5e-5 \
    --set ent_coef=0.002 --set ent_coef_final=0.0005 --set gamma=0.995 --set minibatches=16
```

[`pipeline_bc_mappo.sh`](pipeline_bc_mappo.sh) chains the whole thing: collect demonstrations (unless
`runs/bc/demos.npz` exists) → behaviour cloning → evaluate → install in the game → MAPPO fine-tuning
→ automatic checkpoint evaluation.

### Check the trainer without Unity

```bash
python train.py --mock --workers 4 --set total_updates=50
python -m pytest tests
```

### Resume a run

```bash
python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --resume runs/<run>/checkpoints/latest.pt
```

This restores the weights, optimisers, value normaliser, league, curriculum stage and counters. To
continue a checkpoint a teammate shared, see [Training as a team](#training-as-a-team).

---

## Training as a team

Your own runs stay on your machine (`runs/` is gitignored). To let the rest of the team continue your
training, **publish** a checkpoint into `checkpoints/<name>/`, which is committed to git. Run every
command in this section from the `Training/` folder unless it says otherwise.

### Shared checkpoints

| Name | What it is | Use it with |
|---|---|---|
| `mappo_terrain` | MAPPO fine-tuning of `bc_terrain` on 6v6 domination over random battlefields (stage 3). **The main line to keep training.** | `--resume` |
| `bc_terrain` | Behaviour cloning of the Elite rule AI with the current observation layout (terrain, obstacles, full consumable state) | `--init-from`, to start a new line |
| `full_mappo`, `bc` | The previous lines, trained on the **first observation layout**. The current game cannot resume or load them; kept only for the record. | — |

`python share_checkpoint.py list` shows what is currently shared, who published it and when.

### Continue the team's training

1. Get the latest shared checkpoint:

   ```bash
   git pull
   ```

2. Build the headless player once, **from the repository root** (`Builds/` is not in git, so
   everyone builds their own; rebuild after pulling game code changes):

   ```bash
   Training/build_player.sh
   ```

3. Resume the shared checkpoint:

   ```bash
   python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --resume checkpoints/mappo_terrain/latest.pt
   ```

   This trains in your own `runs/mappo_terrain/`, with the shared checkpoint's settings, and copies
   its self-play opponents there. To watch in the editor instead, use `--ports 5005` in place of
   `--unity-binary ...`. Use `--workers` to match your machine; it does not change the training.

4. Stop with `Ctrl+C` whenever you like. The trainer saves a checkpoint before exiting.
   To carry on later with **your own** session, resume `runs/mappo_terrain/checkpoints/latest.pt`
   instead — but if a teammate has published in the meantime, resume their new shared checkpoint
   (step 3) rather than yours, or your next publish will be refused.

5. Pull again (in case someone published meanwhile), then publish your progress. The trainer
   prints this exact command when you resume a shared checkpoint:

   ```bash
   git pull
   python share_checkpoint.py publish mappo_terrain --note "what you changed or noticed"
   ```

6. Commit and push the checkpoint folder straight away (the publish command prints these too):

   ```bash
   git add checkpoints/mappo_terrain
   git commit -m "Share checkpoint mappo_terrain at update N"
   git push
   ```

### Start a new shared line

After changing the observation or action layout, or to try different settings without disturbing
the main line, start from a warm start and publish under a new name:

```bash
python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --run-name my_line \
    --init-from checkpoints/bc_terrain/latest.pt --set start_stage=3 --set actor_freeze_updates=10 --set lr_actor=5e-5 \
    --set ent_coef=0.002 --set ent_coef_final=0.0005 --set gamma=0.995 --set minibatches=16
python share_checkpoint.py publish my_line --note "what this line is for"
```

### Rules that keep everyone's training safe

- **One shared name is one line of training — take turns.** Tell the team when you start a session.
- `publish` **refuses** to overwrite a shared checkpoint when:
  - your run did not start from it,
  - someone else published to it after your run started, or
  - your checkpoint is not ahead of it.

  In that case publish under your own name (for example `--name mappo_terrain_alex`), compare the two
  with `python evaluate.py --checkpoint checkpoints/<name>/latest.pt --stage 3 --episodes 48`, and
  publish the better one onto the main line with `--force`.
- **Pull right before publishing, push right after.** If git still reports a conflict on
  `latest.pt`, do not simply keep your own copy — that discards a teammate's work. Publish yours
  under another name instead.
- **Settings travel with the checkpoint.** Any `--set` you pass when resuming (for example a smaller
  `rollout`) is saved into your checkpoint, and everyone who resumes your publish inherits it. Only
  override training settings on purpose, and say so in `--note`.
- **A checkpoint only fits a game with the same observation and action layout.** After changing
  `RLLayout.cs`, `RLObservation.cs` or `RLActions.cs`, old checkpoints cannot be resumed (`train.py`
  says so and lists the differences). Start a new line under a new name.
- **Training also rewrites the game's policy file**, `Assets/StreamingAssets/RL/naval_policy.bin`.
  Commit it together with the checkpoint if the game should play the new policy; otherwise discard
  it with `git restore`.
- Each publish adds about 7 MB to the repository, plus about 2 MB for each new self-play snapshot.

### What a shared folder contains

| File | Contents |
|---|---|
| `latest.pt` | The checkpoint, with its self-play snapshot paths made relative so it works on any machine |
| `snapshots/` | The frozen past policies its self-play league needs |
| `info.json` | Update, curriculum stage, rolling win rate vs the rule AI, author, date and note |
| `history.jsonl` | One line per publish — the line's full history |

---

## Configuration

Every setting lives in [`naval_rl/config.py`](naval_rl/config.py). Override any of them with
`--set key=value` (values are parsed as JSON), or load a file with `--config file.yaml`:

```bash
python train.py ... --set workers=12 --set rollout=512
```

- **All hyperparameters and their defaults:** [RL_README.md § 10.6](../RL_README.md#106-hyperparameters)
- **Curriculum stages** (`--set start_stage=N` skips ahead):
  [RL_README.md § 11](../RL_README.md#11-curriculum-and-self-play-league)
- **Ablation switches** (`algo=ippo`, `critic_mode=belief`, …):
  [RL_README.md § 14](../RL_README.md#14-ablation-switches)

---

## Tools for checking and watching

| Command | What it does |
|---|---|
| `python -m pytest tests` | Trainer unit tests, plus a parity test showing the in-game C# policy matches PyTorch to about 1e-7 (needs `dotnet`). |
| `./check_paths.sh` | Briefly runs every curriculum stage, self-play against snapshots, and every ablation mode against the real game. |
| `python probe_actions.py` | Issues hand-written orders in the archipelago battle and checks each one works: moving through islands without grounding, radar, smoke, torpedoes, hold fire. Films it to `runs/probe/battle.mp4`. |
| `python watch.py --checkpoint runs/<run>/checkpoints/latest.pt --stage 3` | A trained policy plays one battle, filmed to `runs/watch/battle.mp4` and `.gif`. |
| `python evaluate.py --checkpoint <ckpt> --stage 3 --episodes 48` | Win rate ± standard error and damage against the rule AI. `--random` and `--scripted` give baselines; `--max-team 12` tests zero-shot on bigger fleets; `--set 'stages=[...]' --stage 0` tests one chosen battlefield. |
| `python watch_checkpoints.py --run <run>` | Evaluates every checkpoint of a running training run (96 battles, same seed each time) and appends a row to `runs/<run>/evals/summary.jsonl`. |
| `python share_checkpoint.py publish <run>` / `list` | Shares a run's checkpoint with the team through git, or lists what is shared — see [Training as a team](#training-as-a-team). |
| `python bc.py collect` / `python bc.py train` | Records rule-AI battles with every decision labelled in the policy's action heads, then trains the actor to imitate them (the warm start for MAPPO). |
| `NavalTrainer.x86_64 -rlDemo both` | Launches the game straight into a 6v6 with the exported policy flying both fleets. `enemy` = you against it, `player` = it against the rule AI. Also takes `-rlDemoShips 6` and `-rlDemoSeconds 60`. |

---

## Playing against the trained policy

Every `export_every` updates (default 25), and when a run ends, the trainer writes the policy to
`Assets/StreamingAssets/RL/naval_policy.bin`, which the game loads at startup.

In the game's setup screen:

- **Enemy AI: Learned** — play against the policy.
- **Your fleet: Learned** — watch it command your side.

The policy runs in plain C# ([`RLPolicy.cs`](../Assets/Scripts/RL/RLPolicy.cs)) with no extra
package: one decision per ship per second, with the forward passes spread over several frames.

---

## What a run writes to disk

Each run writes to `runs/<run_name>/`:

| File | Contents |
|---|---|
| `config.json` | The exact config used |
| `spec.json` | The environment spec (sizes, feature names, action heads) |
| `metrics.jsonl` | One line per update: losses, entropy, KL, clip fraction, grad norms, win rate vs the rule AI, curriculum stage, decisions/s, sim timing (`sim_ms_mean`, and `obs_ms_mean` for building observations), behaviour metrics |
| `episodes.jsonl` | One line per battle: stage, learner side, opponent, result, and `learner_*` / `opponent_*` behaviour metrics |
| `checkpoints/latest.pt` | Everything `--resume` needs. A numbered copy is kept every 250 updates. |
| `snapshots/actor_NNNNNN.pt` | Frozen past policies for the self-play league |
| `unity_logs/unity_<port>.log` | One Unity player log per worker |
| `evals/summary.jsonl` | Written by `watch_checkpoints.py` |
| `tb/` | TensorBoard logs, if TensorBoard is installed |

---

## Results so far

All evaluations use **sampled** actions. Early policies sit near 50/50 on fire vs hold, so always
taking the likelier option (greedy) can mean never firing.

> **Observation layouts.** The "first layout" sections below were measured before ships could
> sense terrain, obstacles, torpedoes, ports and the full state of every consumable, and before the
> move head gained "take cover" and "return to port". Policies from that layout cannot be loaded by
> the current game (`RLPolicyDriver` refuses them) or resumed by `train.py` (it lists the
> differences), so they have to be retrained.

### Current layout — behaviour cloning (`bc_terrain`)

**Training data:** 120 Elite-vs-Elite battles over stages 1–4 — which now include random
battlefields, modes and fleet sizes — about 863k decisions, 80% usable labels. The rule AI takes
cover in 3.3% of its movement labels and heads home to port in 10.1%, so both new moves are learned.

**Held-out accuracy per head:**

| move | speed | target | fire | torpedo | ability |
|---|---|---|---|---|---|
| 0.70 | 0.94 | 0.93 | 0.99 | 1.00 | 0.98 |

**Against the Elite rule AI** (48 battles each, sampled actions):

| Battlefields | Win rate | Damage dealt | Damage taken |
|---|---|---|---|
| Stage 3: random preset, island density and weather | 0.38 ± 0.07 | 4.57 | 3.60 |

It out-damages the rule AI but loses mostly on objective points (every battle ended on points or the
clock, none by a fleet being sunk), which is what MAPPO fine-tuning is for. The main line
`mappo_terrain` starts from it; its actor is frozen for the first 10 updates while the critic warms
up, so there is no fine-tuning result yet.

### First layout — full game, behaviour cloning

6v6 domination against the **Elite** rule AI, 48 battles each:

| Player | Win rate | Damage dealt | Damage taken |
|---|---|---|---|
| Random legal actions | 0.23 ± 0.06 | 4.77 | 3.58 |
| Behaviour cloning v1 (movement labels from steering only) | 0.71 ± 0.07 | 5.27 | 3.45 |
| Behaviour cloning v2 (zone orders labelled from the rule AI's assignments) | 0.60 ± 0.07 | 4.95 | 4.65 |

**Training data:** 120 Elite-vs-Elite battles (stages 1–4), about 890k decisions, 80% usable labels.

**Held-out accuracy per head:**

| move | speed | target | fire | torpedo | ability |
|---|---|---|---|---|---|
| 0.65 | 0.96 | 0.93 | 1.00 | 1.00 | 0.98 |

**v1 vs v2:**

- **v1** almost never chose "go to zone": the rule AI reaches zones by steering, so its labels said
  "move". In a watched game it won the fighting but lost on points.
- **v2** labels zone orders from the fleet commander's assignments (72% of its movement labels in
  domination), so it copies the rule AI's objective play more faithfully.
- The two win rates are within each other's error bars.

**MAPPO fine-tuning** starts from v2 (`runs/full_mappo`, evaluated every 25 updates into
`runs/full_mappo/evals/summary.jsonl`). It has logged 9 updates so far and is still inside the
10-update actor freeze, so there is no fine-tuning result yet.

### First layout — stage 0 battleship duel, learned from scratch

1v1 battleships against the **Recruit** AI, 96 battles each:

| Player | Win rate | Damage dealt | Damage taken |
|---|---|---|---|
| Random legal actions | 0.30 ± 0.05 | 0.56 | 0.99 |
| Trained, 66 updates (~540k decisions, about 25 minutes) | 0.42 ± 0.05 | 0.65 | 0.91 |
| Scripted tactic (auto target, fire at will, broadside) | 0.55 ± 0.07 | 0.87 | 0.69 |

The settings that produced this run — rollout 1024, 4 minibatches, 4 epochs, actor learning rate
2e-4, gamma 0.99, win/loss ±1 — are now the defaults. The earlier ones (256-step rollout, gamma
0.995, ±3 win bonus) left the duel too noisy to learn from.

### First layout — experiments that did not learn

Kept for the record (96 battles each, sampled actions):

| Run | Change | Result |
|---|---|---|
| `stage0_v3`, resumed at update 66 | Ammo cooldown added, and the entropy schedule accidentally reset | Update 100: 0.30 ± 0.05 — back to random level |
| `stage0_v4`, from scratch | Decision every 2 s, gamma 0.995, lambda 0.98 | Update 25: 0.28 ± 0.05; update 50: 0.23 ± 0.04 |

Learning even the duel from random play is slow and fragile, which is why the full-game recipe
imitates the rule-based AI first and only then fine-tunes with MAPPO.
