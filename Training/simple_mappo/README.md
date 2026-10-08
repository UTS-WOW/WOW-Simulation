# MAPPO with a fleet commander — the version to learn from

A readable multi-agent RL method that trains a fleet in the Unity naval simulation, through an
eight-stage curriculum from sinking a target to random fleet battles. It combines what the
game-playing AIs that beat professionals did — **OpenAI Five** (Dota 2), **AlphaStar** (StarCraft II),
**Honor of Kings** — with MAPPO, and it is laid out like the course's Week 9 and 10 notebooks
(config `OrderedDict`, `Monitor`, `SaveOnIntervalCallback`, `model.learn()`, learning curve,
beginning/middle/end replays). It trains on your machine or on Amazon SageMaker, and keeps the
team's checkpoints in one shared S3 folder so teammates continue each other's training.

**Start with the notebooks: [`StageP-Imitation.ipynb`](../StageP-Imitation.ipynb), then
[`Stage0-Gunnery.ipynb`](../Stage0-Gunnery.ipynb) … [`Stage7-Open.ipynb`](../Stage7-Open.ipynb).**

The original research pipeline in `naval_rl/` is unchanged; this package only borrows its TCP
connection to Unity.

---

## What the agents learn

Every ship of the Player fleet is an agent; a fleet commander gives them orders. The enemy is the
game's rule-based AI, a passive target (the first stages), or a copy of our own fleet (the league).
The objective is the game's own:

- **get to the centre of the capture circle and hold it** (it scores points every second),
- **sink enemy ships**,
- **do not lose your own**,
- and win the battle (first to 1000 points, or sink the whole enemy fleet; level at the clock goes to
  overtime, so there are no draws).

## Why this method (and not plain IPPO or MAPPO)

Every AI that beat human professionals at a team strategy game used the same family of method: an
**actor-critic policy gradient (PPO or close to it), self-play, memory, attention over units and
action masks** — OpenAI Five, AlphaStar, Tencent's Honor of Kings AI, and TiZero for 11-a-side
football. None of them used value-decomposition methods such as QMIX (built for purely cooperative
tasks, and awkward with six action heads and a changing opponent). Newer sequence models (MAT,
Sable) top cooperative benchmarks, but need a fixed order of agents and decode the team's actions
one by one — hard to explain and a poor fit for fleets of 1 to 8 ships against a learning opponent.

So the base stays MAPPO (PPO, a shared actor, a centralised critic), and the improvements are the
pieces those systems relied on:

| Piece | What it does | Paper | Course link |
|---|---|---|---|
| **Entity encoder** | every ship, contact, circle and island is a token; a small transformer lets them attend to each other; mean + max pooling | AlphaStar (transformer), OpenAI Five (max-pool) | W10-C custom CNN feature extractor — for a set instead of a picture |
| **GRU memory** | each ship remembers what it has seen (fog of war: an enemy out of sight is still somewhere) | OpenAI Five (LSTM), recurrent MAPPO | W10-C frame stacking, learned |
| **Pointer heads** | *which target* and *which circle* are scored against the contact / circle tokens, so 2 or 8 enemies use the same layer | AlphaStar pointer network, Honor of Kings target attention | W9-A softmax classifier over things present |
| **Fleet commander** | every 10 s an order per ship: *free*, *engage* or *hold circle k*; the captain sees its order, and the circle reward follows it | Honor of Kings' Hierarchical Macro Strategy (macro decides where, micro decides how) | PPO at a slower clock |
| **Centralised critic, multi-head value** | sees the true battle (training only); predicts the objective, combat and outcome rewards separately, plus the chance of winning | MAPPO, Honor of Kings | W10-A/B actor-critic |
| **Dual-clip PPO**, value normalisation + clipping | a very stale sample cannot blow up an update | Honor of Kings, MAPPO | W10-B PPO |
| **Team spirit, zero-sum reward** | ship rewards blend into the fleet's average as training goes on; the enemy's combat reward is subtracted | OpenAI Five | W10-B reward normalisation |
| **Imitation warm start + KL anchor** | the captains first copy the Elite rule AI; a fading KL penalty keeps them close at first | AlphaStar | **W9-A**: the same supervised loop |
| **League self-play** | the enemy is the rule AI, the latest frozen copy, or a past copy chosen by PFSP; Elo ratings | OpenAI Five, AlphaStar | W9-C/D target network (a frozen copy) |
| **Curriculum** | 8 stages, each passed at ≥ 70 % of its own goal; earlier stages replayed 10–15 % of the time | TiZero, OpenAI Five | — |
| **Best-model checkpoint** | `model_best.pt` keeps the best rolling pass rate | — | W9/10 `SaveOnIntervalCallback` |

Each piece is a switch, so the report can show what it adds (`train_simple.py --no-commander`,
`--no-memory`, `--attn-layers 0`, `--ippo`, `--imitation-battles 0`, `--dual-clip 0`).

References: Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*
(NeurIPS 2022); OpenAI, *Dota 2 with Large Scale Deep Reinforcement Learning* (2019); Vinyals et al.,
*Grandmaster level in StarCraft II using multi-agent reinforcement learning* (Nature 2019); Ye et al.,
*Towards Playing Full MOBA Games with Deep Reinforcement Learning* (NeurIPS 2020) and *Mastering
Complex Control in MOBA Games with Deep RL* (AAAI 2020); Wu et al., *Hierarchical Macro Strategy Model
for MOBA Game AI* (AAAI 2019); Lin et al., *TiZero* (AAMAS 2023); Wen et al., *Multi-Agent RL is a
Sequence Modeling Problem* (MAT, NeurIPS 2022); Mahjoub et al., *Sable* (ICML 2025).

## Files

| File | What it does |
|---|---|
| [`rewards.py`](rewards.py) | **The reward.** Every term and its weight in `REWARD_WEIGHTS`, its group, team spirit, zero-sum, order-aware circle pay |
| [`curriculum.py`](curriculum.py) | **The stages** (`DEFAULT_CURRICULUM`), random battle generation, `CurriculumCallback` (promotion, stopping rule) |
| [`env.py`](env.py) | `NavalEnv`: `n_envs` Unity battles behind `reset()` / `step()`; token observations; passive and league opponents; the imitation recorder |
| [`networks.py`](networks.py) | `EntityEncoder`, `Captain`, `Commander`, `CentralCritic`, `FleetPolicy` |
| [`mappo.py`](mappo.py) | Rollouts, GAE (per reward group, and for the commander's longer steps), the PPO update; `learn` / `save` / `load` / `predict` |
| [`league.py`](league.py) | `OpponentPool`: past copies, PFSP, Elo |
| [`imitation.py`](imitation.py) | `collect_expert`, `behaviour_cloning` (the W9-A loop), `save_imitation`, `plot_imitation` |
| [`callbacks.py`](callbacks.py) | `BaseCallback`, `SaveOnIntervalCallback`, `BestModelCallback` |
| [`utils.py`](utils.py) | `Monitor`, plots, `evaluate`, battle replays |
| [`team.py`](team.py) | `TeamStorage`: the shared S3 folder per run, with a lock and live status; `TeamSyncCallback` |
| [`aws.py`](aws.py) | SageMaker with boto3 only: the account's bucket, launching and watching training jobs |
| [`stages.py`](stages.py) | `continue_training`: the model a stage notebook trains (and its imitation start) |
| [`../train_simple.py`](../train_simple.py) | Command-line training (imitation first for a new run); the SageMaker training-job entry point |
| [`../watch_simple.py`](../watch_simple.py) | A trained model in the real game's graphics |
| `../StageP-Imitation.ipynb`, `../Stage0-Gunnery.ipynb` … `../Stage7-Open.ipynb` | One notebook per step, all training the same model in turn |
| [`../make_stage_notebooks.py`](../make_stage_notebooks.py) | Writes those notebooks; edit it, not the notebooks |
| [`../tests/test_simple_mappo.py`](../tests/test_simple_mappo.py) | Tests on the mock environment (no Unity): `python -m pytest tests` |

## How one update works

```
         ┌──────────── n_envs Unity battles (headless, ~400x real time each) ────────────┐
         └────▲─────────┬──────────────────────────────────────────────────▲─────────┬─┘
   actions,   │         │ obs: each ship's view, the true battle (critic),   │         │
   orders     │         ▼ the fleet's view (commander)                       │         ▼
         ┌────┴────────────────────────────────────────────────────────────────────────┐
 collect │ commander (every 10 s): fleet view -> an order per ship                       │ x n_steps
         │ captain (every ship, every 1 s): own view + order + GRU memory -> 6 heads      │
         │ critic: true battle -> value per ship and reward group, commander value, P(win)│
         │ rewards.py -> reward per ship and group (team spirit, zero-sum, orders)        │
         └──────────────────────────────────┬─────────────────────────────────────────┘
                                            ▼
 GAE        per ship and reward group (summed for the captains); per order for the commander
                                            ▼
 update     n_epochs x n_minibatches:
              critic     value loss (normalised, clipped) + commander value + win probability
              captains   dual-clip PPO on chunks of 32 decisions (the GRU replays them)
                         - entropy bonus + KL to the imitation policy (fading)
              commander  dual-clip PPO on its orders (one joint decision per fleet)
```

## How this is still MAPPO

MAPPO (Yu et al. 2022) is PPO for a cooperative team, built on **centralised training with
decentralised execution**: one actor shared by every ship, which acts only on what it can see,
and a critic that sees the true state of the battle during training. The paper's recommendations
are all here — value normalisation (`ValueNorm`), agent-specific global state (the critic reads each
ship's own token next to the whole battle), a few minibatches with several epochs, PPO clip 0.2,
death masking (a sunk ship stops training the actor; its critic keeps learning the team's value).

The commander adds one level above the ships. It only uses what the team could know (own ships,
the contacts it has spotted, the circles, the score), so the trained fleet still plays fair.
`MAPPO(env, centralised_critic=False, commander=False)` (`--ippo --no-commander`) is IPPO, the
baseline the paper compares MAPPO against.

## One method for every stage

Every stage — and any stage you add later — is trained by the **same model**: the same networks,
the same hyperparameters and the same reward function. The observation always has room for 8 ships
a side and 5 circles (`MAX_SHIPS` / `MAX_ZONES` in `env.py`); a duel simply leaves seven ship slots
empty, and the attention masks them out. Promotion only changes which battles are played.

## The observation, actions and reward

**Observation** (see the top of `env.py`). Each ship's view is a set of tokens: itself, its allies,
the enemy contacts its team has spotted (last known positions), the capture circles and the nearest
islands. The critic gets the true battle; the commander gets the same with the enemy's hidden
state zeroed. Unity sends the name of every feature (`env.spec.features`).

**Actions.** Six heads, one option each per decision; illegal options are masked:
move (incl. **sail to circle k**), speed, target, fire, torpedo, ability. The commander's orders:
free / engage / hold circle k, for every ship.

**Reward** (`rewards.py`):

| Term | Weight | Kind | Group |
|---|---|---|---|
| `in_circle` | +0.05 per second inside its circle | ship | objective |
| `approach_circle` | +1 per km closer to its circle's centre (− when moving away) | ship | objective |
| `zone_captured` / `zone_lost` | +1 / −1 | team | objective |
| `survive` | per second afloat (0; 0.02 in the Defend stage) | ship | objective |
| `enemy_sunk` / `ship_lost` | +1 / −1 per ship | team | combat |
| `damage_dealt` / `damage_taken` | +0.5 / −0.25 per whole hull | ship | combat |
| `friendly_fire` | −1 per whole hull | ship | combat |
| `zero_sum` | − the enemy ships' average damage terms | team | combat |
| `win` / `loss` | +5 / −5 | team | outcome |

"Its circle" is the commander's: a ship ordered to hold circle k is paid for circle k, a ship
ordered to engage gets no circle pay, a free ship is paid for the nearest circle.
`approach_circle` is *potential-based* shaping (the change in distance), so a ship cannot farm it by
sailing back and forth.

## The curriculum

`curriculum.py`, `DEFAULT_CURRICULUM`. Every stage is passed at **≥ 70 %** of its own goal:

| Stage | Battle | Fleet | Opponent | Passed when | Window |
|---|---|---|---|---|---|
| P – Imitation | Elite AI vs Elite AI, stages 2–6 | — | — | (supervised: copy the Elite AI) | — |
| 0 – Gunnery | `gunnery`, 7 km, open sea | 1 BB vs 1 CA | passive target (stop, hold fire) | target sunk within 150 s | 50 |
| 1 – Capture | `capture`, one neutral circle | 1 CA | passive DD, far away | circle captured | 50 |
| 2 – Defend | `defend`, neutral circle, enemy coming | 1 CA vs 1 CA | Recruit | won | 100 |
| 3 – Duel | `bb_duel`, open sea | 1v1 BB | Recruit | won | 100 |
| 4 – King of the Hill | `koth_3v3`, one circle | 3v3 mixed, **commander**, τ 0.3 | Veteran | won | 100 |
| 5 – Archipelago | `archipelago_3v3`, 3 circles, islands | 3v3 mixed, commander, τ 0.5 | Veteran | won | 150 |
| 6 – Domination | procedural 6v6, random map and weather | 6v6, commander, τ 0.8 | **league**: Elite 40 % / latest self 40 % / past selves 20 % | won vs Elite | 200 |
| 7 – Open | procedural 4–8 a side, random mode | commander, τ 1.0 | league | final: won vs Elite on 250 **held-out** maps | 250 |

- **Why the first stages are not judged on "won"**: a passive target never scores, so the game's
  tiebreak hands the win to anyone — random play "won" 100 % of them. Measured (32 battles each):
  gunnery, sunk within 150 s — random 52 %, "close in, fire, torpedoes" 76 %; capture — random 0 %,
  "sail to circle A" 100 %; defend — random 12 %, "sail to the circle" 0 %.
- **Qualifying battles** are battles that started on the current stage against the stage's own
  opponent (rule AI or passive target). Self-play and rehearsal battles never count, and the window
  starts empty on every new stage.
- **Rehearsal**: from stage 3 on, 10–15 % of battles replay an earlier stage so its skills stay.
- **Held-out evaluation** (stage 7): every 100,000 timesteps, 50 battles against the Elite AI on
  generated maps whose seeds training never uses. Training stops once the last 250 reach 70 %.
- **Any number of stages**: a stage is a dict (see the docstring in `curriculum.py`).
- A saved model remembers its stage, so `MAPPO.load` (and the team checkpoints) carry on where the
  curriculum was.

### One notebook per step

`StageP-Imitation.ipynb` and `Stage0-Gunnery.ipynb` … `Stage7-Open.ipynb` are the same method end to
end and share one run folder (`RUN_NAME = "fleet"`). There is only ever **one model**:

1. Run `StageP` once: it records the Elite AI, clones it, and saves `models/imitation.pt`.
2. `Stage0` creates the model from that checkpoint. After that, every run of a stage notebook
   continues the run's latest checkpoint — from the team folder, or from `runs_simple/fleet`.
3. A session trains for `n_timesteps`, saves, and pushes. If the stage is not passed yet, run the
   notebook again (you or a teammate): the promotion window carries on where it stopped.
4. When the stage's requirement is met, training stops, saves the model (now marked with the next
   stage), and tells you which notebook to open next. The next notebook loads it with **all its
   training memory**: the networks, the optimisers, the value normalisers, the league, the counters,
   the stage, and every battle so far in `logs/monitor.csv`.

Opening the wrong notebook stops with a message naming the right one; `FORCE_STAGE = True` jumps
ahead or goes back. To change the notebooks, edit `make_stage_notebooks.py` and run it.

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
    videos/battle_<timesteps>_stage<k>.mp4     replays recorded while training
```

**Status while training.** Every 10,000 timesteps the trainer pushes a quick status update (logs,
new videos, and live figures in `run.json`: stage, progress, speed, time left, recent win rate), and
every `save_interval` a checkpoint too. Anyone can watch, also a training job:

```python
storage = TeamStorage("s3://<bucket>/wow-mappo", "fleet", user="me")
storage.show_status()                     # state, stage, progress / ETA, win rate, newest video
storage.pull_view("team_view")            # logs + newest videos, to plot and play in the notebook
```

**Videos.** `VideoCallback` records one battle as a top-down replay every so often while training
(MP4 where ffmpeg is installed, GIF otherwise); `record_battle` and `view` do it on demand.

## It is the real game — and how to see it

Training runs the real Unity game: `Builds/NavalTrainer` is a build of this project, with the same
ships, gunnery, detection, PhysX physics and rule-based AI, and Python only sends each ship's orders
each second. It runs **without graphics** (`-batchmode -nographics`, and SDL's dummy video driver,
because the Unity 6 player otherwise crashes on machines with no window system), so the training
replays are a 2D map drawn from the game's data. To see a trained model in the game's own graphics,
on a computer with a screen:

```bash
python watch_simple.py --model model_final.pt            # a video rendered by Unity (fog lifted)
python watch_simple.py --model model_final.pt --live     # play it in the editor (tick GameBootstrap > Rl Training Server, press Play)
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
python train_simple.py --team-storage s3://<bucket>/wow-mappo --run-name fleet --user Lukita
```

In the notebook, set `YOUR_NAME`, `RUN_NAME` and `TEAM_STORAGE` in the "Team checkpoints" cell.
A plain folder path also works in place of `s3://` (a shared drive, or for testing).

## Running it

### On your computer

```bash
cd Training
jupyter lab WOW-MAPPO-Simple.ipynb                       # the walkthrough
python train_simple.py --total-timesteps 5000000         # imitation, then the curriculum, headless
python train_simple.py --scenario scenarios/stage1_koth_3v3.json --run-name koth   # one battle only
```

Needs Python 3.10+, PyTorch, NumPy, pandas, matplotlib, Pillow, and the headless player
(`Training/build_player.sh` builds it into `Builds/NavalTrainer/`).

### On Amazon SageMaker

1. Locally: `Training/build_player.sh`, then `Training/package_sagemaker.sh` → `wow_sagemaker.zip`
   (≈40 MB: code, scenarios, the Linux player).
2. Upload the zip to SageMaker Studio, open `Training/WOW-MAPPO-Simple.ipynb` (the whole
   walkthrough) or `Training/StageP-Imitation.ipynb`, then `Training/Stage0-Gunnery.ipynb` (one stage at a time) from it, run the
   first cell (it unzips to `~/wow`), and fill in the "Team checkpoints" cell.
3. Use a JupyterLab space with many CPUs and memory (e.g. `ml.c5.4xlarge`: 16 vCPU, 32 GB). No GPU
   is needed. The default `ml.t3.medium` (2 vCPU, 4 GB) can only run one battle and is too small for
   the 8-ship stages.
4. The whole curriculum takes millions of timesteps. Train it in the notebook in sessions, taking
   turns through the team folder. If your account allows SageMaker training jobs (course accounts
   often do not - an "explicit deny" error), the notebook's last section starts
   a **SageMaker training job** (`aws.launch_training_job`: `train_simple.py` in AWS's PyTorch CPU
   container, the player as its `unity` input) that trains the same team run — it pulls the latest
   checkpoint and pushes status, videos and checkpoints while it runs. `job_status(job)` shows the
   job's state and newest log lines; its console page charts the win rate and reward.

All of this uses boto3 only. SageMaker Distribution 4.x ships the SageMaker Python SDK v3, where
`sagemaker.Session`, `sagemaker.get_execution_role` and the `PyTorch` estimator no longer exist as
before; nothing here depends on them.

**Requirements the code checks for you:**

- **glibc 2.35+ (Ubuntu 22.04 or newer).** The Unity 6 player needs it. Training jobs use AWS's
  `pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker` image; images older than PyTorch 2.4 are
  Ubuntu 20.04 and the player cannot start there. The code stops with this explanation if the system
  is too old.
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
the normal step within their error bars, and larger steps buy almost nothing more.

Training speed of the commander method, 12 parallel battles on a 16-core desktop (CPU only):
about **265 timesteps per second** on the one-ship stages and **150** on 3v3. Two settings matter:

- `torch_threads=4` (the default): PyTorch's own threads otherwise fight the 12 Unity players for
  the cores, which cost 2-3x.
- Padding is trimmed before the networks (`networks.trim`): a 1v1 battle does not pay for the 8
  ship slots, 8 contacts and 5 circles the observation has room for. Pointer options whose token is
  empty are never chosen (Unity's action masks agree exactly - checked on 100k recorded decisions).

Most of the remaining time is the Unity simulation itself (~25-40 ms per step for 12 battles).

## Results

The sections below were measured with the **previous version** of this package (flat MLP MAPPO,
no memory, no commander, no imitation, the 5-stage curriculum). They are kept because they explain
choices the new method still uses (the circle reward weights, minibatching, value normalisation).
Results of the new method are added in "Results of the commander method" below.

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
