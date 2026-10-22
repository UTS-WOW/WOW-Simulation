"""Writes the stage notebooks - StageP (imitation) and one per curriculum stage. Every stage notebook trains
the run's MAPPO model AND its PQN model on that stage, and compares the two.

    python make_stage_notebooks.py

Edit the stage descriptions here and rerun, rather than editing the notebooks by hand. The stages
themselves (scenarios, opponents, pass criteria, promotion rules) live in simple_mappo/curriculum.py.
"""

from __future__ import annotations

import glob
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
IMITATION_FILE = "StageP-Imitation.ipynb"

# the method - written into every notebook, identical everywhere
MAPPO_SETTINGS = """mappo_settings = dict(learning_rate=3e-4, critic_learning_rate=5e-4, n_steps=256, n_minibatches=4, n_epochs=5,
                      gamma=0.995, gae_lambda=0.95, clip_range=0.2, dual_clip=3.0, ent_coef=0.01,
                      d_model=64, attn_layers=2, hidden_size=128, recurrent=True,      # the networks
                      commander=True, commander_period=10,                             # the fleet commander
                      kl_bc_coef=0.1, opponent_refresh=10)                             # imitation anchor, league"""

PQN_SETTINGS = """pqn_settings = dict(learning_rate=2.5e-4, num_steps=128, epochs=4, minibatches=4, lam=0.85,
                    gamma=0.99, max_grad_norm=1.0,   # PQN's SMAX setting (Gallici et al. 2025): Q(lambda) on rollouts
                    mixer="vdn",                     # PQN-VDN: the fleet's Q is the sum of its ships' Q-values
                    exploration="softmax", temperature=0.1,   # Boltzmann: options by softmax(Q / 0.1);
                                                     #   "epsilon": epsilon-greedy (the paper's) - best option or random
                    exploration_initial_eps=1.0, exploration_final_eps=0.01, exploration_timesteps=300_000,
                    kl_bc_coef=0.5,                  # kickstarting: the pull towards the imitation clone, fading
                    d_model=64, attn_layers=2, hidden_size=128, recurrent=True,       # the same networks as MAPPO
                    commander=True, commander_period=10)                             # the same commander"""

STAGES = [
    dict(file="Stage0-Gunnery.ipynb", name="Gunnery", scenario="gunnery", fleet="1 battleship vs 1 cruiser",
         opponent="Passive target (stop, hold fire)", rule="target sunk within 150 s in ≥70% of the last 50 battles",
         session=200_000, script="close",
         learns="""The first lesson is the most basic one: **destroy a ship**. Our battleship starts 7 km from an enemy cruiser that neither moves nor shoots - Python flies it with "stop, hold fire". The circle reward is off; damage and the kill pay, and the **win pays only for a kill within 150 s** - at 150 s without one the battle ends as a loss (against a target that never scores, the game itself would call every battle won, however slow).

A battle is *passed* when the target is sunk within 150 seconds - just sinking it eventually is too easy (random play does it), so the stage asks for doing it quickly: picking the target, keeping the guns firing, closing in and using torpedoes.""",
         measured="Measured (32 battles each): random play sinks the target within 150 s in 52% of battles, the script \"close in, fire at will, torpedoes\" 76%."),
    dict(file="Stage1-Capture.ipynb", name="Capture", scenario="capture", fleet="1 cruiser",
         opponent="Passive destroyer, far away", rule="circle captured in ≥70% of the last 50 battles",
         session=250_000, script="circle",
         learns="""**Capture the circle.** One cruiser, one neutral capture circle 4.5 km ahead; the only enemy is a passive destroyer 16 km beyond it, out of sight. Points come only from holding the circle, so the cruiser has to sail into it and stay inside until it is captured. The win reward is paid only for a capture (the passive enemy never scores, so the game would call every battle won).""",
         measured="Measured: random play captures the circle in 0% of battles, the script \"sail to circle A\" in 100%."),
    dict(file="Stage2-Defend.ipynb", name="Defend", scenario="defend", fleet="1 cruiser vs 1 cruiser",
         opponent="Recruit rule-based AI", rule="won in ≥70% of the last 100 battles",
         session=300_000, script="circle",
         learns="""**Defend yourself while holding the circle.** A neutral circle lies 3 km ahead of our cruiser; an enemy cruiser flown by the rule AI comes for it from 8 km beyond. Winning means taking the circle and holding it *while staying afloat* - angling, dodging torpedoes, smoke, repair and fighting back. A per-second "survive" reward is switched on for this stage.""",
         measured="Measured: random play wins 12% of battles, sailing straight to the circle 0% - the cruiser has to fight smart."),
    dict(file="Stage3-Duel.ipynb", name="Duel", scenario="bb_duel", fleet="1v1 battleships",
         opponent="Recruit rule AI 60% · self-play 40%: itself 15%, its champion 10%, past selves 5%, the other method 10%", rule="won in ≥70% of the last 100 battles against the rule AI",
         session=400_000, script="broadside",
         learns="""**1v1.** Two battleships 13 km apart on open sea, both spotted and inside gun range from the first second. The duel is about **choosing and timing fire, angling the hull, and not throwing the ship away**. The circle reward is off here: the circle lies between the two battleships, so racing to it means sailing bow-on into the enemy's guns.

**Self-play starts here** (`league.py`) - 40% of battles are against the fleet itself, in four ways:

* **mirror** (15%): the fleet flies **both** battleships and **both sides learn** (OpenAI Five's self-play) - twice the training data, and an opponent exactly as good as the fleet;
* **its champion** (10%): the strongest version so far. It is replaced only when the fleet beats it in 55% of 40 battles (AlphaGo Zero's gating) - then the fleet becomes the new champion. So it always has its best self to beat;
* **past selves** (5%): older copies, the ones it loses to most often first (AlphaStar's PFSP);
* **the other method** (10%): MAPPO's fleet for the PQN run and the other way round - a differently trained opponent with different habits (cross-play, like AlphaStar's league of different agents).

Only the battles against the rule AI count for promotion.""",
         measured="Measured earlier with this battle: random play wins 16-28%, the broadside script 70%, flat MAPPO from scratch 59% after 300,000 timesteps."),
    dict(file="Stage4-King-of-the-Hill.ipynb", name="King of the Hill", scenario="koth_3v3", fleet="3v3 mixed fleet",
         opponent="Veteran rule AI 60% · self-play 40% (itself, champion, past selves, the other method)", rule="won in ≥70% of the last 100 battles against the rule AI", session=500_000, script="circle",
         learns="""**The first fleet battle - and the commander joins in.** A battleship, a cruiser and a destroyer a side, one capture circle in the middle of the open sea, 6 minutes. Every 10 seconds the **fleet commander** gives each ship an order - *free*, *engage* or *hold the circle* - and the captains carry it out (the circle reward follows the order). Team spirit 0.3: each ship's own reward is now partly shared with the fleet. 15% of battles replay earlier stages so the single-ship skills stay.""",
         measured="Measured earlier with this battle: random play wins about 18% against Veteran, flat MAPPO from scratch 25-33%, the circle script 67-88% (against Recruit), imitation alone (naval_rl) 58%."),
    dict(file="Stage5-Archipelago.ipynb", name="Archipelago", scenario="archipelago_3v3", fleet="3v3 mixed fleet",
         opponent="Veteran rule AI 60% · self-play 40% (itself, champion, past selves, the other method)", rule="won in ≥70% of the last 150 battles against the rule AI", session=750_000, script="circle",
         learns="""The same three ships a side, now among islands with **three capture circles** (A, B, C) and 8 minutes. This is where the commander earns its place: **which ship holds which circle, and who hunts**. Islands break line of sight, so spotting, smoke, radar and cover matter. Team spirit 0.5.""",
         measured="Measured earlier: imitation alone (naval_rl) won 45%, flat MAPPO 27%."),
    dict(file="Stage6-Domination.ipynb", name="Domination", scenario="domination_6v6", fleet="6v6 fleet",
         opponent="League: Elite rule AI 40% · itself 25% · its champion 10% · past selves 15% · the other method 10%",
         rule="won in ≥70% of the last 200 battles against the Elite AI", session=1_000_000, script="circle",
         learns="""The full Domination game at 6v6, 15 minutes, on a **newly generated battlefield** every 8 battles (random preset, island density and weather). The enemy is drawn from a **league** (OpenAI Five / AlphaGo Zero / AlphaStar): the Elite rule AI 40%, the fleet **itself** 25% (a mirror battle - both sides learn), **its champion** 10% (the strongest version so far, replaced only when beaten in 55% of 40 battles), **past selves** 15% (the ones we lose to come up most) and **the other method's fleet** 10% (cross-play). Everyone gets an **Elo rating**. Only battles against the Elite AI count for promotion. Team spirit 0.8.""",
         measured=""),
    dict(file="Stage7-Open.ipynb", name="Open", scenario="open", fleet="Random 4–8 ships per side",
         opponent="League: Elite rule AI 40% · itself 25% · its champion 10% · past selves 15% · the other method 10%",
         rule="won in ≥70% of the last 200 battles against the Elite AI", session=2_000_000,
         script="circle",
         learns="""Anything the game can produce: 4 to 8 ships a side, a random game mode (Domination, Skirmish, Fleet Battle, Capture and Control), map preset, density, weather and circle size, 20 minutes - against the league. Team spirit 1.0: every ship cares about the fleet as much as about itself. Passing it moves the fleet on to the biggest battle, stage 8.""",
         measured="Measured: the SageMaker MAPPO run (2.2 M timesteps) won 83% of 24 held-out battles against the Elite AI here."),
    dict(file="Stage8-Archipelago-9v9.ipynb", name="Archipelago 9v9", scenario="archipelago_9v9",
         fleet="9v9, every class", opponent="League: Elite rule AI 40% · itself 25% · its champion 10% · past selves 15% · the other method 10%",
         rule="Final stage: won in ≥70% of the last 200 evaluation battles against the Elite AI", session=5_000_000,
         script="circle",
         learns="""**The biggest battle: nine ships a side, every class that fights, five capture circles.** Three circles lie across the middle (A, B, C), D up-left near the enemy, E down-right near us - the layout of the game's own *Capture and Control* mode - among hand-placed islands that are the same for both sides (turned 180°). 12 minutes; the fleet that holds more circles scores faster.

This stage is about **holding several circles, not crowding the centre one**. Two things push the fleet to spread out (`rewards.py`): a ship without an order is paid for the nearest circle that still *needs* a ship - one we do not own yet, shared out so 9 ships and 5 neutral circles become 2-2-2-2-1 - and the new `circles_held` term pays the whole fleet every second for holding more circles than the enemy. The commander can still order any ship to hold any circle.

There is **no further promotion**. Every 100,000 timesteps the fleet plays 40 evaluation battles against the Elite AI; training stops once the last 200 reach 70%.""",
         measured="Measured before any training on it (8 battles): the SageMaker MAPPO model, trained up to stage 7, won 3 of 8 against the Elite AI here. It out-fought the enemy - 6.0 enemy ships sunk per battle, 3.5 of its own lost - but lost on points (408 to 436): the enemy held more circles. That is what this stage teaches."),
]

CLASS_ROLES = """| Class | | What it does |
|---|---|---|
| **Battleship** | BB | the biggest guns and the thickest armour; slow, long range - wins duels, anchors the line |
| **Cruiser** | CA | medium guns that fire fast, radar / hydro / smoke support; the all-rounder |
| **Destroyer** | DD | the fastest, torpedoes and smoke, hardest to spot - the circle capper and scout |
| **Submarine** | SS | dives under water, torpedoes, very hard to spot or hit while submerged |"""

SETUP = r"""
import os, time, zipfile
# SageMaker Studio (/home/sagemaker-user) or a SageMaker notebook instance (/home/ec2-user/SageMaker)
ON_SAGEMAKER = any(os.path.isdir(p) for p in ("/home/sagemaker-user", "/home/ec2-user/SageMaker"))
home = os.path.expanduser("~/SageMaker") if os.path.isdir(os.path.expanduser("~/SageMaker")) else os.path.expanduser("~")
bundle = os.path.join(home, "wow_sagemaker.zip")
if ON_SAGEMAKER and os.path.basename(os.getcwd()) != "Training":
    if not os.path.isdir(os.path.join(home, "wow")):
        assert os.path.exists(bundle), f"upload wow_sagemaker.zip to {home} first"
        zipfile.ZipFile(bundle).extractall(os.path.join(home, "wow"))
    %cd {home}/wow/Training

# Uploading a new wow_sagemaker.zip does not replace the unzipped folder: check which version runs here
running = open("BUNDLE_VERSION").read().strip() if os.path.exists("BUNDLE_VERSION") else "an older one"
if ON_SAGEMAKER and os.path.exists(bundle):
    with zipfile.ZipFile(bundle) as z:
        uploaded = z.read("Training/BUNDLE_VERSION").decode().strip() if "Training/BUNDLE_VERSION" in z.namelist() else "unknown"
    if uploaded != running:
        print(f"*** {bundle} is version {uploaded}, but this notebook runs version {running}. Unzip it into a new folder "
              f"and open this notebook from there. ***")
print(os.getcwd(), "| bundle", running, "| glibc", os.confstr("CS_GNU_LIBC_VERSION"))
"""

SCRIPTS = {
    "close": '''class Script:
    """Close in on the target, fire at will, launch torpedoes: a simple hand-written attack."""
    def predict(self, obs, deterministic=False):
        E, N = obs["alive"].shape
        actions = np.zeros((E, N, len(env.head_sizes)), dtype=np.int32)    # speed full, target auto, fire at will
        actions[..., 0] = np.where(obs["action_mask"][..., 9] > 0.5, 9, 0)    # move option 9 = close in
        actions[..., 4] = 1                                                   # torpedoes at the target
        return actions
SCRIPT_NAME = "scripted: close in, fire at will"''',
    "broadside": '''class Script:
    """Broadside and fire at will: a simple hand-written duel tactic."""
    def predict(self, obs, deterministic=False):
        E, N = obs["alive"].shape
        actions = np.zeros((E, N, len(env.head_sizes)), dtype=np.int32)    # speed full, target auto, fire at will
        actions[..., 0] = np.where(obs["action_mask"][..., 10] > 0.5, 10, 0)   # move option 10 = broadside
        return actions
SCRIPT_NAME = "scripted: broadside, fire at will"''',
    "circle": '''class Script:
    """Every ship sails to the first capture circle at full speed and fires at will."""
    def predict(self, obs, deterministic=False):
        E, N = obs["alive"].shape
        actions = np.zeros((E, N, len(env.head_sizes)), dtype=np.int32)
        actions[..., 0] = np.where(obs["action_mask"][..., 15] > 0.5, 15, 0)   # move option 15 = sail to circle 0
        return actions
SCRIPT_NAME = "scripted: go to the circle, fire at will"''',
}


INSTALL = r"""
# install only what is missing (PyTorch as the CPU build)
import importlib.util, shutil
missing = [pkg for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("PIL", "pillow"))
           if importlib.util.find_spec(mod) is None]
if shutil.which("ffmpeg") is None and importlib.util.find_spec("imageio_ffmpeg") is None:
    missing.append("imageio-ffmpeg")      # an ffmpeg: MP4 replays to keep, and GIFs of a teammate's MP4s
if missing:
    !pip install -q {" ".join(missing)}
if importlib.util.find_spec("torch") is None:
    !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"""

TEAM = r"""
YOUR_NAME = "your-name"              # shown to your teammates
RUN_NAME = "fleet"                   # the MAPPO run - the same in every stage notebook (and StageP)
if ON_SAGEMAKER:
    from simple_mappo.aws import team_storage_uri
    TEAM_STORAGE = team_storage_uri()     # s3://sagemaker-<region>-<account>/wow-mappo, or "s3://<team-bucket>/wow-mappo"
else:
    TEAM_STORAGE = None              # e.g. "s3://<team-bucket>/wow-mappo"; None trains on this machine only

storage = TeamStorage(TEAM_STORAGE, RUN_NAME, user=YOUR_NAME) if TEAM_STORAGE else None
run_dir = f"runs_simple/{RUN_NAME}"
log_dir, models_dir, video_dir = (os.path.join(run_dir, d) for d in ("logs", "models", "videos"))
os.makedirs(models_dir, exist_ok=True)
if storage:
    storage.show_status()
"""

TEAM_PQN = r"""
PQN_RUN_NAME = RUN_NAME + "_pqn"     # the PQN run: its own checkpoints, logs and curriculum stage
pqn_storage = TeamStorage(TEAM_STORAGE, PQN_RUN_NAME, user=YOUR_NAME) if TEAM_STORAGE else None
pqn_run_dir = f"runs_simple/{PQN_RUN_NAME}"
pqn_log_dir, pqn_models_dir, pqn_video_dir = (os.path.join(pqn_run_dir, d) for d in ("logs", "models", "videos"))
os.makedirs(pqn_models_dir, exist_ok=True)
if pqn_storage:
    pqn_storage.show_status()
"""

METHOD_MD = """
### Two methods, the same fleet

Every stage notebook trains **two** runs on the same battles - one per method - and compares them:

| | **MAPPO** (`simple_mappo/mappo.py`) | **PQN** (`simple_mappo/pqn.py`) |
|---|---|---|
| Family | policy gradient (PPO, Week 10) | value-based Q-learning - the 2025 successor of DQN (Week 9): **PQN**, the Parallelised Q-Network (Gallici et al., ICLR 2025), which beat QMIX and MAPPO on SMAX (StarCraft multi-agent) |
| What the captain's head scores are | logits: sample an option | Q-values (dueling: V + A): sample an option by softmax(Q / 0.1) - Boltzmann exploration (`exploration="epsilon"`: the best option, epsilon-greedy) |
| Learns from | the battles it just played (on-policy) | **also the battles it just played**: a rollout of 128 decisions from every battle, 4 epochs of minibatches, then thrown away - no replay buffer, so self-play opponents never go stale |
| Target | GAE advantages from a centralised critic | **Q(λ) returns**, λ = 0.85: R_t = r_t + γ[λR_t+1 + (1−λ) max Q(s_t+1)], from the network as it was at the rollout; no separate target network - LayerNorm in the Q-network keeps it stable |
| Six action heads | one probability per head | one Q per head, every head trained to the same target (branching DQN, BDQ) |
| Fleet commander | PPO at a 10 s clock | Q-learning on (ship, order) at a 10 s clock (h-DQN, intra-option) |
| Many ships | shared weights, centralised critic | shared weights; **PQN-VDN**: the fleet's Q is the sum of its ships' Q-values, trained on the fleet's reward |
| Shared | the captain network (transformer + GRU + pointer heads), the commander, the rewards, the curriculum, self-play, the imitation start | ← same |

Each run climbs the curriculum on its own (`runs_simple/fleet` and `runs_simple/fleet_pqn`), so a stage notebook only trains a run that is on its stage; the other one is still evaluated here, for the comparison. (An earlier Double DQN run, `fleet_ddqn`, is kept for the record: `simple_mappo/ddqn.py`.)

### MAPPO in one picture

| Piece | What it does | Where it comes from | Course link |
|---|---|---|---|
| **Captain** (every ship, one shared network) | its own view → 6 action heads; a transformer over ships/circles/islands, a GRU memory, pointer heads for *which target* and *which circle* | AlphaStar, OpenAI Five | W10-C custom feature extractor; frame stacking → memory |
| **Fleet commander** | every 10 s: an order per ship - free / engage / hold circle k | Honor of Kings' hierarchical macro strategy | the same PPO at a slower clock |
| **Centralised critic** | sees the true battle (training only); one value per reward group + win probability | MAPPO, Honor of Kings multi-head value | W10-A/B actor-critic |
| **Update** | PPO with dual clip, value normalisation and clipping | MAPPO, Honor of Kings | W10-B PPO |
| **Warm start** | the captain first copies the Elite rule AI (StageP), then a fading KL penalty keeps it close | AlphaStar | W9-A supervised training loop |
| **Self-play & league** | from stage 3: mirror battles (the fleet on **both** sides, **both** learning), its **champion** (replaced only when beaten in 55% of 40 battles), past copies (PFSP), and the **other method's fleet** (cross-play), with Elo ratings; the rule AI stays as the anchor promotion is judged against | OpenAI Five, AlphaGo Zero, AlphaStar | W9-C/D frozen target network |
| **Curriculum** | 8 stages, each passed at ≥70% of its own goal; earlier stages are replayed 10-15% of the time | TiZero, OpenAI Five | - |
"""


def nav(current: str) -> str:
    items = [f"**Imitation**" if current == "P" else f"[Imitation]({IMITATION_FILE})"]
    items += [f"**Stage {i}**" if str(i) == current else f"[Stage {i}]({s['file']})" for i, s in enumerate(STAGES)]
    items.append("**MAPPO vs PQN**" if current == "V" else "[MAPPO vs PQN](Versus-MAPPO-vs-PQN.ipynb)")
    return " · ".join(items) + " · [the full walkthrough](WOW-MAPPO-Simple.ipynb)"


def cap_md(k: int) -> str:
    """The stage's max_timesteps (curriculum.py), for the table at the top of its notebook."""
    from simple_mappo.curriculum import DEFAULT_CURRICULUM
    cap = DEFAULT_CURRICULUM[k].get("max_timesteps")
    if not cap:
        return ""
    if k == len(DEFAULT_CURRICULUM) - 1:
        return f" - or, not met, training ends after {cap:,} timesteps on it (its cap)"
    return f" - or, NOT passed, after {cap:,} timesteps on it (its cap), it moves on anyway"


def ships_md(k: int) -> str:
    """The ships of stage k by class, from the curriculum (and its scenario file) - always up to date."""
    from simple_mappo.curriculum import DEFAULT_CURRICULUM, SHIP_CLASSES, stage_fleets
    from simple_mappo.env import load_scenario
    f = stage_fleets(DEFAULT_CURRICULUM[k], load_scenario)
    rows = []
    for c in (2, 1, 0, 3):
        name, abbr = SHIP_CLASSES[c]
        rows.append(f"| {name.capitalize()} ({abbr}) | {f['ours'].get(c, 0)} | {f['theirs'].get(c, 0)} |")
    total = f"| **Total** | **{sum(f['ours'].values())}** | **{sum(f['theirs'].values())}** |"
    note = ""
    if "procedural" in DEFAULT_CURRICULUM[k]:
        note = f"\n\nA generated battle: {f['text']}. The table shows the largest fleet."
    elif DEFAULT_CURRICULUM[k].get("opponent") == "passive":
        note = "\n\nThe enemy is a passive target: Python orders it to stop and hold fire."
    return ("| Class | Our fleet | Enemy fleet |\n|---|---|---|\n" + "\n".join(rows) + "\n" + total + note)


def notebook(k: int) -> nbf.NotebookNode:
    st = STAGES[k]
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    last = k == len(STAGES) - 1
    league = k >= 3                  # self-play from stage 3: mirror, champion, past selves, the other method

    md(f"""
# 43008: Reinforcement Learning - WOW Naval Fleet: MAPPO vs PQN

## Stage {k}: {st['name']}

{nav(str(k))}

| Stage | Battle | Fleet | Opponent | {'Stopping' if last else 'Promotion'} requirement |
|---|---|---|---|---|
| {k} - {st['name']} | `{st['scenario']}` | {st['fleet']} | {st['opponent']} | {st['rule']}{cap_md(k)} |

### What the fleet learns here

{st['learns']}

{st['measured']}

### The ships

{ships_md(k)}

{CLASS_ROLES}

### One model per method, through every stage

The stage notebooks train **two runs** - the MAPPO fleet (`RUN_NAME`) and the PQN fleet (`PQN_RUN_NAME`, `RUN_NAME + "_pqn"`) - each through every stage. A session continues each run's latest checkpoint, which carries everything learned so far (networks, optimisers, the league, the timestep count, the stage) and its battle log, from which the promotion window continues where the last session stopped.

* **Training is continuous.** Not passed yet? **Run this notebook again** - each session continues the last one (yours or a teammate's), from `models/model_latest.pt`, refreshed every `save_interval`. `KEEP_GOING = True` carries on into the next stages in the same session instead of stopping at the promotion.
* `TRAIN_MAPPO` / `TRAIN_PQN` (configuration cell) switch either method's training on or off for this session. A run that is on another stage is not trained here, only evaluated for the comparison.
* **Train this stage again** - after a run passed it, reached its cap, or is on another stage: set `FORCE_STAGE = True` (and `TRAIN_MAPPO` / `TRAIN_PQN` to choose the method) and run the notebook. The run comes back to this stage and trains it with a fresh promotion window and cap, until it passes again, reaches the cap, or `n_timesteps` is done; then it is on this stage - or the next one, if it passed - and the notebooks carry on from there.
* {'This is the final stage: training ends when the stopping rule is met.' if last else f"Passed? That run stops and saves, and moves to stage {k + 1}: open [Stage {k + 1}]({STAGES[k + 1]['file']})."}
{'* No model yet? Both start from the imitation checkpoint of the [Imitation notebook](' + IMITATION_FILE + ') - run that one first.' if k == 0 else ''}
""" + METHOD_MD)

    md("## Setup\n\nOn SageMaker, run this from the unzipped bundle (`<folder>/Training/`). On your computer, open it from the repository's `Training/` folder. Either way the battles are the **real Unity game** (the headless player in `Builds/NavalTrainer`, built from this project's `Assets/`), not a simplified copy.")
    code(SETUP)
    code(INSTALL)
    code(r"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from simple_mappo import (NavalEnv, MAPPO, FleetPQN, Monitor, CurriculumCallback, SaveOnIntervalCallback,
                          BestModelCallback, VideoCallback, TeamStorage, TeamSyncCallback, DEFAULT_CURRICULUM,
                          open_run, training_status, evaluate, plot_results, plot_curriculum, plot_reward_terms,
                          plot_progress, plot_comparison, compare_stages, show_videos, find_videos, record_battle)
from simple_mappo.utils import video_ext
""")

    md("## Team checkpoints\n\nUse the **same `RUN_NAME` in every notebook** (and as your teammates): that is the one line of training the stages hand over along - one per method. Only one person can train a run at a time.")
    code(TEAM)
    code(TEAM_PQN)

    md("""
## Configuration

Each method's settings are **identical in every notebook**. They only shape a brand-new model; a continued model keeps the settings it was created with. `n_timesteps` is this session's training, per method; rerun the notebook to continue.
""")
    code(f"""
STAGE = {k}
FORCE_STAGE = False                  # True: train this stage (again) even if a run passed it, reached its cap
                                     # or is on another stage - with a fresh promotion window and cap
KEEP_GOING = False                   # True: when this stage is passed, carry on into the next stages
                                     # in this same session instead of stopping (continuous training)
TRAIN_MAPPO = True                   # train the MAPPO run in this session
TRAIN_PQN = True                    # train the PQN run in this session

config = OrderedDict([
    ('n_envs', max(1, min(16, os.cpu_count() // 2))),     # battles in parallel, one Unity process each
    ('decision_period', 1.0),          # game seconds between decisions
    ('sim_dt', 0.08),                  # game seconds per simulated frame (fastest stable)
    ('n_timesteps', {st['session']:_}),          # this session's training, per method (rises stage by stage)
])
{MAPPO_SETTINGS}
{PQN_SETTINGS}
save_interval = min(config['n_timesteps'] // 3, 250_000)   # a checkpoint (and team push) at least every 250,000

pd.Series(DEFAULT_CURRICULUM[STAGE], name=f"stage {{STAGE}}").to_frame()
""")

    md("## Environment\n\nLaunches `n_envs` headless Unity players - the real game. Both methods fight on the same players, each with its own battle log (`Monitor`) and its own curriculum stage. The observation has the same size in every stage (room for 9 ships a side and 5 circles), which is what lets one network play them all.")
    code(r"""
battles = NavalEnv(stages="default", start_stage=STAGE, n_envs=config['n_envs'],
                   decision_period=config['decision_period'], sim_dt=config['sim_dt'],
                   unity_log_dir=os.path.join(run_dir, "unity_logs"))
env = Monitor(battles, log_dir)              # the MAPPO run's battles
env_pqn = Monitor(battles, pqn_log_dir)      # the PQN run's battles
print("battles in parallel:", battles.n_envs, "| token sizes", battles.dims)
""")

    md(f"""
## The two models

`open_run` loads each run's latest checkpoint with all its training memory{' - or, with no checkpoint yet, starts a new model from the imitation checkpoint (`models/imitation.pt`; the PQN one takes it as its Q-network, so its greedy play starts as the Elite AI)' if k == 0 else ''}. It says when a run is on another stage: that run is then only evaluated here.
""")
    code(r"""
imitation = os.path.join(models_dir, "imitation.pt")
model, mappo_here = open_run(env, STAGE, run_dir, log_dir, "mappo", storage=storage, force=FORCE_STAGE, **mappo_settings)
pqn, pqn_here = open_run(env_pqn, STAGE, pqn_run_dir, pqn_log_dir, "pqn", storage=pqn_storage, force=FORCE_STAGE,
                         imitation_path=imitation, **pqn_settings)
TRAIN_MAPPO, TRAIN_PQN = TRAIN_MAPPO and mappo_here, TRAIN_PQN and pqn_here
if storage and TRAIN_MAPPO:
    storage.lock()                   # stops here if a teammate is training this run right now
if pqn_storage and TRAIN_PQN:
    pqn_storage.lock()
for name, m, m_env, trained in (("MAPPO", model, env, TRAIN_MAPPO), ("PQN", pqn, env_pqn, TRAIN_PQN)):
    if trained:
        print(f"{name}: trains here, on stage {STAGE}")
    elif m is None:
        print(f"{name}: no model yet - start it in the Stage0 notebook")
    else:                            # each method climbs the curriculum on its own
        print(f"{name}: on stage {m_env.stage} - only evaluated here. To train it, open the Stage{m_env.stage} notebook "
              f"(or set FORCE_STAGE = True to train it on stage {STAGE}{' again' if m_env.stage > STAGE else ''})")
model.actor if model else None       # the captain (and the commander), as in the Week 10 notebooks
""")

    md("""
## Baselines

What the trained fleets have to beat on this stage: random legal actions, and a simple hand-written tactic. `pass_rate` is the stage's own goal (see the table at the top). Set `RUN_BASELINES = False` to skip them in later sessions.
""")
    code(SCRIPTS[st['script']] + r"""

RUN_BASELINES = True
baselines = {}
if RUN_BASELINES:
    baselines = {"random": evaluate(None, env, n_battles=24, stage=STAGE),
                 SCRIPT_NAME: evaluate(Script(), env, n_battles=24, stage=STAGE)}
    display(pd.DataFrame(baselines).T[["stage", "battles", "win_rate", "pass_rate", "mean_reward"]])
""")

    md(f"""
## Train MAPPO

{'Training stops by itself when the stopping rule is met' if last else 'Training stops by itself as soon as the promotion requirement is met (unless `KEEP_GOING`)'}{(', or when the stage cap in the table at the top is reached (training then ends)' if last else ', or when the stage cap in the table at the top is reached (the run then moves on, recorded as not passed)') if cap_md(k) else ''}, or after `n_timesteps`. `models/model_best.pt` always holds the checkpoint with the best rolling pass rate. The team folder gets a status update every 10,000 timesteps and a checkpoint every `save_interval`.

**Replays**: five times a session a battle is recorded and **shown right here, under the training table**, while training goes on (`runs_simple/<RUN_NAME>/videos/`: an MP4 to keep, and the GIF the notebook shows - an image plays in every notebook app, VS Code included; double-click it in the file browser). A replay shown while the cell runs is lost if the page is closed or reloaded meanwhile - **Replays from this stage** at the end shows every one again, from the files.

In the table: `win_rate` / `pass_rate` against the stage's own opponent{', `win_rate_self_play` in battles against itself (about 0.5 in mirror battles - both sides are the fleet), `elo` the fleet''s rating' if k >= 3 else ''}{', `orders_free` the share of the commander''s orders that leave the ship free' if k >= 4 else ''}.
""")
    code(r"""
def callbacks_for(run_model, run_dir, models_dir, video_dir, run_storage):
    curriculum = CurriculumCallback(stop_on_promotion=not KEEP_GOING, fresh=FORCE_STAGE)
    callbacks = [SaveOnIntervalCallback(save_interval, models_dir),     # also refreshes model_latest.pt
                 curriculum,
                 BestModelCallback(models_dir),
                 VideoCallback(every=config['n_timesteps'] // 5, video_dir=video_dir, show=True)]   # replays appear below
    if run_storage:
        callbacks.append(TeamSyncCallback(run_storage, run_dir, push_every=save_interval, status_every=10_000))
    return callbacks, curriculum

if TRAIN_MAPPO:
    model.set_rival(pqn)             # cross-play: the PQN fleet joins MAPPO's league as "rival"
    callbacks, curriculum = callbacks_for(model, run_dir, models_dir, video_dir, storage)
    model.learn(total_timesteps=model.num_timesteps + config['n_timesteps'], callback=callbacks, log_interval=5)
    model.save(os.path.join(models_dir, "model_final"))
    model.save(os.path.join(models_dir, "model_latest"))
""")

    md(f"""
## Train PQN

The same battles, the same rewards, the same stopping rule - learned with PQN: every battle plays `num_steps` (128) decisions, then the Q-network learns from that rollout - Q(λ) targets, `epochs` (4) passes in `minibatches` (4) minibatches - and the rollout is thrown away. No replay buffer to fill and no target network to copy: it trains from the first rollout, and needs about a fifth of the Double DQN's network work per timestep.

Cross-play: before each method trains, the other one's fleet - its champion, if it has one - joins its league as the **rival** (`set_rival`), so each learns to beat a differently trained fleet too. A *new champion* message means the fleet just beat its best self.

In the table: `q_loss` the squared TD error against the Q(λ) targets, `q_value` / `target_value` the Q-values and their targets (both should rise as the fleet starts winning), `epsilon` the commander's exploration rate, `replay_decisions` the size of the last rollout.
""")
    code(r"""
if TRAIN_PQN:
    pqn.set_rival(model)             # ... and the MAPPO fleet (just trained) joins PQN's
    callbacks, pqn_curriculum = callbacks_for(pqn, pqn_run_dir, pqn_models_dir, pqn_video_dir, pqn_storage)
    pqn.learn(total_timesteps=pqn.num_timesteps + config['n_timesteps'], callback=callbacks, log_interval=8)
    pqn.save(os.path.join(pqn_models_dir, "model_final"))
    pqn.save(os.path.join(pqn_models_dir, "model_latest"))
""")

    md("""
## Save and hand over

Saves both runs (`models/model_latest.pt`, where the next session continues) and pushes them to the team. **If you stop training early (Kernel > Interrupt), run this cell** - it saves what the interrupted session learned, so nothing is lost, and frees the runs for the next person.
""")
    code(f"""
for name, m, m_env, m_storage, m_run_dir, trained in (("MAPPO", model, env, storage, run_dir, TRAIN_MAPPO),
                                                      ("PQN", pqn, env_pqn, pqn_storage, pqn_run_dir, TRAIN_PQN)):
    if m is None or not trained:
        continue
    m.save(os.path.join(m_run_dir, "models", "model_latest"))     # also after Kernel > Interrupt
    if m_storage:
        m_storage.push(m_run_dir, m, status=training_status(m, m_run_dir))
        m_storage.release()
    if m_env.stage > STAGE:
        unpassed = (curriculum if name == "MAPPO" else pqn_curriculum).moved_on_unpassed
        print(f"{{name}}: stage {{STAGE}} {{'NOT passed - its timestep cap was reached' if unpassed else 'passed'}} at "
              f"{{m.num_timesteps:,}} timesteps - now on '{{m_env.stages[m_env.stage]['name']}}': open the Stage{{m_env.stage}} notebook.")
    elif STAGE == len(m_env.stages) - 1 and (curriculum if name == "MAPPO" else pqn_curriculum).finished:
        capped = (curriculum if name == "MAPPO" else pqn_curriculum).capped_out
        print(f"{{name}}: training is complete ({{m.num_timesteps:,}} timesteps) - "
              f"{{'its timestep cap was reached before the stopping rule was met' if capped else 'the stopping rule was met'}}.")
    else:
        print(f"{{name}}: stage {{STAGE}} {{'- the stopping rule decides when it is done' if STAGE == len(m_env.stages) - 1 else 'not passed yet'}} "
              f"({{m.num_timesteps:,}} timesteps) - run this notebook again to continue.")
""")

    md("## Progress on this stage\n\nThe two methods side by side, against the timesteps each has spent **on this stage**: the pass rate (what promotion counts) and the reward.")
    code(r"""
if storage:
    storage.show_status()
plot_comparison({"MAPPO": log_dir, "PQN": pqn_log_dir}, stage=STAGE)
""")
    md("""
**Reading the curriculum plots** (one per method): each colour is a stage, the y-axis its pass rate over the last 50 battles, and the dashed line the 70 % bar. A new colour starting means the run moved on - by passing, or at a stage's cap. A colour drawn **dotted** after that is not training on that stage: from stage 3 on, 10-15 % of the battles replay earlier stages (rehearsal) so their skills are not forgotten. The title says the furthest stage the run has reached - **that** stage's notebook is the one that trains it.
""")
    code(r"""
plot_curriculum(log_dir, name="MAPPO")                          # pass rate per stage, one colour each
plot_curriculum(pqn_log_dir, name="PQN")
plot_reward_terms(log_dir, stage=STAGE, name="MAPPO")           # what the fleet is paid for on this stage
plot_reward_terms(pqn_log_dir, stage=STAGE, name="PQN")
""")
    code(r"""
plot_progress(log_dir)                       # PPO's diagnostics
plot_progress(pqn_log_dir)                   # PQN's diagnostics
""")
    if k >= 4:
        code(r"""
# the commanders: how often each leaves a ship free (1.0 = never gives orders)
for name, d in (("MAPPO", log_dir), ("PQN", pqn_log_dir)):
    path = os.path.join(d, "progress.csv")
    p = pd.read_csv(path) if os.path.exists(path) else None
    if p is None or not (p.stage == STAGE).any():
        print(f"{name}: no training updates on stage {STAGE} - it has not trained here (see 'The two models')")
    else:
        print(name)
        display(p[p.stage == STAGE][["total_timesteps", "orders_free", "win_rate", "pass_rate"]].tail(5))
""")
    if league:
        code(r"""
# the leagues: Elo ratings of each fleet, its champion, past copies, the rival (the other method) and the
# rule AI (chess-style; 400 points = 10:1 odds)
for name, m in (("MAPPO", model), ("PQN", pqn)):
    if m is not None:
        print(name)
        display(pd.DataFrame(m.league.table(), columns=["player", "elo", "our record against it"]).head(8))
""")

    md(f"""
## Evaluation: MAPPO vs PQN

Fresh battles on this stage against its own opponent, for every model{', and on held-out maps' if st['scenario'] == 'open' else ''}: each plays the way it trains - MAPPO samples its policy, PQN samples softmax(Q / 0.1) (with `exploration="epsilon"`: its best action, 2% random). The best checkpoints are evaluated too.
""")
    code(r"""
results = dict(baselines)
for name, cls, m, mdir in (("MAPPO", MAPPO, model, models_dir), ("PQN", FleetPQN, pqn, pqn_models_dir)):
    if m is None:
        print(f"{name}: no model yet - train it from Stage 0")
        continue
    m_env = env if cls is MAPPO else env_pqn
    results[f"{name} (latest)"] = evaluate(m, m_env, n_battles=40, stage=STAGE)
    best_path = os.path.join(mdir, "model_best.pt")
    if os.path.exists(best_path):
        stage_now = m_env.stage
        best = cls.load(best_path, env=m_env)          # (loading moves that run's stage to the one it was saved on)
        results[f"{name} (best)"] = evaluate(best, m_env, n_battles=40, stage=STAGE)
        m_env.set_stage(stage_now)
""" + (r"""    results[f"{name}, held-out maps"] = evaluate(m, m_env, n_battles=40, stage=STAGE, held_out=True)
""" if st['scenario'] == 'open' else "") + r"""columns = ["battles", "win_rate", "pass_rate", "win_rate_stderr", "mean_reward", "enemy_ships_sunk", "own_ships_lost",
           "points", "enemy_points"]
table = pd.DataFrame(results).T
table[[c for c in columns if c in table]]
""")

    md("""
## Every stage so far

How each method did on every stage of the curriculum, from its battle log: the timesteps it spent there until it passed, its pass rate over its last 50 qualifying battles there, and whether it passed.
""")
    code(r"""
compare_stages({"MAPPO": log_dir, "PQN": pqn_log_dir}, env.stages)
""")

    md("""
## Put the fleets into your Unity game

Exports each run's best checkpoint (or its latest, if that has reached a later stage) as a policy file the game runs in plain C# (`Assets/Scripts/RL/RLPolicy.cs`): MAPPO's as `naval_policy.bin`, PQN's as `naval_policy_ddqn.bin`. From the repository they go straight into `Assets/StreamingAssets/RL/`; elsewhere (SageMaker) into the run folders - download them (right-click > Download) and copy them there.

Then in Unity: **Play** > **TRAINING STAGES**; at the bottom pick **TRAINED FLEET: MAPPO** or **PQN**, and **WATCH AI** on any stage (or on the setup screen: **YOUR FLEET: LEARNED** / **ENEMY AI: LEARNED**). From a computer later: `python export_simple.py` and `python export_simple.py --run-name fleet_ddqn`.
""")
    code(r"""
from simple_mappo.export import GAME_POLICY, GAME_POLICY_PQN, export_policy, pick_checkpoint
in_repo = os.path.isdir(os.path.join("..", "Assets", "StreamingAssets"))
for mdir, rdir, game_file in ((models_dir, run_dir, GAME_POLICY), (pqn_models_dir, pqn_run_dir, GAME_POLICY_PQN)):
    checkpoint = pick_checkpoint(mdir)        # model_best.pt, or model_latest.pt if it reached a later stage
    if checkpoint is None or checkpoint.endswith("imitation.pt") and mdir == pqn_models_dir:
        continue
    target = game_file if in_repo else os.path.join(rdir, os.path.basename(game_file))
    print("exported", checkpoint, "->", os.path.abspath(export_policy(checkpoint, target)))
""")

    md("""
## Replays from this stage

**Every** replay recorded on this stage, MAPPO's then PQN's, oldest first: the ones from this session (also those whose live display above was lost when the page was closed), from earlier sessions, and - with team checkpoints - the ones a teammate or a training job recorded, downloaded from the team folder first.

A run records replays only on the stage it **trains** on. A run that is on another stage (each method climbs the curriculum at its own pace) is only evaluated here - so for it, one battle on this stage is recorded now (`..._evaluated`, once per checkpoint), and both methods always have a replay of this stage to compare. Each is embedded in the notebook as an animated GIF - an image plays in every notebook app (JupyterLab, SageMaker Studio, VS Code, Cursor; VS Code blocks embedded videos) and stays in the notebook when it is reopened or downloaded. An older replay with only an MP4 gets its GIF made here. `ALL_STAGES = True` shows every replay of both runs, from every stage.

They are a 2D map drawn from the game's data (the training player has no graphics); to see a fleet in the real game's graphics, export it (above) and use **WATCH AI** in Unity.
""")
    code(r"""
ALL_STAGES = False                   # True: every replay of both runs, from every stage
for replay_storage, replay_run_dir in ((storage, run_dir), (pqn_storage, pqn_run_dir)):
    if replay_storage:
        print(f"{replay_storage.pull_videos(replay_run_dir)} new replay(s) from {replay_storage.location}")
# a run that is on another stage did not train here, so it recorded nothing on this stage: record one battle
for name, m, m_env, m_video_dir, trained in (("MAPPO", model, env, video_dir, TRAIN_MAPPO),
                                             ("PQN", pqn, env_pqn, pqn_video_dir, TRAIN_PQN)):
    stem = f"battle_{m.num_timesteps:09d}_stage{STAGE}" if m is not None else ""
    if m is not None and not trained and not find_videos(m_video_dir, prefix=stem):
        print(f"{name} is on stage {m_env.stage} ('{m_env.stages[m_env.stage]['name']}'), so it did not train here - "
              f"recording one battle of it on stage {STAGE}:")
        record_battle(m, m_env, os.path.join(m_video_dir, f"{stem}_evaluated{video_ext()}"), stage=STAGE)
show_videos({"MAPPO": video_dir, "PQN": pqn_video_dir}, stage=None if ALL_STAGES else STAGE)
""")
    code(r"""
battles.close()
""")
    return _finish(cells)


def imitation_notebook() -> nbf.NotebookNode:
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    md(f"""
# 43008: Reinforcement Learning - WOW Naval MAPPO with a Fleet Commander

## Imitation: copy the Elite rule AI before reinforcement learning

{nav("P")}

A random policy almost never wins a naval battle, so reinforcement learning from scratch spends most of its time discovering what "sail, aim, shoot" means. AlphaStar (StarCraft II) solved the same problem by **first copying human players with supervised learning, then improving with RL**. We have no human replays, but the game has an **Elite rule-based AI** - so we copy that.

1. **Record** - the Elite AI flies both fleets; for every decision of every one of our ships the game reports the option it chose on each of the six action heads. That is a labelled dataset: *observation → action*.
2. **Clone** - train the captain network to predict those labels. It is **Week 9 Part A** again: a classifier, a `DataLoader`, `CrossEntropyLoss` (one per action head), `Adam`, a loss curve and an accuracy curve.
3. **Hand over** - the cloned captain is saved as `models/imitation.pt`. The Stage 0 notebook starts **both** RL models from it: MAPPO's captain (a KL penalty that fades out keeps it close to the clone at first - AlphaStar's KL to the supervised policy) and PQN's Q-network (the clone's head scores become its advantages, so its greedy play starts as the Elite AI's).

In this project's first pipeline, cloning alone won **58%** of King of the Hill battles where RL from scratch won 20%.

Run this notebook **once per run, before Stage 0**.
""" + METHOD_MD)
    md("## Setup")
    code(SETUP)
    code(INSTALL)
    code(r"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simple_mappo import (NavalEnv, MAPPO, TeamStorage, DEFAULT_CURRICULUM, collect_expert, save_dataset, load_dataset,
                          behaviour_cloning, save_imitation, plot_imitation, evaluate)
""")
    md("## Team checkpoints\n\nThe same `RUN_NAME` as the stage notebooks: the imitation checkpoint goes into that run's folder, where Stage 0 picks it up.")
    code(TEAM + r"""
if storage:
    storage.pull(run_dir)            # the run's files so far (and its imitation checkpoint, if a teammate made one)
if os.path.exists(os.path.join(models_dir, "imitation.pt")):
    print("this run already has an imitation checkpoint (models/imitation.pt) - running on replaces it")
""")
    md("""
## Configuration

The network settings (`d_model`, `attn_layers`, `hidden_size`, `recurrent`) must be **the same as in the stage notebooks** - the clone becomes the RL model's captain.
""")
    code(f"""
config = OrderedDict([
    ('n_envs', max(1, min(16, os.cpu_count() // 2))),     # battles recorded in parallel
    ('n_battles', 60),                 # battles of the Elite AI to record
    ('record_stages', [2, 3, 4, 5, 6]),  # curriculum stages to record (both fleets: the Elite AI)
    ('num_epochs', 8),                 # passes over the recorded data
    ('batch_size', 64),                # windows per minibatch
    ('window', 32),                    # decisions per window (the GRU learns from sequences)
    ('learning_rate', 3e-4),
])
{MAPPO_SETTINGS}
pd.Series(config).to_frame("value")
""")
    md("## Step 1: record the Elite AI\n\nEach battle slot draws one of `record_stages`; both fleets are the Elite AI, and our fleet's decisions are labelled. About a minute per 10 battles with 8 battle slots.")
    code(r"""
recorder = NavalEnv(stages="default", n_envs=config['n_envs'], record_expert=True, base_port=5300,
                    unity_log_dir=os.path.join(run_dir, "unity_logs"))
data = collect_expert(recorder, config['n_battles'], stages=config['record_stages'])
dims, heads = recorder.dims, recorder.heads
recorder.close()
save_dataset(data, os.path.join(run_dir, "imitation", "demos.npz"))
""")
    md("### Look at the data\n\nWhat the Elite AI chooses on each head - like looking at the Fashion-MNIST samples before training. A head where one option dominates is easy to copy; the interesting ones are *move* and *target*.")
    code(r"""
names = [h["name"] for h in heads]
fig, axes = plt.subplots(1, len(names), figsize=(18, 3))
for h, (ax, name) in enumerate(zip(axes, names)):
    labels = data["labels"][data["valid"], h]
    ax.bar(*np.unique(labels, return_counts=True))
    ax.set_title(name)
    ax.set_xlabel("option")
plt.tight_layout()
plt.show()
print(f"{len(data['lengths'])} ship sequences, {int(data['valid'].sum()):,} labelled decisions")
""")
    md("## Step 2: clone it\n\nThe Week 9 Part A loop, with one cross-entropy per action head. Held-out accuracy is measured on ships the training never saw.")
    code(r"""
captain, history = behaviour_cloning(data, dims, heads, d_model=mappo_settings['d_model'],
                                     attn_layers=mappo_settings['attn_layers'], hidden_size=mappo_settings['hidden_size'],
                                     recurrent=mappo_settings['recurrent'], num_epochs=config['num_epochs'],
                                     batch_size=config['batch_size'], window=config['window'],
                                     learning_rate=config['learning_rate'])
plot_imitation(history, names)
""")
    md("## Step 3: save and hand over")
    code(r"""
path = save_imitation(os.path.join(models_dir, "imitation.pt"), captain, dims, heads, history)
print("saved", path)
if storage:
    storage.push(run_dir, None, status={"imitation_accuracy": dict(zip(names, history["test_accuracy"][-1]))})
    storage.release()                # a push holds the run: free it for the Stage 0 notebook
""")
    md("""
## How good is the clone?

Before any RL: the cloned captains against each stage's opponent, next to random play. The stage notebooks then improve on this with reinforcement learning (and the commander).
""")
    code(r"""
env = NavalEnv(stages="default", n_envs=config['n_envs'], unity_log_dir=os.path.join(run_dir, "unity_logs"))
model = MAPPO(env, **mappo_settings)
model.init_from_imitation(path)
rows = {}
for stage in range(6):
    clone, rand = evaluate(model, env, n_battles=24, stage=stage), evaluate(None, env, n_battles=24, stage=stage)
    rows[env.stages[stage]["name"]] = {"clone: pass rate": clone["pass_rate"], "clone: win rate": clone["win_rate"],
                                       "random: pass rate": rand["pass_rate"]}
env.close()
pd.DataFrame(rows).T
""")
    md(f"Next: [Stage 0 - {STAGES[0]['name']}]({STAGES[0]['file']}).")
    return _finish(cells)


VERSUS_FILE = "Versus-MAPPO-vs-PQN.ipynb"

VERSUS_METHODS_MD = r"""
## 1. The two methods, in detail

### 1.1 What they share

Both fleets are built from **the same parts** — only the learning rule differs, so a difference in the result is a difference of the method:

| Shared part | What it is | File |
|---|---|---|
| The environment | the real Unity game, headless, one battle per process; a decision every 1 s of game time | `env.py` |
| What a ship sees | tokens: itself (200 numbers), up to 8 allies, up to 9 spotted enemies (fog of war), 5 circles, 4 islands | `env.py`, `RLObservation.cs` |
| What a ship does | 6 heads: move (15 + sail to circle k), speed, target (auto or enemy k), fire, torpedo, ability | `RLActions.cs` |
| The **captain** | one network for every ship: an embedding per token kind → 2 attention layers → pooling → GRU memory → 6 heads; target and circle are *pointer* heads | `networks.py` |
| The **commander** | every 10 s an order per ship: free / engage / hold circle k | `networks.py` |
| The reward | circles (spread over the circles that need a ship), circles held, damage, kills, the result; team spirit, zero-sum | `rewards.py` |
| The start | a copy of the Elite rule AI (supervised learning, StageP), kept close by a fading penalty | `imitation.py` |
| Self-play | mirror battles, a champion, past copies, and the other method as rival | `league.py` |

What differs is **what the captain's head outputs mean** and **how they are trained**:

```
                         the captain's 6 heads output scores for every option
                                          │
              ┌───────────────────────────┴───────────────────────────┐
         MAPPO: logits                                         PQN: advantages A_d
   probabilities = softmax(logits)                   Q_d = V + A_d − mean(A_d)  (dueling)
   act: SAMPLE an option                             act: SAMPLE by softmax(Q / τ) (Boltzmann; or ε-greedy)
   learn: make good options likelier                 learn: make Q match the reward that followed
```

### 1.2 MAPPO — policy gradient (Week 10's PPO, for a team)

**Idea.** The captain outputs a probability for every option. After playing, options that turned out *better than expected* become more likely, worse ones less likely. "Expected" comes from a **critic** that predicts the return; MAPPO's critic is **centralised** — during training it sees the *true* battle (every enemy, even unspotted), so its predictions are much better, while the ships themselves still act only on what they can see (centralised training, decentralised execution).

**One update** (`mappo.py`):

1. **Collect** 256 decisions in every battle with the current policy (sampling). Store observations, actions, log-probabilities, the GRU memory, the critic's values and the rewards (per reward group: objective, combat, outcome).
2. **Advantages** with GAE, per ship and reward group, backwards in time (γ = 0.995, λ = 0.95):
   ```
   δ_t = r_t + γ V(s_t+1) − V(s_t)          A_t = δ_t + γλ A_t+1          A = A_objective + A_combat + A_outcome
   ```
   The commander's step lasts 10 decisions, so its reward is the fleet's reward summed over those 10 (a semi-Markov step).
3. **Update** — 5 epochs × 4 minibatches:
   - **captains** — PPO's clipped objective on chunks of 32 decisions (the GRU replays them):
     ```
     ratio = π_new(a|s) / π_old(a|s)
     loss  = − min(ratio·A, clip(ratio, 0.8, 1.2)·A)        (and ≥ 3·A when A < 0: Honor of Kings' dual clip)
             − 0.01 × entropy                                 (explore; fades to 0.002)
             + 0.1 × KL(Elite clone ‖ policy)                 (stay near the clone at first; fades to 0.01)
     ```
   - **critic** — predict the returns (normalised, clipped), the commander's value and the chance of winning;
   - **commander** — the same PPO on its orders (one joint decision for the fleet).
4. Throw the data away and collect again (**on-policy**: every decision is used for one update only).

### 1.3 PQN — value-based (Week 9's DQN, for a team)

**Idea.** Learn **Q(s, a)** — the return to expect after choosing a in state s — and always take the option with the highest Q. Learning makes Q match what actually followed: the rewards plus the value of where the ship ended up.

**Six heads at once.** One Q per *complete* action would need 20 × 5 × 10 × 2 × 2 × 15 = 60,000 outputs. Instead every head has its own Q, sharing one state value (**dueling + branching** DQN):
```
Q_d(s, a_d) = V(s) + A_d(s, a_d) − mean of A_d over the legal options
```
The advantages A_d are the captain's head outputs; V is one extra layer. So **the Q-network is the captain** — which is why the game can fly it with the same C# code (it just takes the best option).

**One round** (`ddqn.py`):

1. **Play** 32 decisions in every battle. With `exploration="softmax"` (the default in these notebooks) each option is chosen with probability ∝ exp(Q / 0.1) - **Boltzmann exploration**: from the clone, whose scores × 0.1 are the first advantages, that is exactly the clone's own choice, sampled; as the Q-values grow apart it becomes greedier. With `exploration="epsilon"`: the best option, or with probability ε a random legal one (ε 0.3 → 0.05 from the clone). Measured on Gunnery over 200,000 timesteps from the clone: ε-greedy passed 28-49 % of every 100 battles, softmax 50-71 % - greedy play of a cloned policy takes only its single likeliest option on every head (e.g. always "hold fire" when the clone is 55/45), and loses most of what the clone knew.
2. **Store** one **chunk** per fleet in the **replay buffer**: 33 observations, the actions, orders, rewards, battle ends — and the GRU memory at the chunk's start, so training can replay the chunk through the memory (**R2D2**). The buffer holds 40,000 decisions; old ones drop out.
3. Once the buffer holds 4,000 decisions: **4 gradient steps**, each on 32 chunks sampled from it (**off-policy**: a decision is reused many times). For every decision:
   ```
   a*_d = argmax Q_online,d(s_t+5, ·)                       the ONLINE network picks the next option (per head)…
   next = mean over heads of Q_target,d(s_t+5, a*_d)        …the TARGET network scores it  (PQN)
   y    = r_t + γ r_t+1 + … + γ⁴ r_t+4 + γ⁵ · next          5-step return, γ = 0.99 (stops at the battle's end)
   loss = mean over heads of Huber(Q_online,d(s_t, a_t,d) − y)   every head trained to the same y (BDQ)
   ```
   Plain DQN uses the target network for both picking and scoring and **over-estimates** (it takes the max of noisy numbers); PQN splits the two jobs.
4. Every 250 gradient steps, copy the online network into the **target network** (Week 9 C/D).
5. **Commander**: its scores for (ship, order) are Q-values too; a set of orders is worth the sum. Between its decisions an order stays in force, so it learns at every step: the next value is Q(next state, *the same* orders), or the best orders at its next decision (h-DQN, intra-option Q-learning).
6. **Starting from the clone** ("kickstarting"): the clone's scores × 0.1 become the first advantages (the same best options, at the size of Q-values), and a fading penalty KL(clone ‖ softmax(A / 0.1)) keeps it close while the Q-values are still meaningless — without it the first updates erased the clone.

### 1.4 Side by side

| | MAPPO | PQN |
|---|---|---|
| Family | policy gradient | value-based |
| Acts by | **sampling** its probabilities | **sampling** softmax(Q / 0.1) - Boltzmann (or the **best** Q, ε-greedy) |
| Learns from | the battles it just played (on-policy) | a replay buffer of past chunks (off-policy) |
| Learning signal | advantage from a centralised critic (GAE) | Double-Q target with 5-step returns |
| Exploration | entropy bonus | Boltzmann (softmax over Q), or ε-greedy |
| Stability tricks | clip, dual clip, value normalisation | target network, Huber loss, n-step returns |
| Commander | PPO at a 10 s clock | Q-learning on (ship, order) |
| Strength (in theory) | copes well with an opponent that keeps changing; what OpenAI Five and Honor of Kings used | reuses its data many times; the StarCraft micro-management classic (QMIX) |
"""

VERSUS_TEST_MD = r"""
## 2. What this notebook tests

The stage notebooks compare the methods **side by side** — each against the rule AI. Here they meet **head to head**: MAPPO's fleet on one side of the battle, PQN's on the other.

**Head to head, from both sides.** A battle is played with MAPPO's fleet as *blue* (our side) and PQN's as *red* (the enemy), and again the other way round — the maps are symmetric, but spawning first or second, north or south, could still favour one side, so both seats are played and added up. Each fleet plays the way it trains: MAPPO samples its policy, PQN samples softmax(Q / 0.1) (or its best option, 2 % random, if it was trained ε-greedy). Technically the enemy is the **rival** of `league.py`: `blue.set_rival(red)` puts red's network in blue's league, and the head-to-head stage sends every battle's enemy to it.

**Training each other** (`TRAIN_EACH_OTHER = True`). Rounds of *iterated best response*: MAPPO trains against PQN's current fleet, then PQN trains against MAPPO's improved one, and so on — after every round both are measured again: head to head, and each against the rule AI. Things to watch for:

* **Who adapts faster.** Off-policy PQN reuses every battle many times; on-policy MAPPO needs fresh battles. The head-to-head win rate over the rounds shows which one learns to counter the other first.
* **Exploitation.** Two fleets that only meet each other can learn tricks that only work on each other. That is why `RIVAL_SHARE` keeps the stage's rule AI in the training mix (30 % by default) — and why the rule-AI win rates are measured every round: if a fleet beats its rival more but the rule AI less, it is exploiting, not improving.
* **Cycles.** A beats B, B adapts and beats A, A adapts back... — rock-paper-scissors. The league's champion and past copies exist to break such cycles in the main training.

Nothing here touches the main runs: the fleets are loaded from `runs_simple/fleet` and `runs_simple/fleet_pqn` (or the team folder), and what they learn here is saved under `runs_simple/versus/`.
"""


def versus_notebook() -> nbf.NotebookNode:
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    md(f"""
# 43008: Reinforcement Learning - WOW Naval Fleet: MAPPO vs PQN, fleet against fleet

{nav("V")}

What happens when the two methods **fight each other**? This notebook loads the team's MAPPO fleet and PQN fleet, puts them on opposite sides of the same battle (both ways round), and — if you like — lets them **train against each other** in rounds, measuring after every round who is winning and whether either got better or just learned to exploit the other.

It starts with a detailed explanation of both methods (section 1) and of the test (section 2).
""")
    md(VERSUS_METHODS_MD)
    md(VERSUS_TEST_MD)
    md("## 3. Setup\n\nOn SageMaker, run this from the unzipped bundle (`<folder>/Training/`); on your computer, from the repository's `Training/` folder. The battles are the real Unity game.")
    code(SETUP)
    code(INSTALL)
    code(r"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simple_mappo import (NavalEnv, MAPPO, FleetPQN, Monitor, TeamStorage, DEFAULT_CURRICULUM, versus_stages,
                          head_to_head, both_sides, elo_gap, evaluate, record_battle, show_videos, stage_fleets)
from simple_mappo.env import load_scenario
from simple_mappo.export import pick_checkpoint
from simple_mappo.versus import FIGHT, TRAIN
""")
    md("""
## 4. Where the fleets come from

The latest MAPPO run (`RUN_NAME`) and PQN run (`PQN_RUN_NAME`) — pulled from the team folder if there is one (read only: nothing is pushed or locked), otherwise from this machine. A run with no checkpoint yet starts from the imitation clone (the Elite AI's copy), so the notebook always runs.
""")
    code(r"""
RUN_NAME = "fleet"                   # the stage notebooks' MAPPO run
PQN_RUN_NAME = RUN_NAME + "_pqn"     # and their PQN run
if ON_SAGEMAKER:
    from simple_mappo.aws import team_storage_uri
    TEAM_STORAGE = team_storage_uri()
else:
    TEAM_STORAGE = None              # e.g. "s3://<team-bucket>/wow-mappo"

runs = {"MAPPO": f"runs_simple/{RUN_NAME}", "PQN": f"runs_simple/{PQN_RUN_NAME}"}
if TEAM_STORAGE:
    for name, run in (("MAPPO", RUN_NAME), ("PQN", PQN_RUN_NAME)):
        TeamStorage(TEAM_STORAGE, run, user="versus-notebook").pull(runs[name])     # read only
imitation = os.path.join(runs["MAPPO"], "models", "imitation.pt")
versus_dir = "runs_simple/versus"     # what the fleets learn here goes here, never into the main runs
""")
    md("""
## 5. Configuration

`STAGE` picks the battle from the curriculum (its map, fleets, commander and team spirit) — 4 (King of the Hill 3v3, 6 minutes) is quick; 8 (9v9, five circles, 12 minutes) is the big one. `N_BATTLES` per seat (so twice that per measurement).
""")
    code(r"""
STAGE = 4                            # the battle: a curriculum stage (3 duel, 4 King of the Hill, 5 archipelago, 8 9v9)
N_BATTLES = 24                       # battles per seat, per measurement
TRAIN_EACH_OTHER = True              # False: only the head-to-head
ROUNDS = 3                           # rounds of training each other
TIMESTEPS_PER_ROUND = 50_000         # per method, per round
RIVAL_SHARE = 0.7                    # while training: the other fleet in 70 % of battles, the rule AI in 30 %

config = OrderedDict([
    ('n_envs', max(1, min(16, os.cpu_count() // 2))),     # battles in parallel, one Unity process each
    ('decision_period', 1.0),
    ('sim_dt', 0.08),
])
stage = DEFAULT_CURRICULUM[STAGE]
print(f"Stage {stage['name']}: {stage['battle']}")
print("Ships:", stage_fleets(stage, load_scenario)["text"])
""")
    md("## 6. Environment\n\nTwo stages made from the chosen one: **head to head** (every enemy is the other method's fleet) and **training each other** (the other fleet in `RIVAL_SHARE` of the battles, the rule AI in the rest). One set of game players, a battle log for each fleet.")
    code(r"""
battles = NavalEnv(stages=versus_stages(STAGE, RIVAL_SHARE), n_envs=config['n_envs'],
                   decision_period=config['decision_period'], sim_dt=config['sim_dt'],
                   unity_log_dir=os.path.join(versus_dir, "unity_logs"))
env_mappo = Monitor(battles, os.path.join(versus_dir, "mappo", "logs"))
env_pqn = Monitor(battles, os.path.join(versus_dir, "pqn", "logs"))
pd.DataFrame(battles.stages)[["name", "opponents", "commander", "team_spirit"]]
""")
    md("## 7. Load the two fleets")
    code(r"""
def load_fleet(cls, run_dir, env, name):
    checkpoint = pick_checkpoint(os.path.join(run_dir, "models"), verbose=False)
    log_dir = os.path.join(versus_dir, "mappo" if cls is MAPPO else "pqn", "logs")
    if checkpoint and not checkpoint.endswith("imitation.pt"):
        model = cls.load(checkpoint, env=env, log_dir=log_dir, verbose=1)
        print(f"{name}: {checkpoint}, {model.num_timesteps:,} timesteps")
    else:
        model = cls(env, log_dir=log_dir, verbose=1, **({"exploration": "softmax"} if cls is FleetPQN else {}))
        model.init_from_imitation(imitation)
        print(f"{name}: no checkpoint yet - the Elite AI clone, untrained")
    env.set_stage(FIGHT)
    return model

mappo = load_fleet(MAPPO, runs["MAPPO"], env_mappo, "MAPPO")
pqn = load_fleet(FleetPQN, runs["PQN"], env_pqn, "PQN")
""")
    md("""
## 8. Head to head

Both seats: MAPPO as blue against PQN as red, then the other way round. `win_rate` is **blue's**. The overall MAPPO win rate adds both seats up; the Elo gap is the rating difference that would predict it (+200 ≈ 76 %, +400 ≈ 91 %).
""")
    code(r"""
def measure(label):
    r = both_sides(mappo, pqn, env_mappo, env_pqn, n_battles=N_BATTLES)
    table = pd.DataFrame({k: r[k] for k in ("MAPPO as blue", "PQN as blue")}).T
    print(f"{label}: MAPPO wins {r['mappo_win_rate']:.0%} of {r['battles']} battles against PQN "
          f"(Elo gap {elo_gap(r['mappo_win_rate']):+.0f})")
    return r, table

result, table = measure("Before training each other")
table[["battles", "win_rate", "win_rate_stderr", "enemy_ships_sunk", "own_ships_lost", "points", "enemy_points"]]
""")
    md("### For reference: each against the rule AI\n\nThe same battle against the stage's rule AI — so a head-to-head result can be read against how good each fleet is in general.")
    code(r"""
def vs_rule():
    return {"MAPPO": evaluate(mappo, env_mappo, n_battles=N_BATTLES, stage=TRAIN),
            "PQN": evaluate(pqn, env_pqn, n_battles=N_BATTLES, stage=TRAIN)}

rule = vs_rule()
pd.DataFrame(rule).T[["battles", "win_rate", "win_rate_stderr", "enemy_ships_sunk", "own_ships_lost"]]
""")
    md("### Watch them fight\n\nOne battle, MAPPO's fleet in blue (cyan trails) against PQN's in red - a 2D map drawn from the game's data.")
    code(r"""
video_dir = os.path.join(versus_dir, "videos")
os.makedirs(video_dir, exist_ok=True)
mappo.set_rival(pqn, champion=False)
battles.set_opponent_policy(mappo._opponent_act, mappo.league.sample)
path = record_battle(mappo, env_mappo, os.path.join(video_dir, "mappo_blue_vs_pqn_red_round0.mp4"), stage=FIGHT,
                     evaluation=False)
show_videos(video_dir, contains="round0", newest=1)
""")
    md("""
## 9. Train each other

Each round: **MAPPO trains** against PQN's current fleet (`set_rival`), then **PQN trains** against MAPPO's improved fleet; then both are measured head to head and against the rule AI. The progress tables are the usual ones (`win_rate` there is against the rule AI, `win_rate_self_play` against the rival). Every round's fleets are saved to `runs_simple/versus/<method>/models/round_<k>.pt`.
""")
    code(r"""
history = [{"round": 0, "mappo_vs_pqn": result["mappo_win_rate"],
            "mappo_vs_rule": rule["MAPPO"]["win_rate"], "pqn_vs_rule": rule["PQN"]["win_rate"]}]
if TRAIN_EACH_OTHER:
    for k in range(1, ROUNDS + 1):
        print(f"===== round {k}: MAPPO trains against PQN =====")
        env_mappo.set_stage(TRAIN)
        mappo.set_rival(pqn, champion=False)
        mappo.learn(total_timesteps=mappo.num_timesteps + TIMESTEPS_PER_ROUND, log_interval=5)
        print(f"===== round {k}: PQN trains against MAPPO =====")
        env_pqn.set_stage(TRAIN)
        pqn.set_rival(mappo, champion=False)
        pqn.learn(total_timesteps=pqn.num_timesteps + TIMESTEPS_PER_ROUND, log_interval=8)
        mappo.save(os.path.join(versus_dir, "mappo", "models", f"round_{k}"))
        pqn.save(os.path.join(versus_dir, "pqn", "models", f"round_{k}"))

        result, table = measure(f"After round {k}")
        rule = vs_rule()
        history.append({"round": k, "mappo_vs_pqn": result["mappo_win_rate"],
                        "mappo_vs_rule": rule["MAPPO"]["win_rate"], "pqn_vs_rule": rule["PQN"]["win_rate"]})
        display(pd.DataFrame(history).set_index("round"))
""")
    md("## 10. What happened")
    code(r"""
h = pd.DataFrame(history).set_index("round")
fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.5))
left.plot(h.index, h.mappo_vs_pqn, "o-", label="MAPPO's win rate against PQN")
left.plot(h.index, 1 - h.mappo_vs_pqn, "o-", label="PQN's win rate against MAPPO")
left.axhline(0.5, color="gray", lw=0.8, ls="--")
left.set_title("Head to head (both seats)")
right.plot(h.index, h.mappo_vs_rule, "o-", label="MAPPO vs rule AI")
right.plot(h.index, h.pqn_vs_rule, "o-", label="PQN vs rule AI")
right.set_title(f"Against the rule AI ({DEFAULT_CURRICULUM[STAGE]['name']})")
for ax in (left, right):
    ax.set_ylim(0, 1)
    ax.set_xlabel("round of training each other")
    ax.set_ylabel("win rate")
    ax.legend()
plt.tight_layout()
plt.show()
h
""")
    md("""
**Reading it**

* **Left, above 0.5**: MAPPO's fleet is the stronger one in this battle; the gap from 0.5 is the size of the edge (±0.1 is within the noise of 48 battles).
* **Left, moving**: whoever trained last against the other should gain — if the line swings back and forth round after round, the two are chasing each other (a cycle); if one side keeps winning, it adapts better.
* **Right**: if a fleet's win rate against the rule AI falls while it beats its rival more, it is learning tricks that only work on the rival (exploitation) — raise the rule AI's share (`RIVAL_SHARE` lower). If both rise, training each other made both genuinely better.
""")
    md("### The last battle, after training each other")
    code(r"""
mappo.set_rival(pqn, champion=False)
battles.set_opponent_policy(mappo._opponent_act, mappo.league.sample)
k = len(history) - 1
path = record_battle(mappo, env_mappo, os.path.join(video_dir, f"mappo_blue_vs_pqn_red_round{k}.mp4"), stage=FIGHT,
                     evaluation=False)
show_videos(video_dir, contains=f"round{k}", newest=1)
""")
    md("### Every replay of this notebook\n\nAll the head-to-head battles recorded here so far (`runs_simple/versus/videos/`), from this session and earlier ones, oldest first - MAPPO's fleet in blue, PQN's in red.")
    code(r"""
show_videos(video_dir)
""")
    code(r"""
battles.close()
""")
    return _finish(cells)


def _finish(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    keep = {IMITATION_FILE} | {st["file"] for st in STAGES}
    nbf.write(versus_notebook(), os.path.join(HERE, VERSUS_FILE))
    print("wrote", os.path.join(HERE, VERSUS_FILE))
    for old in glob.glob(os.path.join(HERE, "Stage*.ipynb")):
        if os.path.basename(old) not in keep:          # notebooks of an earlier curriculum
            os.remove(old)
            print("removed", old)
    path = os.path.join(HERE, IMITATION_FILE)
    nbf.write(imitation_notebook(), path)
    print("wrote", path)
    for k, st in enumerate(STAGES):
        path = os.path.join(HERE, st["file"])
        nbf.write(notebook(k), path)
        print("wrote", path)
