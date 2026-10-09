"""Writes the stage notebooks - StageP (imitation) and one per curriculum stage, all training the SAME model.

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

STAGES = [
    dict(file="Stage0-Gunnery.ipynb", name="Gunnery", scenario="gunnery", fleet="1 battleship vs 1 cruiser",
         opponent="Passive target (stop, hold fire)", rule="target sunk within 150 s in ≥70% of the last 50 battles",
         session=200_000, script="close",
         learns="""The first lesson is the most basic one: **destroy a ship**. Our battleship starts 7 km from an enemy cruiser that neither moves nor shoots - Python flies it with "stop, hold fire". The circle reward is off; damage, the kill and the win pay.

A battle is *passed* when the target is sunk within 150 seconds - just sinking it eventually is too easy (random play does it), so the stage asks for doing it quickly: picking the target, keeping the guns firing, closing in and using torpedoes.""",
         measured="Measured (32 battles each): random play sinks the target within 150 s in 52% of battles, the script \"close in, fire at will, torpedoes\" 76%."),
    dict(file="Stage1-Capture.ipynb", name="Capture", scenario="capture", fleet="1 cruiser",
         opponent="Passive destroyer, far away", rule="circle captured in ≥70% of the last 50 battles",
         session=200_000, script="circle",
         learns="""**Capture the circle.** One cruiser, one neutral capture circle 4.5 km ahead; the only enemy is a passive destroyer 16 km beyond it, out of sight. Points come only from holding the circle, so the cruiser has to sail into it and stay inside until it is captured.""",
         measured="Measured: random play captures the circle in 0% of battles, the script \"sail to circle A\" in 100%."),
    dict(file="Stage2-Defend.ipynb", name="Defend", scenario="defend", fleet="1 cruiser vs 1 cruiser",
         opponent="Recruit rule-based AI", rule="won in ≥70% of the last 100 battles",
         session=300_000, script="circle",
         learns="""**Defend yourself while holding the circle.** A neutral circle lies 3 km ahead of our cruiser; an enemy cruiser flown by the rule AI comes for it from 8 km beyond. Winning means taking the circle and holding it *while staying afloat* - angling, dodging torpedoes, smoke, repair and fighting back. A per-second "survive" reward is switched on for this stage.""",
         measured="Measured: random play wins 12% of battles, sailing straight to the circle 0% - the cruiser has to fight smart."),
    dict(file="Stage3-Duel.ipynb", name="Duel", scenario="bb_duel", fleet="1v1 battleships",
         opponent="Recruit rule AI 70% · itself 30% (both sides learn)", rule="won in ≥70% of the last 100 battles against the rule AI",
         session=300_000, script="broadside",
         learns="""**1v1.** Two battleships 13 km apart on open sea, both spotted and inside gun range from the first second. The duel is about **choosing and timing fire, angling the hull, and not throwing the ship away**. The circle reward is off here: the circle lies between the two battleships, so racing to it means sailing bow-on into the enemy's guns.

**Self-play starts here.** 30% of battles are *mirror* battles: the fleet flies **both** battleships and **both sides learn** (OpenAI Five's self-play) - twice the training data from those battles, and an opponent that is always exactly as good as the fleet. Only the battles against the rule AI count for promotion.""",
         measured="Measured earlier with this battle: random play wins 16-28%, the broadside script 70%, flat MAPPO from scratch 59% after 300,000 timesteps."),
    dict(file="Stage4-King-of-the-Hill.ipynb", name="King of the Hill", scenario="koth_3v3", fleet="3v3 mixed fleet",
         opponent="Veteran rule AI 70% · itself 30% (both sides learn)", rule="won in ≥70% of the last 100 battles against the rule AI", session=500_000, script="circle",
         learns="""**The first fleet battle - and the commander joins in.** A battleship, a cruiser and a destroyer a side, one capture circle in the middle of the open sea, 6 minutes. Every 10 seconds the **fleet commander** gives each ship an order - *free*, *engage* or *hold the circle* - and the captains carry it out (the circle reward follows the order). Team spirit 0.3: each ship's own reward is now partly shared with the fleet. 15% of battles replay earlier stages so the single-ship skills stay.""",
         measured="Measured earlier with this battle: random play wins about 18% against Veteran, flat MAPPO from scratch 25-33%, the circle script 67-88% (against Recruit), imitation alone (naval_rl) 58%."),
    dict(file="Stage5-Archipelago.ipynb", name="Archipelago", scenario="archipelago_3v3", fleet="3v3 mixed fleet",
         opponent="Veteran rule AI 70% · itself 30% (both sides learn)", rule="won in ≥70% of the last 150 battles against the rule AI", session=500_000, script="circle",
         learns="""The same three ships a side, now among islands with **three capture circles** (A, B, C) and 8 minutes. This is where the commander earns its place: **which ship holds which circle, and who hunts**. Islands break line of sight, so spotting, smoke, radar and cover matter. Team spirit 0.5.""",
         measured="Measured earlier: imitation alone (naval_rl) won 45%, flat MAPPO 27%."),
    dict(file="Stage6-Domination.ipynb", name="Domination", scenario="domination_6v6", fleet="6v6 fleet",
         opponent="League: Elite rule AI 40% · itself 40% (both sides learn) · past selves 20%",
         rule="won in ≥70% of the last 200 battles against the Elite AI", session=500_000, script="circle",
         learns="""The full Domination game at 6v6, 15 minutes, on a **newly generated battlefield** every 8 battles (random preset, island density and weather). The enemy is drawn from a **league** (OpenAI Five / AlphaStar): the Elite rule AI, the fleet **itself** (a mirror battle - both sides learn), or an older copy picked by prioritised fictitious self-play - the ones we lose to come up most. Everyone gets an **Elo rating**. Only battles against the Elite AI count for promotion. Team spirit 0.8.""",
         measured=""),
    dict(file="Stage7-Open.ipynb", name="Open", scenario="open", fleet="Random 4–8 ships per side",
         opponent="League: Elite rule AI 40% · itself 40% (both sides learn) · past selves 20%",
         rule="Final stage: won in ≥70% of the last 250 held-out battles against the Elite AI", session=500_000,
         script="circle",
         learns="""Anything the game can produce: 4 to 8 ships a side, a random game mode (Domination, Skirmish, Fleet Battle, Capture and Control), map preset, density, weather and circle size, 20 minutes - against the league. Team spirit 1.0: every ship cares about the fleet as much as about itself.

There is **no further promotion**. Every 100,000 timesteps the fleet plays 50 **held-out** battles against the Elite AI on generated maps whose seeds training never uses. Training stops once the last 250 held-out battles reach 70%.""",
         measured=""),
]

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
import importlib.util
missing = [pkg for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("PIL", "pillow"))
           if importlib.util.find_spec(mod) is None]
if missing:
    !pip install -q {" ".join(missing)}
if importlib.util.find_spec("torch") is None:
    !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"""

TEAM = r"""
YOUR_NAME = "your-name"              # shown to your teammates
RUN_NAME = "fleet"                   # the same in every stage notebook (and StageP)
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

METHOD_MD = """
### The method in one picture

| Piece | What it does | Where it comes from | Course link |
|---|---|---|---|
| **Captain** (every ship, one shared network) | its own view → 6 action heads; a transformer over ships/circles/islands, a GRU memory, pointer heads for *which target* and *which circle* | AlphaStar, OpenAI Five | W10-C custom feature extractor; frame stacking → memory |
| **Fleet commander** | every 10 s: an order per ship - free / engage / hold circle k | Honor of Kings' hierarchical macro strategy | the same PPO at a slower clock |
| **Centralised critic** | sees the true battle (training only); one value per reward group + win probability | MAPPO, Honor of Kings multi-head value | W10-A/B actor-critic |
| **Update** | PPO with dual clip, value normalisation and clipping | MAPPO, Honor of Kings | W10-B PPO |
| **Warm start** | the captain first copies the Elite rule AI (StageP), then a fading KL penalty keeps it close | AlphaStar | W9-A supervised training loop |
| **Self-play & league** | from stage 3: mirror battles - the fleet on **both** sides, **both** learning; stages 6-7 add past copies (PFSP) and Elo ratings; the rule AI stays as the anchor promotion is judged against | OpenAI Five, AlphaStar | W9-C/D frozen target network |
| **Curriculum** | 8 stages, each passed at ≥70% of its own goal; earlier stages are replayed 10-15% of the time | TiZero, OpenAI Five | - |
"""


def nav(current: str) -> str:
    items = [f"**Imitation**" if current == "P" else f"[Imitation]({IMITATION_FILE})"]
    items += [f"**Stage {i}**" if str(i) == current else f"[Stage {i}]({s['file']})" for i, s in enumerate(STAGES)]
    return " · ".join(items) + " · [the full walkthrough](WOW-MAPPO-Simple.ipynb)"


def notebook(k: int) -> nbf.NotebookNode:
    st = STAGES[k]
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    last = k == len(STAGES) - 1
    league = "League" in st["opponent"]

    md(f"""
# 43008: Reinforcement Learning - WOW Naval MAPPO with a Fleet Commander

## Stage {k}: {st['name']}

{nav(str(k))}

| Stage | Battle | Fleet | Opponent | {'Stopping' if last else 'Promotion'} requirement |
|---|---|---|---|---|
| {k} - {st['name']} | `{st['scenario']}` | {st['fleet']} | {st['opponent']} | {st['rule']} |

### What the fleet learns here

{st['learns']}

{st['measured']}

### One model through every stage

All stage notebooks train the **same model with the same method**. This notebook continues the run's latest checkpoint, which carries everything learned so far - the networks, the optimisers, the value normalisers, the league, the timestep count and the stage - and the run's battle log, from which the promotion window continues where the last session stopped.

* **Training is continuous.** Not passed yet? **Run this notebook again** - each session continues the last one (yours or a teammate's), from `models/model_latest.pt`, which is refreshed every `save_interval` (so even a crashed session loses little). `KEEP_GOING = True` (configuration cell) carries on into the next stages in the same session instead of stopping at the promotion.
* {'This is the final stage: training ends when the held-out stopping rule is met.' if last else f"Passed? Training stops and saves, and the model moves to stage {k + 1}: open [Stage {k + 1}]({STAGES[k + 1]['file']})."}
{'* No model yet? It starts from the imitation checkpoint of the [Imitation notebook](' + IMITATION_FILE + ') - run that one first.' if k == 0 else ''}
""" + METHOD_MD)

    md("## Setup\n\nOn SageMaker, run this from the unzipped bundle (`<folder>/Training/`). On your computer, open it from the repository's `Training/` folder.")
    code(SETUP)
    code(INSTALL)
    code(r"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from simple_mappo import (NavalEnv, MAPPO, Monitor, CurriculumCallback, SaveOnIntervalCallback, BestModelCallback,
                          VideoCallback, TeamStorage, TeamSyncCallback, DEFAULT_CURRICULUM, continue_training,
                          training_status, evaluate, plot_results, plot_curriculum, plot_reward_terms, plot_progress,
                          show_videos)
""")

    md("## Team checkpoints\n\nUse the **same `RUN_NAME` in every notebook** (and as your teammates): that is the one line of training the stages hand over along. Only one person can train it at a time.")
    code(TEAM)

    md("""
## Configuration

The method's settings are **identical in every notebook**. They only shape a brand-new model; a continued model keeps the settings it was created with. `n_timesteps` is this session's training; rerun the notebook to continue.
""")
    code(f"""
STAGE = {k}
FORCE_STAGE = False                  # True: train this stage even if the run is on another one
KEEP_GOING = False                   # True: when this stage is passed, carry on into the next stages
                                     # in this same session instead of stopping (continuous training)

config = OrderedDict([
    ('n_envs', max(1, min(16, os.cpu_count() // 2))),     # battles in parallel, one Unity process each
    ('decision_period', 1.0),          # game seconds between decisions
    ('sim_dt', 0.08),                  # game seconds per simulated frame (fastest stable)
    ('n_timesteps', {st['session']:_}),          # this session's training
])
{MAPPO_SETTINGS}
save_interval = config['n_timesteps'] // 3

pd.Series(DEFAULT_CURRICULUM[STAGE], name=f"stage {{STAGE}}").to_frame()
""")

    md("## Environment\n\nLaunches `n_envs` headless Unity players. The observation has the same size in every stage (room for 8 ships a side and 5 circles), which is what lets one network play them all.")
    code(r"""
env = Monitor(NavalEnv(stages="default", start_stage=STAGE, n_envs=config['n_envs'],
                       decision_period=config['decision_period'], sim_dt=config['sim_dt'],
                       unity_log_dir=os.path.join(run_dir, "unity_logs")), log_dir)
print("battles in parallel:", env.n_envs, "| token sizes", env.dims)
""")

    md(f"""
## Continue the model from the previous stage

`continue_training` loads the run's latest checkpoint with all its training memory{', or - with no checkpoint yet - starts a new model from the imitation checkpoint (`models/imitation.pt`)' if k == 0 else ''}. If the run is on another stage it says which notebook to open.
""")
    code(r"""
model = continue_training(env, STAGE, run_dir, log_dir, storage=storage, force=FORCE_STAGE, **mappo_settings)
if storage:
    storage.lock()                   # stops here if a teammate is training this run right now
model.actor                          # the captain (and the commander), as in the Week 10 notebooks
""")

    md("""
## Baselines

What the trained fleet has to beat on this stage: random legal actions, and a simple hand-written tactic. `pass_rate` is the stage's own goal (see the table at the top). Set `RUN_BASELINES = False` to skip them in later sessions.
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
## Train

{'Training stops by itself when the held-out stopping rule is met' if last else 'Training stops by itself as soon as the promotion requirement is met (unless `KEEP_GOING`)'}, or after `n_timesteps`. `models/model_best.pt` always holds the checkpoint with the best rolling pass rate. The team folder gets a status update every 10,000 timesteps and a checkpoint every `save_interval`.

**Replays**: five times a session a battle is recorded and **shown right here, under the training table**, while training goes on. The files are in `runs_simple/<RUN_NAME>/videos/` (`.gif` on SageMaker, which has no ffmpeg - double-click one in the file browser; `.mp4` where ffmpeg exists), and in the team folder.

In the table: `win_rate` / `pass_rate` against the stage's own opponent{', `win_rate_self_play` in battles against itself (about 0.5 in mirror battles - both sides are the fleet), `elo` the fleet''s rating' if k >= 3 else ''}{', `orders_free` the share of the commander''s orders that leave the ship free' if k >= 4 else ''}.
""")
    code(r"""
curriculum = CurriculumCallback(stop_on_promotion=not KEEP_GOING)
callbacks = [SaveOnIntervalCallback(save_interval, models_dir),     # also refreshes model_latest.pt
             curriculum,
             BestModelCallback(models_dir),
             VideoCallback(every=config['n_timesteps'] // 5, video_dir=video_dir, show=True)]   # replays appear below
if storage:
    callbacks.append(TeamSyncCallback(storage, run_dir, push_every=save_interval, status_every=10_000))

model.learn(total_timesteps=model.num_timesteps + config['n_timesteps'], callback=callbacks, log_interval=5)
""")

    md("""
## Save and hand over

Saves the model and pushes it to the team. **If you stop training early (Kernel > Interrupt), run this cell too**, so your progress is kept and the run is free for the next person.
""")
    code(f"""
model.save(os.path.join(models_dir, "model_final"))
model.save(os.path.join(models_dir, "model_latest"))
if storage:
    storage.push(run_dir, model, status=training_status(model, run_dir))
    storage.release()

if env.stage > STAGE:
    print(f"Stage {{STAGE}} passed at {{model.num_timesteps:,}} timesteps - the model is on '{{env.stages[env.stage]['name']}}' now: "
          f"open the Stage{{env.stage}} notebook.")
elif curriculum.finished:
    print(f"The final stopping rule is met at {{model.num_timesteps:,}} timesteps - the curriculum is complete. "
          f"models/model_final.pt is the trained fleet (running this notebook again trains on to improve it).")
else:
    print(f"Stage {{STAGE}} not passed yet ({{model.num_timesteps:,}} timesteps) - run this notebook again to continue.")
""")

    md("## Progress on this stage")
    code(r"""
if storage:
    storage.show_status()
plot_results(log_dir, title=f"Stage {STAGE} learning curve", stage=STAGE)
plot_curriculum(log_dir)                     # win rate per stage, one colour each
plot_reward_terms(log_dir, stage=STAGE)      # what the fleet is being paid for on this stage
""")
    code(r"""
plot_progress(log_dir)                       # the PPO diagnostics
""")
    if k >= 4:
        code(r"""
# the commander: how often it leaves a ship free (1.0 = never gives orders) and how sure it is
progress = pd.read_csv(os.path.join(log_dir, "progress.csv"))
progress[progress.stage == STAGE][["total_timesteps", "orders_free", "commander_entropy", "win_rate", "pass_rate"]].tail(10)
""")
    if league:
        code(r"""
# the league: Elo ratings of the fleet, its past copies and the rule AI (chess-style; 400 points = 10:1 odds)
pd.DataFrame(model.league.table(), columns=["player", "elo", "our record against it"])
""")

    md(f"""
## Evaluation

Fresh battles on this stage against its own opponent, with sampled actions{', and on held-out maps (generated from seeds training never uses)' if last else ''}. The best checkpoint is evaluated too.
""")
    code(r"""
results = dict(baselines)
results["MAPPO (latest)"] = evaluate(model, env, n_battles=40, stage=STAGE)
best_path = os.path.join(models_dir, "model_best.pt")
if os.path.exists(best_path):
    stage_now = env.stage
    best = MAPPO.load(best_path, env=env)          # (loading moves the environment to the stage it was saved on)
    results["MAPPO (best)"] = evaluate(best, env, n_battles=40, stage=STAGE)
    env.set_stage(stage_now)
""" + (r"""results["MAPPO, held-out maps"] = evaluate(model, env, n_battles=40, stage=STAGE, held_out=True)
""" if last else "") + r"""pd.DataFrame(results).T[["battles", "win_rate", "pass_rate", "win_rate_stderr", "mean_reward", "enemy_ships_sunk", "own_ships_lost"]]
""")

    md("""
## Put the fleet into your Unity game

Exports the best checkpoint (captains + fleet commander) as the game's policy file. The game runs it in plain C# (`Assets/Scripts/RL/RLPolicy.cs`), no Python needed. From the repository it goes straight into `Assets/StreamingAssets/RL/naval_policy.bin` (the old file is kept as `.bak`); elsewhere (SageMaker) it goes into the run folder - download it (right-click > Download) and copy it there.

Then in Unity: **Play**, and on the setup screen set **ENEMY AI: LEARNED** (fight against it) or **YOUR FLEET: LEARNED** (watch it fight). The game reads the file when it starts, so press Play again after exporting. From a computer later: `python export_simple.py` (or `--team-storage s3://... --run-name fleet` to fetch a teammate's run first).
""")
    code(r"""
from simple_mappo.export import export_policy
checkpoint = os.path.join(models_dir, "model_best.pt" if os.path.exists(os.path.join(models_dir, "model_best.pt")) else "model_latest.pt")
in_repo = os.path.isdir(os.path.join("..", "Assets", "StreamingAssets"))
target = os.path.join("..", "Assets", "StreamingAssets", "RL", "naval_policy.bin") if in_repo else os.path.join(run_dir, "naval_policy.bin")
print("exported", checkpoint, "->", os.path.abspath(export_policy(checkpoint, target)))
""")

    md("""
## Replays from this stage

The newest replays recorded while training on this stage. They are a 2D map drawn from the game's data (the training player has no graphics); to see the model in the real game's graphics, run `python watch_simple.py --model <model_final.pt>` on a computer with a screen.
""")
    code(r"""
show_videos(video_dir, contains=f"_stage{STAGE}", newest=3)
""")
    code(r"""
env.close()
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
3. **Hand over** - the cloned captain is saved as `models/imitation.pt`. The Stage 0 notebook starts the RL model from it, and a KL penalty that fades out keeps the captain close to the clone at first (AlphaStar's KL to the supervised policy).

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


def _finish(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    keep = {IMITATION_FILE} | {st["file"] for st in STAGES}
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
