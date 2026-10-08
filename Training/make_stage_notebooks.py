"""Writes the five Stage<k> notebooks - one per curriculum stage, all training the SAME model.

    python make_stage_notebooks.py

Edit the stage descriptions here and rerun, rather than editing the five notebooks by hand. The
stages themselves (scenarios, opponents, promotion rules) live in simple_mappo/curriculum.py.
"""

from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    dict(file="Stage0-Battleship-Duel.ipynb", name="Battleship Duel", scenario="bb_duel", fleet="1v1 battleships",
         environment="Open-sea duel scenario", opponent="Recruit rule-based AI",
         rule="≥55% rolling win rate after at least 100 qualifying episodes", session=500_000,
         learns="""Two battleships start 13 km apart on open sea, both spotted and inside gun range from the first second, with 3 minutes on the clock (plus overtime - there are no draws). The duel teaches the basics every later stage builds on: **choosing and timing fire, angling the hull, and not throwing the ship away**.

This stage does **not** pay for the capture circle: it lies between the two battleships, so racing to it means sailing bow-on into the enemy's guns (measured: sailing to the circle wins 33% of duels, broadside and fire at will 70%). Every other stage uses the full reward.""",
         measured="Measured with this setup: random play wins 16-28%, the broadside script 70%, and MAPPO reached 59% after 300,000 timesteps.",
         script="broadside"),
    dict(file="Stage1-King-of-the-Hill.ipynb", name="King of the Hill", scenario="koth_3v3", fleet="3v3 mixed fleet",
         environment="Single-objective combat scenario", opponent="Veteran rule-based AI",
         rule="≥65% rolling win rate after at least 100 qualifying episodes", session=500_000,
         learns="""A battleship, a cruiser and a destroyer on each side, one capture circle (1.9 km radius) in the middle of the open sea, no cover, 6 minutes. Holding the circle scores points every second; first to 1000 points or the last fleet afloat wins.

This is the first **multi-agent** stage: three ships share one actor network and the team's reward, and have to learn to **sail to the circle and hold it while fighting** - the destroyer can reach the ring in about two minutes, the battleship covers it.""",
         measured="Measured with this setup, trained from scratch: random play wins about 18% against Veteran, MAPPO 25% after 400,000 timesteps - starting from the duel-trained model should do better.",
         script="circle"),
    dict(file="Stage2-Archipelago.ipynb", name="Archipelago", scenario="archipelago_3v3", fleet="3v3 mixed fleet",
         environment="Islands / archipelago scenario", opponent="Veteran rule-based AI",
         rule="≥65% rolling win rate after at least 150 qualifying episodes", session=500_000,
         learns="""The same three ships a side, now among islands with three capture circles (A, B and C, 1.3 km radius each) and 8 minutes. Islands break line of sight, so **spotting, smoke, radar and cover** start to matter: a ship that is not seen cannot be shot.""",
         measured="",
         script="circle"),
    dict(file="Stage3-Full-Domination.ipynb", name="Full Domination", scenario="domination_6v6", fleet="6v6 fleet",
         environment="Procedural battlefield with random weather", opponent="Elite rule-based AI + self-play",
         rule="≥60% rolling win rate after at least 200 qualifying Elite rule-AI episodes", session=300_000,
         learns="""The full Domination game at 6v6, 15 minutes, on a **newly generated battlefield** every 8 battles: a random preset (archipelago, open sea or strait), island density and weather. The opponent is the best rule-based AI (Elite) - and in half the battles a **frozen copy of our own network** (self-play), refreshed every 10 updates, so the fleet also learns to beat a smarter, changing opponent.

Only battles against the Elite AI count towards promotion; the self-play battles are training only.""",
         measured="",
         script="circle"),
    dict(file="Stage4-Open-Generalisation.ipynb", name="Open / Generalisation", scenario="open", fleet="Random 4–8 ships per side",
         environment="Random map preset, terrain density and weather", opponent="Elite rule-based AI + self-play",
         rule="Final stage: ≥60% rolling win rate after at least 250 qualifying held-out evaluation episodes against the Elite rule-based AI",
         session=300_000,
         learns="""Anything the game can produce: 4 to 8 ships a side, a random map preset, terrain density and weather - and also a random game mode (Domination, Skirmish, Fleet Battle, Capture and Control) and circle size - 20 minutes, against the Elite AI and self-play.

There is **no further promotion**. Every 100,000 timesteps the fleet plays 50 **held-out** battles against the Elite AI on generated maps whose seeds training never uses (a test set, across many evaluation seeds). Training stops once the last 250 held-out battles reach 60%.""",
         measured="",
         script="circle"),
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


def notebook(k: int) -> nbf.NotebookNode:
    st = STAGES[k]
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    nav = " · ".join(f"**Stage {i}**" if i == k else f"[Stage {i}]({s['file']})" for i, s in enumerate(STAGES))
    last = k == len(STAGES) - 1

    md(f"""
# 43008: Reinforcement Learning - WOW Naval MAPPO

## Stage {k}: {st['name']}

{nav} · [the full walkthrough](WOW-MAPPO-Simple.ipynb)

| Stage | Training scenario | Fleet | Environment | Opponent | {'Stopping' if last else 'Promotion'} requirement |
|---|---|---|---|---|---|
| {k} - {st['name']} | `{st['scenario']}` | {st['fleet']} | {st['environment']} | {st['opponent']} | {st['rule']} |

### What the fleet learns here

{st['learns']}

{st['measured']}

### One model through every stage

All five stage notebooks train the **same model with the same method** (MAPPO, the same network and hyperparameters). This notebook continues the run's latest checkpoint, which carries everything learned so far - the network weights, the optimiser's state, the value normaliser, the timestep count and the stage - and the run's battle log, from which the promotion window continues where the last session stopped.

* Not passed yet? **Run this notebook again** - each session continues the last one (yours or a teammate's).
* {'This is the final stage: training ends when the held-out stopping rule is met.' if last else f"Passed? Training stops and saves, and the model moves to stage {k + 1}: open [Stage {k + 1}]({STAGES[k + 1]['file']})."}

How MAPPO, the environment, the reward and the team checkpoints work is explained in [the full walkthrough](WOW-MAPPO-Simple.ipynb).
""")

    md("## Setup\n\nOn SageMaker, run this from the unzipped bundle (`<folder>/Training/`). On your computer, open it from the repository's `Training/` folder.")
    code(SETUP)
    code(r"""
# install only what is missing (PyTorch as the CPU build)
import importlib.util
missing = [pkg for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("PIL", "pillow"))
           if importlib.util.find_spec(mod) is None]
if missing:
    !pip install -q {" ".join(missing)}
if importlib.util.find_spec("torch") is None:
    !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
""")
    code(r"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from simple_mappo import (NavalEnv, MAPPO, Monitor, CurriculumCallback, SaveOnIntervalCallback, VideoCallback,
                          TeamStorage, TeamSyncCallback, DEFAULT_CURRICULUM, REWARD_WEIGHTS, continue_training,
                          training_status, evaluate, plot_results, plot_curriculum, plot_reward_terms, plot_progress,
                          show_videos)
""")

    md("""
## Team checkpoints

Use the **same `RUN_NAME` in every stage notebook** (and as your teammates): that is the one line of training the stages hand over along. Only one person can train it at a time.
""")
    code(r"""
YOUR_NAME = "your-name"              # shown to your teammates
RUN_NAME = "curriculum"              # the same in all five stage notebooks
if ON_SAGEMAKER:
    from simple_mappo.aws import team_storage_uri
    TEAM_STORAGE = team_storage_uri()     # s3://sagemaker-<region>-<account>/wow-mappo, or "s3://<team-bucket>/wow-mappo"
else:
    TEAM_STORAGE = None              # e.g. "s3://<team-bucket>/wow-mappo"; None trains on this machine only

storage = TeamStorage(TEAM_STORAGE, RUN_NAME, user=YOUR_NAME) if TEAM_STORAGE else None
run_dir = f"runs_simple/{RUN_NAME}"
log_dir, models_dir, video_dir = (os.path.join(run_dir, d) for d in ("logs", "models", "videos"))
if storage:
    storage.show_status()
""")

    md(f"""
## Configuration

The MAPPO settings are **identical in every stage notebook** - one method throughout. They only shape a brand-new model (stage 0); a continued model keeps the settings it was created with. `n_timesteps` is this session's training; rerun the notebook to continue.
""")
    code(f"""
STAGE = {k}
FORCE_STAGE = False                  # True: train this stage even if the run is on another one

config = OrderedDict([
    ('n_envs', max(1, min(16, os.cpu_count() // 2))),     # battles in parallel, one Unity process each
    ('decision_period', 1.0),          # game seconds between decisions
    ('sim_dt', 0.08),                  # game seconds per simulated frame (fastest stable)
    ('n_timesteps', {st['session']:_}),          # this session's training
])
# the method - the same in all five notebooks
mappo_settings = dict(learning_rate=3e-4, n_steps=256, n_minibatches=4, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                      clip_range=0.2, ent_coef=0.01, vf_coef=0.5, hidden_size=256, opponent_refresh=10)
save_interval = config['n_timesteps'] // 3

pd.Series(DEFAULT_CURRICULUM[STAGE], name=f"stage {{STAGE}}").to_frame()
""")

    md("## Environment\n\nLaunches `n_envs` headless Unity players. The observation has the same size in every stage (room for 8 ships a side and 5 circles), which is what lets one network play them all.")
    code(r"""
env = Monitor(NavalEnv(stages="default", start_stage=STAGE, n_envs=config['n_envs'],
                       decision_period=config['decision_period'], sim_dt=config['sim_dt'],
                       unity_log_dir=os.path.join(run_dir, "unity_logs")), log_dir)
print("battles in parallel:", env.n_envs, "| actor input", env.local_dim, "| critic input", env.state_dim)
""")

    md(f"""
## Continue the model from the previous stage

`continue_training` loads the run's latest checkpoint with all its training memory{', or starts a new model if there is none yet' if k == 0 else ''}. If the run is on another stage it says which notebook to open.
""")
    code(r"""
model = continue_training(env, STAGE, run_dir, log_dir, storage=storage, force=FORCE_STAGE, **mappo_settings)
if storage:
    storage.lock()                   # stops here if a teammate is training this run right now
""")

    md(f"""
## Baselines

What the trained fleet has to beat on this stage: random legal actions, and a simple hand-written tactic. (Set `RUN_BASELINES = False` to skip them in later sessions.)
""")
    code(SCRIPTS[st['script']] + r"""

RUN_BASELINES = True
baselines = {}
if RUN_BASELINES:
    baselines = {"random": evaluate(None, env, n_battles=20, stage=STAGE),
                 SCRIPT_NAME: evaluate(Script(), env, n_battles=20, stage=STAGE)}
    display(pd.DataFrame(baselines).T[["stage", "battles", "win_rate", "win_rate_stderr", "mean_reward"]])
""")

    md(f"""
## Train

{'Training stops by itself when the held-out stopping rule is met' if last else 'Training stops by itself as soon as the promotion requirement is met'}, or after `n_timesteps`. A replay video is recorded now and then, and the team folder gets a status update every 10,000 timesteps and a checkpoint every `save_interval`.
""")
    code(r"""
curriculum = CurriculumCallback(stop_on_promotion=True)
callbacks = [SaveOnIntervalCallback(save_interval, models_dir),
             curriculum,
             VideoCallback(every=config['n_timesteps'] // 5, video_dir=video_dir)]
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
plot_curriculum(log_dir)                     # win rate against the rule AI, one colour per stage
plot_reward_terms(log_dir, stage=STAGE)      # what the fleet is being paid for on this stage
""")
    code(r"""
plot_progress(log_dir)                       # the PPO diagnostics
""")

    md(f"""
## Evaluation

Fresh battles on this stage against the rule AI, with sampled actions{', and on held-out maps (generated from seeds training never uses)' if last else ''}.
""")
    code(r"""
results = dict(baselines)
results["MAPPO"] = evaluate(model, env, n_battles=40, stage=STAGE)
""" + (r"""results["MAPPO, held-out maps"] = evaluate(model, env, n_battles=40, stage=STAGE, held_out=True)
""" if last else "") + r"""pd.DataFrame(results).T[["battles", "win_rate", "win_rate_stderr", "mean_reward", "enemy_ships_sunk", "own_ships_lost"]]
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
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    for k, st in enumerate(STAGES):
        path = os.path.join(HERE, st["file"])
        nbf.write(notebook(k), path)
        print("wrote", path)
