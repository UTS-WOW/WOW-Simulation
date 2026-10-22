"""Logging, plots, evaluation and battle replays - the helpers the Week 9/10 notebooks use, for the
naval environment.

    env = Monitor(NavalEnv(...), log_dir)      # one CSV line per finished battle
    plot_results(log_dir)                      # the learning curve
    plot_curriculum(log_dir)                   # pass rate per curriculum stage
    evaluate(model, env, n_battles=40)         # win rate against the rule AI (model=None: random play)
    record_battle(model, env, "videos/x.mp4")  # a top-down replay of one battle (.gif without ffmpeg)
    VideoCallback(every=100_000, video_dir=...)  # a replay every so often while training
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
import contextlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .callbacks import BaseCallback
from .rewards import REWARD_WEIGHTS

REASON_KEYS = ["player_score", "enemy_score", "player_kills", "enemy_kills", "damage_dealt_hulls",
               "enemy_damage_dealt_hulls", "first_capture_time"]


# ---------------------------------------------------------------------------------------------- Monitor

class Monitor:
    """Wraps a NavalEnv and writes <log_dir>/monitor.csv, one line per finished battle.

    Columns: r (average reward per ship), l (decisions), t (seconds since start), won, battle_time,
    the game's own scores and kills, and term_<name>: what each reward term paid per ship.

    Two Monitors can share one NavalEnv - one Unity player per battle for both methods - each with its
    own log and its own curriculum stage (MAPPO and Double DQN climb the stages separately):

        battles = NavalEnv(stages="default", n_envs=8)
        env, env_dqn = Monitor(battles, "runs_simple/fleet/logs"), Monitor(battles, "runs_simple/fleet_ddqn/logs")
    """

    def __init__(self, env, log_dir: str):
        self.env = env
        self._stage = env.stage
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "monitor.csv")
        self.terms = list(env.reward_weights)
        self.columns = (["r", "l", "t", "won", "passed", "stage", "opponent", "battle_time", "survived"] + REASON_KEYS
                        + ["term_" + k for k in self.terms] + ["reason"])
        self._fit_header()

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def stage(self) -> int:
        return self._stage

    def set_stage(self, stage: int) -> None:
        self.env.set_stage(stage)
        self._stage = self.env.stage

    def _own_stage(self) -> None:
        if self.env.stage != self._stage:             # another Monitor of the same battles moved it
            self.env.set_stage(self._stage)

    def reset(self):
        self._own_stage()
        return self.env.reset()

    def step(self, actions, orders=None):
        self._own_stage()
        obs, rewards, dones, infos = self.env.step(actions, orders)
        rows = [self._row(i) for i in infos if i]
        if rows:
            self._fit_header()
            with open(self.path, "a") as f:
                f.writelines(rows)
        return obs, rewards, dones, infos

    def _fit_header(self) -> None:
        """Makes sure monitor.csv has this Monitor's columns before rows are added. A log started by an
        older version (fewer reward terms - e.g. before "circles_held" existed), or downloaded again
        from the team folder, is rewritten in the new layout, every old row kept and matched by column
        name (a term the old rows did not have stays empty)."""
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write("#" + json.dumps({"t_start": time.time(), "env_id": "WOW naval MAPPO"}) + "\n")
                f.write(",".join(self.columns) + "\n")
            return
        with open(self.path) as f:
            f.readline()
            header = f.readline().strip().split(",")
        if header != self.columns:
            migrate_monitor(self.path, self.columns)

    def _row(self, info: dict) -> str:
        ep, stats, terms = info["episode"], info.get("stats", {}), info.get("reward_terms", {})
        # "rule" / "passive" / "mirror" / "latest" / "past" / "champion" / "rival": training battles (who flew the enemy);
        # "eval": evaluations; "eval-heldout": held-out evaluations
        kind = ("eval-heldout" if info.get("held_out") else "eval") if info.get("evaluation") else info.get("opponent", "rule")
        values = [ep["r"], ep["l"], ep["t"], int(info["won"]), int(info.get("passed", info["won"])),
                  info.get("stage", 0), kind, info["battle_time"],
                  info.get("survived", "")]
        values += [stats.get(k, "") for k in REASON_KEYS]
        values += [terms.get(k, 0.0) for k in self.terms]
        values += ['"' + str(info.get("reason", "")).replace('"', "'") + '"']
        return ",".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in values) + "\n"


def _monitor_rows(path: str, columns: list[str] | None = None):
    """(first line, header, rows as dicts) of a monitor.csv - also one that mixes two layouts: rows
    written before a reward term was added have as many fields as the header, rows written after as
    many as the current Monitor's columns (default: every term in rewards.REWARD_WEIGHTS)."""
    if columns is None:
        columns = (["r", "l", "t", "won", "passed", "stage", "opponent", "battle_time", "survived"] + REASON_KEYS
                   + ["term_" + k for k in REWARD_WEIGHTS] + ["reason"])
    with open(path, newline="") as f:
        first = f.readline()
        table = list(csv.reader(f))
    if not table:
        return first, [], []
    header, rows = table[0], table[1:]
    out = []
    for row in rows:
        names = header if len(row) == len(header) else columns if len(row) == len(columns) else header
        out.append(dict(zip(names, row)))
    return first, header, out


def migrate_monitor(path: str, columns: list[str]) -> None:
    """Rewrites a monitor.csv in the layout `columns`, keeping every row (see Monitor._fit_header)."""
    first, header, rows = _monitor_rows(path, columns)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.write(first)
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows([[r.get(c, "") for c in columns] for r in rows])
    os.replace(tmp, path)


def load_results(log_dir: str, training_only: bool = True, stage: int | None = None):
    """monitor.csv as a DataFrame. opponent is "rule" (the rule AI), "self" (self-play), "eval" or
    "eval-heldout" (evaluation battles); training_only drops the evaluation battles, stage keeps one
    curriculum stage."""
    import pandas as pd
    path = os.path.join(log_dir, "monitor.csv")
    try:
        df = pd.read_csv(path, skiprows=1)
    except pd.errors.ParserError:                     # rows of two layouts (an older log): read them by name
        _, _, rows = _monitor_rows(path)
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col not in ("opponent", "reason"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if training_only and "opponent" in df:
        df = df[~df.opponent.astype(str).str.startswith("eval")]
    if stage is not None and "stage" in df:
        df = df[df.stage == stage]
    return df.reset_index(drop=True)


def ts2xy(df, x_axis: str = "timesteps"):
    """x: cumulative timesteps (or episodes / walltime_hrs), y: reward per battle."""
    if x_axis == "timesteps":
        x = np.cumsum(df.l.values)
    elif x_axis == "episodes":
        x = np.arange(len(df))
    else:
        x = df.t.values / 3600.0
    return x, df.r.values


# ---------------------------------------------------------------------------------------------- plots

def _no_battles(df, what: str) -> bool:
    """True (and says so) when there is nothing to plot yet - e.g. a session too short to finish a battle."""
    if len(df):
        return False
    print(f"{what}: no finished battles to plot yet")
    return True


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(y)))
    return np.convolve(y, np.ones(window) / window, mode="valid")


def plot_results(log_folder: str, title: str = "Learning Curve", window: int = 50, stage: int | None = None):
    """The notebooks' learning curve: reward per battle against timesteps, smoothed (one stage's
    battles only, if stage is given)."""
    import matplotlib.pyplot as plt
    df = load_results(log_folder, stage=stage)
    if _no_battles(df, title):
        return
    x, y = ts2xy(df, "timesteps")
    y = _smooth(y, window)
    x = x[len(x) - len(y):]
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


def plot_curriculum(log_folder: str, window: int = 50, name: str = ""):
    """Pass rate over training (each stage's own goal - usually a win), one colour per curriculum stage
    (a new colour = the run moved on). Only qualifying battles: against the stage's own opponent.
    Solid while the run trains ON that stage; dotted afterwards - rehearsal battles of an earlier stage
    (10-15 % of the battles from stage 3 on), which never decide anything. name: the run, for the title."""
    import matplotlib.pyplot as plt
    df = load_results(log_folder)
    if _no_battles(df, "Curriculum"):
        return
    x = np.cumsum(df.l.values)
    rule = df.opponent.astype(str).isin(["rule", "passive"]).values if "opponent" in df else np.ones(len(df), bool)
    passed = df.passed.values if "passed" in df else df.won.values
    furthest = np.maximum.accumulate(df.stage.values)       # the furthest stage reached by each battle
    plt.figure(figsize=(10, 4))
    for st in sorted(df.stage.unique()):
        part = (df.stage.values == st) & rule                # what counts: the stage's own opponent
        y = passed[part].astype(float)
        if not len(y):
            continue
        w = max(1, min(window, len(y)))
        xs, ys = x[part][w - 1:], np.convolve(y, np.ones(w) / w, mode="valid")
        on = furthest[part][w - 1:] == st                    # trained on it, not a rehearsal of it
        line, = plt.plot(np.where(on, xs, np.nan), np.where(on, ys, np.nan), label=f"stage {st}")
        if (~on).any():
            plt.plot(np.where(~on, xs, np.nan), np.where(~on, ys, np.nan), ls=":", lw=0.9, color=line.get_color())
    plt.ylim(0, 1)
    plt.xlabel("Number of Timesteps")
    plt.axhline(0.7, color="gray", lw=0.8, ls="--")      # the promotion requirement
    plt.ylabel(f"Pass rate (last {window} battles)")
    plt.title(f"{name + ' - ' if name else ''}curriculum: pass rate per stage (furthest stage reached: "
              f"{int(furthest[-1])}; dotted = rehearsal battles)")
    plt.legend()
    plt.show()


def plot_reward_terms(log_folder: str, window: int = 50, stage: int | None = None, name: str = ""):
    """What the fleet is actually being paid for: each reward term per battle, smoothed. name: the run."""
    import matplotlib.pyplot as plt
    df = load_results(log_folder, stage=stage)
    if _no_battles(df, "Reward terms"):
        return
    x = np.cumsum(df.l.values)
    plt.figure(figsize=(10, 5))
    for col in [c for c in df.columns if c.startswith("term_")]:
        y = _smooth(df[col].values.astype(float), window)
        plt.plot(x[len(x) - len(y):], y, label=col[5:])
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Reward per ship per battle")
    plt.title(f"{name + ' - ' if name else ''}reward terms" + (f" on stage {stage}" if stage is not None else ""))
    plt.legend(ncol=3, fontsize=8)
    plt.show()


def plot_progress(log_folder: str):
    """The training diagnostics from progress.csv, one row per update: PPO's for a MAPPO run, the
    Q-learning ones (TD loss, Q-values, epsilon, replay size) for a Double DQN run."""
    import matplotlib.pyplot as plt
    import pandas as pd
    path = os.path.join(log_folder, "progress.csv")
    if not os.path.exists(path):
        print("diagnostics: no updates logged yet")
        return
    df = pd.read_csv(path)
    if "q_loss" in df:
        cols, title = ["q_loss", "q_value", "target_value", "epsilon", "commander_loss", "replay_decisions"], "Double DQN"
    else:
        cols, title = ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"], "MAPPO (PPO)"
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, c in zip(axes.ravel(), cols):
        if c in df and df[c].notna().any():
            ax.plot(df.total_timesteps, df[c])
        ax.set_title(c)
        ax.set_xlabel("timesteps")
    fig.suptitle(f"{title} diagnostics")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------- comparing methods

def _stage_battles(log_dir: str, stage: int):
    """A run's training battles on one stage, with the timesteps it had spent on the stage after each."""
    path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(path):
        return None
    df = load_results(log_dir, stage=stage)
    df["on_stage"] = np.cumsum(df.l.values) if len(df) else []
    return df


def plot_comparison(runs: dict, stage: int, window: int = 50):
    """Two methods (or more) on one stage: {"MAPPO": its log_dir, "Double DQN": its log_dir}.
    Left: the pass rate against the stage's own opponent (what promotion counts); right: the reward
    per battle. x: timesteps each run spent on this stage - a fair race from the same starting line."""
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, log_dir in runs.items():
        df = _stage_battles(log_dir, stage)
        if df is None or not len(df):
            print(f"{name}: no battles on stage {stage} yet")
            continue
        gate = df[df.opponent.astype(str).isin(["rule", "passive"])]
        if len(gate):
            w = max(1, min(window, len(gate)))
            y = np.convolve(gate.passed.values.astype(float), np.ones(w) / w, mode="valid")
            left.plot(gate.on_stage.values[w - 1:], y, label=name)
        y = _smooth(df.r.values, window)
        right.plot(df.on_stage.values[len(df) - len(y):], y, label=name)
    left.axhline(0.7, color="gray", lw=0.8, ls="--")
    left.set_ylim(0, 1)
    left.set_title(f"Stage {stage}: pass rate (last {window} battles vs its own opponent)")
    right.set_title(f"Stage {stage}: reward per ship per battle (smoothed)")
    for ax in (left, right):
        ax.set_xlabel("Timesteps on this stage")
        ax.legend()
    plt.tight_layout()
    plt.show()


def compare_stages(runs: dict, stages: list[dict], window: int = 50):
    """Every stage, every method, side by side (from the runs' monitor.csv): how long each needed on the
    stage, how it was doing at the end, and whether it passed - "no - moved on" when it reached a later
    stage without meeting the bar (the stage's max_timesteps, or FORCE_STAGE)."""
    import pandas as pd
    rows = []
    events = {}                                  # logs/curriculum.csv: how each run left each stage
    for name, log_dir in runs.items():
        path = os.path.join(log_dir, "curriculum.csv")
        if os.path.exists(path):
            events[name] = dict(zip(*[pd.read_csv(path)[c].tolist() for c in ("stage", "event")]))   # the latest
    for k, st in enumerate(stages):
        row = {"stage": st["name"]}
        for name, log_dir in runs.items():
            path = os.path.join(log_dir, "monitor.csv")
            if not os.path.exists(path):
                continue
            everything = load_results(log_dir)
            df = everything[everything.stage == k]
            later = (everything.stage > k).values
            first_later = int(np.argmax(later)) if later.any() else len(everything)
            before = df[df.index < first_later]                         # until promotion (no rehearsals after it)
            gate = before[before.opponent.astype(str).isin(["rule", "passive"])]
            row[f"{name}: timesteps"] = int(before.l.sum()) if len(before) else None     # None: before this log
            row[f"{name}: pass rate"] = round(float(gate.passed.tail(window).mean()), 2) if len(gate) else None
            if k in events.get(name, {}):        # the curriculum's own record
                row[f"{name}: passed"] = {"passed": "yes", "not passed": "no - cap reached" if k == len(stages) - 1
                                          else "no - moved on"}.get(events[name][k], "training again")
            elif later.any():                    # moved on: by passing, or after its max_timesteps / FORCE_STAGE
                need, n = st.get("promote_win_rate"), int(st.get("window", window))
                met = need is None or (len(gate) >= n and float(gate.passed.tail(n).mean()) >= need)
                row[f"{name}: passed"] = "yes" if met else "no - moved on"
            else:
                row[f"{name}: passed"] = "training" if len(df) else "-"
        rows.append(row)
    return pd.DataFrame(rows).set_index("stage")


# ---------------------------------------------------------------------------------------------- evaluation

def evaluate(model, env, n_battles: int = 40, deterministic: bool = False, stage: int | None = None,
             held_out: bool = False, keep_results: bool = False) -> dict:
    """Plays fresh battles against the rule AI (never self-play). model=None plays random legal actions.

    stage: which curriculum stage to play (default: the current one).
    held_out: generated maps from seeds training never uses (a test set for the procedural stages).
    keep_results: include every battle's result under "results".
    Sampled actions (deterministic=False) are the fair test: a young policy is often near 50/50 on
    options like fire / hold, and always taking the likelier one can mean never firing.
    """
    previous = env.stage
    if stage is not None:
        env.set_stage(stage)
    try:
        with as_evaluation(env, held_out):
            out = _evaluate(model, env, n_battles, deterministic)
    finally:
        env.set_stage(previous)
    if not keep_results:
        out.pop("results")
    return out


@contextmanager
def as_evaluation(env, held_out: bool = False):
    """Battles played inside are evaluations: against the rule AI (never self-play), and logged as
    "eval" in monitor.csv, so they never count towards a promotion. Replays use it too."""
    was = env.evaluating, env.held_out
    env.set_evaluation(True, held_out)
    try:
        yield
    finally:
        env.set_evaluation(*was)


def _evaluate(model, env, n_battles: int, deterministic: bool) -> dict:
    obs = env.reset()
    E = env.n_envs
    quota = math.ceil(n_battles / E)        # the same number per battle slot, so short battles are not over-counted
    done_per_env = np.zeros(E, dtype=int)
    results = []
    while done_per_env.min() < quota:
        actions = env.sample_random_actions(obs) if model is None else model.predict(obs, deterministic)
        obs, _, dones, infos = env.step(actions)
        for e in np.flatnonzero(dones[:E]):                # battles (our fleet's rows)
            if done_per_env[e] < quota:
                results.append(infos[e])
            done_per_env[e] += 1
    won = np.array([r["won"] for r in results], dtype=float)
    passed = np.array([r.get("passed", r["won"]) for r in results], dtype=float)
    out = {
        "stage": env.stages[env.stage]["name"],
        "battles": len(results),
        "win_rate": float(won.mean()),
        "win_rate_stderr": float(won.std(ddof=1) / np.sqrt(len(won))) if len(won) > 1 else 0.0,
        "pass_rate": float(passed.mean()),             # the stage's own goal (curriculum.py "pass_if")
        "mean_reward": float(np.mean([r["episode"]["r"] for r in results])),
        "mean_battle_time": float(np.mean([r["battle_time"] for r in results])),
    }
    for k, name in (("player_kills", "enemy_ships_sunk"), ("enemy_kills", "own_ships_lost"),
                    ("player_score", "points"), ("enemy_score", "enemy_points")):
        vals = [r.get("stats", {}).get(k) for r in results if r.get("stats", {}).get(k) is not None]
        if vals:
            out[name] = float(np.mean(vals))
    reasons = {}
    for r in results:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    out["end_reasons"] = reasons
    out["results"] = results
    return out


# ---------------------------------------------------------------------------------------------- replays

def _cols(env, group: str) -> dict:
    names = env.spec.features.get(group)
    if not names:
        raise RuntimeError("replays need the real Unity environment (the mock has no feature names)")
    return {n: i for i, n in enumerate(names)}


def draw_battle(env, arrays: dict, ax, title: str = "", trails: list | None = None):
    """Top-down picture of one battle from the critic's (true) view, in the learning fleet's frame:
    its base is to the south. Cyan = learning fleet, red = rule AI."""
    from matplotlib.patches import Circle, Polygon
    own_c, enemy_c, zone_c = _cols(env, "critic_own"), _cols(env, "critic_enemy"), _cols(env, "critic_zone")
    obst_c = _cols(env, "obstacle")
    half = 2000.0                                          # positions are divided by half the map width
    own, enemy = arrays["critic_own"][0], arrays["critic_enemy"][0]
    zones, zmask = arrays["critic_zones"][0], arrays["critic_zone_mask"][0]
    pts = []

    ax.clear()
    ax.set_facecolor("#0b2740")
    # islands, rocks and smoke the learning fleet can see (ship-relative tokens turned into map positions)
    for i in range(own.shape[0]):
        if own[i, own_c["alive"]] < 0.5:
            continue
        x, y = own[i, own_c["x"]], own[i, own_c["y"]]
        hs, hc = own[i, own_c["heading_sin"]], own[i, own_c["heading_cos"]]
        for j in np.flatnonzero(arrays["obstacle_mask"][0, i] > 0.5):
            ob = arrays["obstacles"][0, i, j]
            rx, ry = ob[obst_c["rel_x"]] * 1000 / half, ob[obst_c["rel_y"]] * 1000 / half
            cx, cy = x + rx * hc + ry * hs, y - rx * hs + ry * hc
            smoke = ob[obst_c["is_smoke"]] > 0.5
            ax.add_patch(Circle((cx, cy), ob[obst_c["radius"]] * 200 / half, color="#9aa5ad" if smoke else "#6b5b3e",
                                alpha=0.5 if smoke else 0.9, lw=0))
    for z in np.flatnonzero(zmask > 0.5):
        zx, zy, r = zones[z, zone_c["x"]], zones[z, zone_c["y"]], zones[z, zone_c["radius"]] * 200 / half
        col = "#35d0e0" if zones[z, zone_c["owner_mine"]] > 0.5 else "#e04848" if zones[z, zone_c["owner_theirs"]] > 0.5 else "#d0d0d0"
        ax.add_patch(Circle((zx, zy), r, fill=False, ec=col, lw=2, ls="--"))
        ax.plot(zx, zy, "+", color=col, ms=10)
        pts.append((zx - r, zy - r)), pts.append((zx + r, zy + r))
    if trails:
        for tr in trails:
            if len(tr) > 1:
                t = np.array(tr)
                ax.plot(t[:, 0], t[:, 1], color=tr.color, lw=1, alpha=0.4)

    def ship(x, y, hs, hc, hp, alive, color, size):
        pts.append((x, y))
        if not alive:
            ax.plot(x, y, "x", color="#777777", ms=8, mew=2)
            return
        f = np.array([hs, hc]) * size
        s = np.array([hc, -hs]) * size * 0.45
        p = np.array([x, y])
        ax.add_patch(Polygon([p + f, p - f + s, p - f - s], closed=True, color=color))
        ax.plot([x - size, x - size + 2 * size * hp], [y - 1.8 * size] * 2, color="#7CFC00" if hp > 0.5 else "#ffb000", lw=2)

    size = 0.012
    for i in range(own.shape[0]):
        if own[i].any():
            ship(own[i, own_c["x"]], own[i, own_c["y"]], own[i, own_c["heading_sin"]], own[i, own_c["heading_cos"]],
                 own[i, own_c["hp"]], own[i, own_c["alive"]] > 0.5, "#35d0e0", size * (1.4 if own[i, own_c["cls_bb"]] > 0.5 else 1.0))
    for i in np.flatnonzero(arrays["critic_enemy_mask"][0] > 0.5):
        ship(enemy[i, enemy_c["true_x"]], enemy[i, enemy_c["true_y"]], enemy[i, enemy_c["true_heading_sin"]],
             enemy[i, enemy_c["true_heading_cos"]], enemy[i, enemy_c["true_hp"]], enemy[i, enemy_c["alive"]] > 0.5,
             "#e04848", size * (1.4 if enemy[i, enemy_c["cls_bb"]] > 0.5 else 1.0))

    pts = np.array(pts) if pts else np.zeros((1, 2))
    c = (pts.min(0) + pts.max(0)) / 2
    span = max(0.25, (pts.max(0) - pts.min(0)).max() * 0.65)
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


class _Trail(list):
    def __init__(self, color):
        super().__init__()
        self.color = color


def ffmpeg_path() -> str | None:
    """ffmpeg for .mp4 videos: on the PATH, or the copy that comes with the imageio-ffmpeg package."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def video_ext() -> str:
    """".mp4" where ffmpeg is available, otherwise ".gif" (Pillow writes those anywhere)."""
    return ".mp4" if ffmpeg_path() else ".gif"


GIF_HEIGHT, GIF_COLORS = 400, 64                       # the replays the notebooks show


def save_gif(frames: list, path: str, fps: float = 10, height: int = GIF_HEIGHT, colors: int = GIF_COLORS) -> str:
    """Writes PIL frames as a compact animated GIF - the replay the notebooks show, because an image is
    the one thing every notebook front end plays (VS Code and Cursor block embedded videos): at most
    `height` pixels high, one palette of `colors` shared by every frame and no dithering, and in a long
    battle every second frame, shown twice as long. A third of the size of a plain GIF."""
    from PIL import Image
    stride = 2 if len(frames) > 150 else 1
    last = frames[-1]
    frames = frames[::stride] + ([last] if (len(frames) - 1) % stride else [])
    w, h = frames[0].size
    if h > height:
        frames = [f.resize((round(w * height / h), height), Image.LANCZOS) for f in frames]
        w, h = frames[0].size
    # one palette for the whole battle, from its start, middle and end (trails and colours build up)
    sample = Image.new("RGB", (3 * w, h))
    for i, k in enumerate((0, len(frames) // 2, len(frames) - 1)):
        sample.paste(frames[k], (i * w, 0))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 * stride / fps), loop=0)
    return path


def save_video(frames: list, path: str, fps: int = 10, gif: bool = True) -> str:
    """Writes PIL frames to path: an .mp4 through ffmpeg (small, full quality - to keep or share), and
    the compact .gif next to it that the notebooks show (save_gif). Without ffmpeg an .mp4 request is
    written as the .gif only. gif=False: no .gif next to an .mp4, and a full-size one without ffmpeg
    (record_unity_video - the game's graphics make big GIFs). Returns the path actually written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    gif_path = os.path.splitext(path)[0] + ".gif"
    ffmpeg = ffmpeg_path()
    if path.endswith(".mp4") and ffmpeg:
        w, h = frames[0].size
        w, h = w - w % 2, h - h % 2                    # H.264 needs even sizes
        proc = None
        try:
            # +faststart: the index first, so a browser can start playing before the whole file is read
            proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
            for f in frames:
                proc.stdin.write(f.crop((0, 0, w, h)).tobytes())
            proc.stdin.close()
            if proc.wait() == 0:
                if gif:
                    save_gif(frames, gif_path, fps)
                return path
        except OSError:                                # ffmpeg cannot run here, or stopped early
            if proc is not None:
                proc.kill()
                proc.wait()
        if os.path.exists(path):
            os.remove(path)                            # a half-written file would show as a broken replay
    if gif:                                            # no ffmpeg, or this one has no H.264
        return save_gif(frames, gif_path, fps)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return gif_path


def _read_video(path: str, height: int = GIF_HEIGHT) -> tuple[list, float]:
    """The frames of a video (PIL images, at most `height` pixels high) and its frame rate, decoded with
    ffmpeg - ([], 0) where there is no ffmpeg or the file cannot be read."""
    from PIL import Image
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return [], 0.0
    info = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    size, rate = re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", info), re.search(r"([\d.]+) fps", info)
    if not size:
        return [], 0.0
    w, h = int(size.group(1)), int(size.group(2))
    if h > height:
        w, h = round(w * height / h / 2) * 2, height
    proc = subprocess.Popen([ffmpeg, "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    frames = []
    while len(chunk := proc.stdout.read(w * h * 3)) == w * h * 3:
        frames.append(Image.frombytes("RGB", (w, h), chunk))
    proc.stdout.close()
    proc.wait()
    return frames, float(rate.group(1)) if rate else 10.0


def record_battle(model, env, path: str, every: int = 2, deterministic: bool = False, dpi: int = 80,
                  stage: int | None = None, evaluation: bool = True) -> str:
    """Plays one battle against the rule AI (in the env's first battle slot) and writes a top-down
    replay to path: an .mp4 and the .gif the notebooks show (the .gif only where ffmpeg is missing).
    Returns the file written (the .mp4, or the .gif).

    model=None plays random legal actions. every: draw a frame every N decisions (seconds).
    stage: the curriculum stage to play (default: the current one).
    evaluation=False: against the stage's own opponent mix instead of the rule AI (e.g. a "rival" -
    the other method's fleet - in the versus notebook).
    The environment's battles are restarted for it; call model.reset_battles() before training on.
    """
    if stage is not None and stage != env.stage:
        previous = env.stage
        env.set_stage(stage)
        try:
            return record_battle(model, env, path, every, deterministic, dpi, evaluation=evaluation)
        finally:
            env.set_stage(previous)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    own_c, enemy_c = _cols(env, "critic_own"), _cols(env, "critic_enemy")
    # logged as "eval": replays never count for promotion
    with as_evaluation(env) if evaluation else contextlib.nullcontext():
        obs = env.reset()
        fig = Figure(figsize=(6, 6), dpi=dpi)          # off-screen, so the notebook's own plots still show
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        frames, t, total = [], 0, 0.0
        trails = {}
        result = None
        while True:
            arrays = env.raw_arrays(0) if result is None else env.final_arrays(0)
            for i in range(env.n_agents):
                if arrays["critic_own"][0, i, own_c["alive"]] > 0.5:
                    trails.setdefault(("o", i), _Trail("#35d0e0")).append((arrays["critic_own"][0, i, own_c["x"]], arrays["critic_own"][0, i, own_c["y"]]))
                if arrays["critic_enemy_mask"][0, i] > 0.5 and arrays["critic_enemy"][0, i, enemy_c["alive"]] > 0.5:
                    trails.setdefault(("e", i), _Trail("#e04848")).append((arrays["critic_enemy"][0, i, enemy_c["true_x"]], arrays["critic_enemy"][0, i, enemy_c["true_y"]]))
            if t % every == 0 or result is not None:
                m = {n: k for k, n in enumerate(env.spec.features["critic_match"])}
                score = arrays["critic_match"][0]
                title = (f"t = {t} s   points {score[m['my_score']] * 1000:.0f} vs {score[m['their_score']] * 1000:.0f}"
                         f"   reward/ship {total:+.2f}")
                if result is not None:
                    title = ("WON" if result["won"] else "LOST") + f" - {result['reason']}\n" + title
                draw_battle(env, arrays, ax, title, list(trails.values()))
                fig.canvas.draw()
                frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
            if result is not None:
                break
            actions = env.sample_random_actions(obs) if model is None else model.predict(obs, deterministic)
            obs, rewards, dones, infos = env.step(actions)
            n = max(1.0, float(obs["exists"][0].sum()))
            total += float(rewards[0].sum()) / n
            t += 1
            if dones[0]:
                result = infos[0]                          # draw the final picture next time round
    frames += [frames[-1]] * 15                        # hold the result for a moment
    path = save_video(frames, path)
    print(f"{'won' if result['won'] else 'lost'} ({result['reason']}) after {result['battle_time']:.0f} s - wrote {path}")
    return path


def record_unity_video(model, env, path: str, every: int = 2, width: int = 1280, height: int = 720,
                       fps: int = 10, deterministic: bool = False) -> str:
    """Plays one battle and records frames rendered by Unity itself - the real game's graphics,
    with the fog of war lifted so both fleets show. env must be a NavalEnv(..., graphics=True, n_envs=1),
    which needs a computer with a screen (not a SageMaker space). Returns the video written."""
    import tempfile
    from PIL import Image, ImageDraw
    worker = env.workers[0]
    with as_evaluation(env):
        obs = env.reset()
        frames, t, result = [], 0, None
        with tempfile.TemporaryDirectory() as tmp:
            while result is None:
                if t % every == 0:
                    png = os.path.join(tmp, f"{len(frames):05d}.png")
                    worker.render(png, width=width, height=height)
                    img = Image.open(png).convert("RGB")
                    ImageDraw.Draw(img).text((12, 10), f"t = {t * env.decision_period:.0f} s   {env.stages[env.stage]['name']}",
                                             fill=(255, 255, 255))
                    frames.append(img)
                actions = env.sample_random_actions(obs) if model is None else model.predict(obs, deterministic)
                obs, _, dones, infos = env.step(actions)
                t += 1
                if dones[0]:
                    result = infos[0]
    caption = ("WON" if result["won"] else "LOST") + f" - {result['reason']}"
    last = frames[-1].copy()
    ImageDraw.Draw(last).text((12, 30), caption, fill=(255, 230, 120))
    frames += [last] * (2 * fps)                       # hold the result for two seconds
    path = save_video(frames, path, fps=fps, gif=False)
    print(f"{caption.lower()} after {result['battle_time']:.0f} s - wrote {path}")
    return path


class VideoCallback(BaseCallback):
    """Records a replay of one battle every `every` timesteps while training, on the stage being
    trained: videos/battle_<timesteps>_stage<k>.mp4 and the .gif the notebooks show (the .gif only
    where ffmpeg is missing). With TeamSyncCallback they go to the team folder too.

    show=True also shows each replay in the notebook right away, under the training table - so you
    can watch the fleet while the training cell is still running."""

    def __init__(self, every: int, video_dir: str, show: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.every, self.video_dir, self.show = every, video_dir, show
        self._last = 0
        self.videos: list[str] = []

    def _on_training_start(self) -> None:
        self._last = self.num_timesteps
        self._stage = self.model.env.stage           # the stage this session trains
        self._session = 0

    def _on_rollout_end(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self._record(self.model.env.stage)
        return True

    def _on_training_end(self) -> None:
        # a session too short for `every` (a stage passed quickly) still leaves one replay of its stage
        if self._session == 0:
            self._record(self._stage)

    def _record(self, stage: int) -> None:
        env = self.model.env
        name = f"battle_{self.num_timesteps:09d}_stage{stage}{video_ext()}"
        path = record_battle(self.model, env, os.path.join(self.video_dir, name), stage=stage)
        self.videos.append(path)
        self._session += 1
        if self.show:
            _show_replay(Path(path))
        self.model.reset_battles()                     # the recording used the training battles


VIDEO_SUFFIXES = (".mp4", ".gif")


def _replay_gif(f: Path) -> Path | None:
    """The .gif to show for replay f: f itself, the .gif recorded next to an .mp4, or one made from the
    .mp4 now and kept next to it (an older or a teammate's replay) - None where there is no ffmpeg to
    make one."""
    gif = f.with_suffix(".gif")
    if gif.exists():
        return gif
    frames, fps = _read_video(str(f))
    return Path(save_gif(frames, str(gif), fps)) if frames else None


def _video_html(f: Path, height: int = GIF_HEIGHT, caption: str | None = None) -> str:
    """One replay as HTML, embedded in the notebook so it is still there when the notebook is reopened
    or downloaded. It is shown as an animated image (<img>, the .gif): the one kind of embedded media
    every front end shows - VS Code and Cursor block embedded <video> outright, and JupyterLab (also
    SageMaker Studio) strips the data off a <source> tag once a notebook is reopened. Only an .mp4
    without a .gif and without ffmpeg to make one is embedded as a video, with its data on the <video>
    tag itself, which JupyterLab keeps."""
    gif = _replay_gif(f)
    if gif is not None:
        b64 = base64.b64encode(gif.read_bytes()).decode("ascii")
        media = f'<img src="data:image/gif;base64,{b64}" style="height: {height}px;"/>'
    else:
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        media = (f'<video src="data:video/mp4;base64,{b64}" autoplay loop muted controls playsinline '
                 f'style="height: {height}px;"></video>')
        caption = (caption or f.name) + " (an MP4: VS Code cannot play it here - install ffmpeg to see it as a GIF)"
    return (f'<figure style="display: inline-block; margin: 0 12px 12px 0;">{media}'
            f'<figcaption>{caption or f.name}</figcaption></figure>')


def _display_html(html: str, text: str = "") -> None:
    """Shows HTML in a notebook (text where HTML cannot be shown); in a plain script it does nothing
    (the files are written anyway)."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is not None:
        display({"text/html": html, "text/plain": text or "(a replay - open this notebook in Jupyter to watch it)"},
                raw=True)


def _replay_stage(f: Path) -> int | None:
    """The curriculum stage in a replay's name (battle_<timesteps>_stage<k>.mp4), or None."""
    m = re.search(r"_stage(\d+)(?:_|\.|$)", f.name)
    return int(m.group(1)) if m else None


def _replay_caption(f: Path, label: str = "") -> str:
    parts = [label] if label else []
    m = re.match(r"battle_(\d+)_stage(\d+)", f.name)
    if m:
        parts.append(f"stage {int(m.group(2))}, after {int(m.group(1)):,} timesteps"
                     + (" (evaluated on it, trained on another stage)" if "_evaluated" in f.name else ""))
    parts.append(f.name)
    return " · ".join(parts)


def _show_replay(f: Path, label: str = "", height: int = GIF_HEIGHT) -> None:
    _display_html(_video_html(f, height, _replay_caption(f, label)), text=f"replay {f}")


def find_videos(video_path: str = "", prefix: str = "", contains: str | None = None, stage: int | None = None,
                newest: int | None = None) -> list[Path]:
    """The replays in a folder, oldest first - one file per replay: its .mp4 where there is one (its
    .gif is what the notebooks show), otherwise its .gif. prefix / contains: only files whose name
    starts with / includes it; stage: only those recorded on that curriculum stage; newest: the last n."""
    replays: dict[Path, Path] = {}
    for f in sorted(Path(video_path).glob(f"{prefix}*")):
        if f.suffix in VIDEO_SUFFIXES and f.is_file() and (contains is None or contains in f.name) \
                and (stage is None or _replay_stage(f) == stage):
            if f.suffix == ".mp4" or f.with_suffix("") not in replays:
                replays[f.with_suffix("")] = f
    files = [replays[k] for k in sorted(replays)]
    return files[-newest:] if newest else files


def show_videos(video_path: str | list | dict = "", prefix: str = "", contains: str | None = None,
                newest: int | None = None, stage: int | None = None, height: int = GIF_HEIGHT) -> list[Path]:
    """Shows the replays in the notebook - every one there is, oldest first (the notebooks' show_videos,
    for .mp4 and .gif). Each replay is its own output, embedded in the notebook as an animated image
    (see _video_html), so it plays in JupyterLab, SageMaker Studio, VS Code and Cursor alike, and is
    still there when the notebook is reopened or downloaded.

    video_path  a folder, a list of folders, or {"MAPPO": folder, "Double DQN": folder} to show each
                run's replays under its name
    prefix / contains   only files whose name starts with / includes it
    stage       only the replays recorded on that curriculum stage
    newest      only the last n (of each folder)
    Says so when there is nothing to show, instead of showing nothing. Returns the files shown."""
    if isinstance(video_path, dict):
        groups = list(video_path.items())
    elif isinstance(video_path, (list, tuple)):
        groups = [("", p) for p in video_path]
    else:
        groups = [("", video_path)]
    shown = []
    for label, path in groups:
        folder = Path(path)
        files = find_videos(path, prefix, contains, stage, newest)
        what = (f" of stage {stage}" if stage is not None else "") + \
               (f" matching {contains or prefix!r}" if (contains or prefix) else "")
        if not files:
            others = find_videos(path) if folder.is_dir() else []
            seen = sorted({_replay_stage(f) for f in others if _replay_stage(f) is not None})
            print(f"{label + ': ' if label else ''}no replays{what} in {folder.resolve()} yet"
                  + (f" - it has {len(others)} from other stages ({', '.join(map(str, seen)) or '?'}; newest "
                     f"{others[-1].name}): stage=None shows them" if others else "")
                  + ". A run records replays on the stage it trains on (VideoCallback, every `every` timesteps); "
                    "record_battle(model, env, path, stage=k) records one on any stage.")
            continue
        rest = len(find_videos(path, prefix, contains)) - len(files) if stage is not None else 0
        print(f"{label + ': ' if label else ''}{len(files)} replay{'s' if len(files) > 1 else ''}{what} in {folder}"
              + (f" - and {rest} from other stages" if rest else ""))
        for f in files:
            _show_replay(f, label, height)
        shown += files
    return shown


# ---------------------------------------------------------------------------------------------- beginning / middle / end

def get_model_identifiers(models_dir: str) -> list[str]:
    return [f[len("model_"):-len(".pt")] for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".pt")]


def find_key_identifiers(identifiers: list[str]):
    numeric = sorted(int(i) for i in identifiers if i.isdigit())
    final = "final" if "final" in identifiers else str(numeric[-1])
    return str(numeric[0]), str(numeric[len(numeric) // 2]), final


def view(models_dir: str, env, video_folder: str = "videos", every: int = 2):
    """Replays of the policy at the beginning, middle and end of training (the notebooks' view())."""
    from .mappo import MAPPO
    earliest, middle, final = find_key_identifiers(get_model_identifiers(models_dir))
    for stage, ident in zip(["beginning", "middle", "end"], [earliest, middle, final]):
        print(f"loading model_{ident}")
        model = MAPPO.load(os.path.join(models_dir, f"model_{ident}"), env=env)
        record_battle(model, env, os.path.join(video_folder, f"mappo-naval-{stage}{video_ext()}"), every=every)
    for stage in ["beginning", "middle", "end"]:
        show_videos(video_folder, prefix=f"mappo-naval-{stage}")
