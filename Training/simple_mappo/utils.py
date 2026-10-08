"""Logging, plots, evaluation and battle replays - the helpers the Week 9/10 notebooks use, for the
naval environment.

    env = Monitor(NavalEnv(...), log_dir)      # one CSV line per finished battle
    plot_results(log_dir)                      # the learning curve
    plot_curriculum(log_dir)                   # win rate per curriculum stage
    evaluate(model, env, n_battles=40)         # win rate against the rule AI (model=None: random play)
    record_battle(model, env, "videos/x.gif")  # a top-down replay of one battle
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from pathlib import Path

import numpy as np

REASON_KEYS = ["player_score", "enemy_score", "player_kills", "enemy_kills", "damage_dealt_hulls",
               "enemy_damage_dealt_hulls", "first_capture_time"]


# ---------------------------------------------------------------------------------------------- Monitor

class Monitor:
    """Wraps a NavalEnv and writes <log_dir>/monitor.csv, one line per finished battle.

    Columns: r (average reward per ship), l (decisions), t (seconds since start), won, battle_time,
    the game's own scores and kills, and term_<name>: what each reward term paid per ship.
    """

    def __init__(self, env, log_dir: str):
        self.env = env
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "monitor.csv")
        self.terms = list(env.reward_weights)
        self.columns = (["r", "l", "t", "won", "stage", "opponent", "battle_time"] + REASON_KEYS
                        + ["term_" + k for k in self.terms] + ["reason"])
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write("#" + json.dumps({"t_start": time.time(), "env_id": "WOW naval MAPPO"}) + "\n")
                f.write(",".join(self.columns) + "\n")

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        rows = [self._row(i) for i in infos if i]
        if rows:
            with open(self.path, "a") as f:
                f.writelines(rows)
        return obs, rewards, dones, infos

    def _row(self, info: dict) -> str:
        ep, stats, terms = info["episode"], info.get("stats", {}), info.get("reward_terms", {})
        kind = "eval" if info.get("evaluation") else info.get("opponent", "rule")
        values = [ep["r"], ep["l"], ep["t"], int(info["won"]), info.get("stage", 0), kind, info["battle_time"]]
        values += [stats.get(k, "") for k in REASON_KEYS]
        values += [terms.get(k, 0.0) for k in self.terms]
        values += ['"' + str(info.get("reason", "")).replace('"', "'") + '"']
        return ",".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in values) + "\n"


def load_results(log_dir: str, training_only: bool = True):
    """monitor.csv as a DataFrame. opponent is "rule" (the rule AI), "self" (self-play) or "eval"
    (an evaluation battle); training_only drops the evaluation battles."""
    import pandas as pd
    df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
    if training_only and "opponent" in df:
        df = df[df.opponent != "eval"].reset_index(drop=True)
    return df


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

def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(y)))
    return np.convolve(y, np.ones(window) / window, mode="valid")


def plot_results(log_folder: str, title: str = "Learning Curve", window: int = 50):
    """The notebooks' learning curve: reward per battle against timesteps, smoothed."""
    import matplotlib.pyplot as plt
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = _smooth(y, window)
    x = x[len(x) - len(y):]
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


def plot_curriculum(log_folder: str, window: int = 50):
    """Win rate over training, one colour per curriculum stage (a new colour = a promotion)."""
    import matplotlib.pyplot as plt
    df = load_results(log_folder)
    x = np.cumsum(df.l.values)
    plt.figure(figsize=(10, 4))
    for st in sorted(df.stage.unique()):
        part = df.stage.values == st
        y = df.won.values[part].astype(float)
        w = max(1, min(window, len(y)))
        plt.plot(x[part][w - 1:], np.convolve(y, np.ones(w) / w, mode="valid"), label=f"stage {st}")
    plt.ylim(0, 1)
    plt.xlabel("Number of Timesteps")
    plt.ylabel(f"Win rate (last {window} battles)")
    plt.title("Curriculum: win rate per stage")
    plt.legend()
    plt.show()


def plot_reward_terms(log_folder: str, window: int = 50):
    """What the fleet is actually being paid for: each reward term per battle, smoothed."""
    import matplotlib.pyplot as plt
    df = load_results(log_folder)
    x = np.cumsum(df.l.values)
    plt.figure(figsize=(10, 5))
    for col in [c for c in df.columns if c.startswith("term_")]:
        y = _smooth(df[col].values.astype(float), window)
        plt.plot(x[len(x) - len(y):], y, label=col[5:])
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Reward per ship per battle")
    plt.title("Reward terms")
    plt.legend(ncol=3, fontsize=8)
    plt.show()


def plot_progress(log_folder: str):
    """The PPO diagnostics from progress.csv, one row per update."""
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(os.path.join(log_folder, "progress.csv"))
    cols = ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.plot(df.total_timesteps, df[c])
        ax.set_title(c)
        ax.set_xlabel("timesteps")
    plt.tight_layout()
    plt.show()


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
    env.set_evaluation(True, held_out)
    try:
        out = _evaluate(model, env, n_battles, deterministic)
    finally:
        env.set_stage(previous)
        env.set_evaluation(False)
    if not keep_results:
        out.pop("results")
    return out


def _evaluate(model, env, n_battles: int, deterministic: bool) -> dict:
    obs = env.reset()
    E = env.n_envs
    quota = math.ceil(n_battles / E)        # the same number per battle slot, so short battles are not over-counted
    done_per_env = np.zeros(E, dtype=int)
    results = []
    while done_per_env.min() < quota:
        actions = env.sample_random_actions(obs) if model is None else model.predict(obs, deterministic)
        obs, _, dones, infos = env.step(actions)
        for e in np.flatnonzero(dones):
            if done_per_env[e] < quota:
                results.append(infos[e])
            done_per_env[e] += 1
    won = np.array([r["won"] for r in results], dtype=float)
    out = {
        "stage": env.stages[env.stage]["name"],
        "battles": len(results),
        "win_rate": float(won.mean()),
        "win_rate_stderr": float(won.std(ddof=1) / np.sqrt(len(won))) if len(won) > 1 else 0.0,
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


def record_battle(model, env, path: str, every: int = 2, deterministic: bool = False, dpi: int = 80) -> str:
    """Plays one battle (in the env's first battle slot) and writes a top-down replay to path (.gif).

    model=None plays random legal actions. every: draw a frame every N decisions (seconds).
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image

    matplotlib.use("Agg", force=False)
    own_c, enemy_c = _cols(env, "critic_own"), _cols(env, "critic_enemy")
    obs = env.reset()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
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
    plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames += [frames[-1]] * 15                        # hold the result for a moment
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"{'won' if result['won'] else 'lost'} ({result['reason']}) after {result['battle_time']:.0f} s - wrote {path}")
    return path


def show_videos(video_path: str = "", prefix: str = ""):
    """Shows the replays in the notebook (the notebooks' show_videos, for .gif and .mp4)."""
    from IPython import display as ipythondisplay
    html = []
    for f in sorted(Path(video_path).glob(f"{prefix}*")):
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        if f.suffix == ".gif":
            html.append(f'<figure style="display:inline-block"><img src="data:image/gif;base64,{b64}" style="height: 400px;"/>'
                        f"<figcaption>{f.name}</figcaption></figure>")
        elif f.suffix == ".mp4":
            html.append(f'<video autoplay loop controls style="height: 400px;"><source src="data:video/mp4;base64,{b64}" type="video/mp4"/></video>')
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


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
        record_battle(model, env, os.path.join(video_folder, f"mappo-naval-{stage}.gif"), every=every)
    for stage in ["beginning", "middle", "end"]:
        show_videos(video_folder, prefix=f"mappo-naval-{stage}")
