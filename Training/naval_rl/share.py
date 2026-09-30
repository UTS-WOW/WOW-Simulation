"""Checkpoints the whole team can train from.

runs/ never leaves one machine (it is gitignored, and a checkpoint's self-play pool points at
absolute paths on that machine). publish() packs a checkpoint into checkpoints/<name>/, which is
committed to git: latest.pt with its snapshot paths made relative, the snapshot actors it needs, an
info.json describing it and a history.jsonl of every publish. localize_snapshots() is the other
half: on --resume it finds those snapshots again, wherever the checkpoint now lives, and copies them
into the new run.

A shared checkpoint is one line of training that team members take turns extending. publish()
refuses to replace it with a run that is behind it, or that started before someone else's publish,
so nobody overwrites a teammate's progress by accident.
"""

from __future__ import annotations

import datetime
import getpass
import json
import os
import shutil
import subprocess
import uuid

import torch

SHARED_DIR = "checkpoints"
INFO = "info.json"
HISTORY = "history.jsonl"


# ------------------------------------------------------------------ compatibility

def _layout(spec_json: dict) -> dict:
    """The parts of the spec that fix the network's weight shapes (padding caps do not)."""
    return {"dims": spec_json["dims"], "action_mode": spec_json["action_mode"],
            "heads": [(h["name"], h["fixed"], h["pointer"]) for h in spec_json["heads"]]}


def check_compatible(saved_spec: dict | None, spec, what: str) -> None:
    """Stop with a readable message if a checkpoint was trained on a different observation or
    action layout than the environment in front of us (a PyTorch size-mismatch trace otherwise)."""
    if saved_spec is None:
        return
    old, new = _layout(saved_spec), _layout(spec.to_json())
    if old == new:
        return
    diffs = []
    for k in sorted(set(old["dims"]) | set(new["dims"])):
        if old["dims"].get(k) != new["dims"].get(k):
            diffs.append(f"{k}: checkpoint {old['dims'].get(k)}, game {new['dims'].get(k)}")
    if old["action_mode"] != new["action_mode"]:
        diffs.append(f"action_mode: checkpoint {old['action_mode']}, game {new['action_mode']}")
    if old["heads"] != new["heads"]:
        diffs.append(f"action heads: checkpoint {old['heads']}, game {new['heads']}")
    raise SystemExit(
        f"[train] {what} was trained on a different observation/action layout than this build of "
        f"the game:\n  " + "\n  ".join(diffs) + "\n"
        "Rebuild the player from the commit the checkpoint came from, or start a new run.")


# ------------------------------------------------------------------ snapshots

def _find_snapshot(path: str, ckpt_dir: str) -> str | None:
    name = os.path.basename(path)
    candidates = [path] if os.path.isabs(path) else []
    candidates += [os.path.join(ckpt_dir, path),                        # shared: snapshots/actor_X.pt
                   os.path.join(ckpt_dir, "snapshots", name),
                   os.path.join(ckpt_dir, os.pardir, "snapshots", name)]  # runs/<run>/checkpoints/..
    return next((os.path.normpath(c) for c in candidates if os.path.isfile(c)), None)


def localize_snapshots(league, ckpt_path: str, run_dir: str) -> None:
    """Point every snapshot in a resumed league at a file inside this run, copying it there if it
    came from somewhere else. Snapshots that cannot be found are dropped with a warning."""
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    dest_dir = os.path.join(run_dir, "snapshots")
    os.makedirs(dest_dir, exist_ok=True)
    kept = []
    for s in league.snapshots:
        src = _find_snapshot(s["path"], ckpt_dir)
        if src is None:
            print(f"[train] snapshot {os.path.basename(s['path'])} not found - dropped from the league")
            continue
        dst = os.path.join(dest_dir, os.path.basename(src))
        if not (os.path.exists(dst) and os.path.samefile(src, dst)):
            shutil.copy2(src, dst)
        kept.append({**s, "path": os.path.abspath(dst)})
    league.snapshots = kept


# ------------------------------------------------------------------ lineage
#
# Every checkpoint carries a lineage: {"line": id, "shared": name, "from_update": n}. A line is one
# person's stretch of training; "shared"/"from_update" say which shared checkpoint it grew from.

def new_line(shared: str | None = None, from_update: int | None = None) -> dict:
    return {"line": uuid.uuid4().hex[:8], "shared": shared, "from_update": from_update}


def lineage_for_run(resume: dict | None, resume_path: str | None) -> dict:
    """Resuming a shared checkpoint starts a new line from it; resuming a local checkpoint stays on
    the line it was already on; a fresh run is a new line of its own."""
    if resume is None:
        return new_line()
    info_path = os.path.join(os.path.dirname(os.path.abspath(resume_path)), INFO)
    if os.path.isfile(info_path):
        with open(info_path) as f:
            return new_line(json.load(f)["name"], int(resume["update"]))
    return resume.get("lineage") or new_line()


def _history(dest_dir: str) -> list[dict]:
    p = os.path.join(dest_dir, HISTORY)
    if not os.path.isfile(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _check_can_replace(dest_dir: str, name: str, update: int, mine: dict) -> None:
    """Refuse to publish over a shared checkpoint unless this run continues it and nobody else has
    published to it since this run started."""
    history = _history(dest_dir)
    with open(os.path.join(dest_dir, INFO)) as f:
        current = json.load(f)
    own = [h for h in history if h.get("line") == mine.get("line")]
    if mine.get("shared") != name and not own:
        raise SystemExit(
            f"This run did not start from the shared checkpoint '{name}', so publishing it there would "
            f"replace {current['author']}'s training rather than continue it. Resume from "
            f"checkpoints/{name}/latest.pt first, publish under another --name, or pass --force.")
    if update <= int(current["update"]):
        raise SystemExit(
            f"'{name}' is already at update {current['update']} ({current['author']}, "
            f"{current['published']}); this checkpoint is at update {update}, so there is nothing "
            f"new to publish. Pass --force to replace it anyway.")
    base = mine["from_update"] if mine.get("shared") == name else min(h["update"] for h in own)
    others = [h for h in history if h.get("line") != mine.get("line") and int(h["update"]) > base]
    if others:
        h = max(others, key=lambda h: h["update"])
        raise SystemExit(
            f"This run continues '{name}' from update {base}, but {h['author']} published update "
            f"{h['update']} on {h['published']} since then. Publishing would throw their training "
            f"away. Publish under another --name (and compare the two with evaluate.py), or pass --force.")


# ------------------------------------------------------------------ publishing

def _author() -> str:
    try:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
        if name:
            return name
    except Exception:
        pass
    return getpass.getuser()


def _stage_info(ckpt: dict) -> dict:
    league = ckpt.get("league")
    if not league:
        return {}
    stage = int(league["stage"])
    stages = ckpt.get("config", {}).get("stages") or []
    window = league.get("window", {}).get(str(stage)) or league.get("window", {}).get(stage) or []
    return {"stage": stage,
            "stage_name": stages[stage]["name"] if stage < len(stages) else None,
            "winrate_vs_rule": round(sum(window) / len(window), 3) if window else None,
            "snapshots": len(league.get("snapshots", []))}


def default_name(source: str, runs_dir: str) -> str | None:
    """runs/<run>/... publishes as <run>."""
    rel = os.path.relpath(os.path.abspath(source), os.path.abspath(runs_dir))
    return None if rel.startswith(os.pardir) else rel.split(os.sep)[0]


def publish(source: str, dest_dir: str, note: str = "", force: bool = False) -> dict:
    """Copy a checkpoint into a shared folder. Returns the info written to info.json."""
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    name = os.path.basename(os.path.normpath(dest_dir))
    update = int(ckpt.get("update", 0))
    resumable = "opt_actor" in ckpt and "league" in ckpt

    info_path = os.path.join(dest_dir, INFO)
    exists = os.path.isfile(info_path)
    mine = ckpt.get("lineage")
    if mine is None and exists:
        mine = new_line()              # written before checkpoints carried a lineage: origin unknown
    elif mine is None:
        # first publish of a checkpoint older than lineages: stamp it, so training resumed from that
        # file later is recognised as continuing this shared checkpoint
        mine = new_line(name, update)
        torch.save({**ckpt, "lineage": mine}, source + ".tmp")
        os.replace(source + ".tmp", source)
    if exists and not force:
        _check_can_replace(dest_dir, name, update, mine)

    os.makedirs(os.path.join(dest_dir, "snapshots"), exist_ok=True)
    wanted = set()
    if ckpt.get("league"):
        ckpt_dir = os.path.dirname(os.path.abspath(source))
        kept = []
        for s in ckpt["league"]["snapshots"]:
            src = _find_snapshot(s["path"], ckpt_dir)
            if src is None:
                print(f"[share] snapshot {os.path.basename(s['path'])} not found - left out")
                continue
            fname = os.path.basename(src)
            dst = os.path.join(dest_dir, "snapshots", fname)
            if not (os.path.exists(dst) and os.path.samefile(src, dst)):
                shutil.copy2(src, dst)
            wanted.add(fname)
            kept.append({**s, "path": f"snapshots/{fname}"})
        ckpt["league"]["snapshots"] = kept
    for fname in os.listdir(os.path.join(dest_dir, "snapshots")):
        if fname not in wanted:
            os.remove(os.path.join(dest_dir, "snapshots", fname))
    ckpt.pop("lineage", None)      # the published checkpoint is the start of everyone's next line

    out = os.path.join(dest_dir, "latest.pt")
    torch.save(ckpt, out + ".tmp")
    os.replace(out + ".tmp", out)

    info = {"name": name, "update": update, "env_steps": ckpt.get("env_steps"),
            **_stage_info(ckpt),
            "use_with": "--resume" if resumable else "--init-from",
            "line": mine["line"],
            "author": _author(),
            "published": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source": os.path.relpath(os.path.abspath(source),
                                      os.path.dirname(os.path.dirname(os.path.abspath(dest_dir)))),
            "note": note}
    if "bc_heldout_accuracy" in ckpt:
        info["bc_heldout_accuracy"] = {k: round(v, 3) for k, v in ckpt["bc_heldout_accuracy"].items()}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    with open(os.path.join(dest_dir, HISTORY), "a") as f:
        f.write(json.dumps(info) + "\n")
    return info


def list_shared(root: str) -> list[dict]:
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, INFO)
        if os.path.isfile(p):
            with open(p) as f:
                out.append(json.load(f))
    return out
