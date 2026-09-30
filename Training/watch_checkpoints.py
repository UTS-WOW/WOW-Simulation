"""Evaluates every new checkpoint of a training run automatically.

    python watch_checkpoints.py --run stage0_v3 --stage 0

Whenever the run saves a checkpoint (every checkpoint_every updates), it is copied to
runs/<run>/evals/update_NNNNNN.pt and evaluated against the rule-based AI with evaluate.py (sampled
actions, fixed seed so every checkpoint faces the same battles). Results are appended to
runs/<run>/evals/summary.jsonl and printed as a table row. Stops once training has ended and the
last checkpoint has been evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def training_running(run: str) -> bool:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("train.py" in line and f"--run-name {run}" in line for line in out.splitlines())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--episodes", type=int, default=96)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--base-port", type=int, default=5300)
    p.add_argument("--poll", type=float, default=20.0)
    a = p.parse_args()

    run_dir = os.path.join(HERE, "runs", a.run)
    evals = os.path.join(run_dir, "evals")
    os.makedirs(evals, exist_ok=True)
    summary_path = os.path.join(evals, "summary.jsonl")
    done = set()
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            done = {json.loads(line)["update"] for line in f if line.strip()}
    latest = os.path.join(run_dir, "checkpoints", "latest.pt")
    print(f"watching {latest}; already evaluated: {sorted(done)}", flush=True)

    while True:
        update = None
        if os.path.exists(latest):
            try:
                update = int(torch.load(latest, map_location="cpu", weights_only=False)["update"])
            except Exception:
                update = None             # caught it mid-write; try again next poll
        if update is not None and update not in done:
            snap = os.path.join(evals, f"update_{update:06d}.pt")
            shutil.copyfile(latest, snap)
            out = os.path.join(evals, f"update_{update:06d}.json")
            cmd = [sys.executable, os.path.join(HERE, "evaluate.py"), "--checkpoint", snap, "--stage", str(a.stage),
                   "--episodes", str(a.episodes), "--workers", str(a.workers), "--base-port", str(a.base_port),
                   "--out", out]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
            if r.returncode == 0 and os.path.exists(out):
                with open(out) as f:
                    s = json.load(f)["summary"]
                row = {"update": update, "win_rate": s["win_rate"], "stderr": s["stderr"],
                       "damage_dealt": s["damage_dealt"], "damage_taken": s["damage_taken"],
                       "episodes": s["episodes"], "minutes": round((time.time() - t0) / 60, 1)}
                with open(summary_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"update {update:4d}: win {s['win_rate']:.2f} ± {s['stderr']:.2f}   "
                      f"damage dealt {s['damage_dealt']:.2f} vs taken {s['damage_taken']:.2f}", flush=True)
            else:
                print(f"update {update}: evaluation failed\n{r.stdout[-800:]}\n{r.stderr[-800:]}", flush=True)
            done.add(update)
            continue
        if not training_running(a.run):
            print("training has stopped and the last checkpoint is evaluated", flush=True)
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
