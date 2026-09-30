"""Train a MAPPO fleet policy against the Unity naval simulation.

    # 8 headless players launched and managed by the trainer
    python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --workers 8

    # the Unity editor (GameBootstrap.rlTrainingServer ticked, press Play) - handy for watching
    python train.py --ports 5005 --workers 1

    # no Unity at all: the mock environment, to check the trainer itself
    python train.py --mock --workers 4 --set total_updates=50

Outputs go to runs/<run_name>/: config.json, metrics.jsonl (one line per update), episodes.jsonl
(one line per finished battle), checkpoints/, snapshots/, and every export_every updates the
policy is exported for the game to Assets/StreamingAssets/RL/naval_policy.bin.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from naval_rl.config import Config
from naval_rl.env import init_message, make_workers
from naval_rl.export import export_policy
from naval_rl.league import League
from naval_rl.mappo import MAPPO

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML or JSON file of Config overrides")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one config value")
    p.add_argument("--unity-binary", help="headless training player to launch")
    p.add_argument("--ports", type=int, nargs="*", help="connect to already-running environments")
    p.add_argument("--workers", type=int)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--run-name")
    p.add_argument("--resume", help="checkpoint to resume from (optimiser, league and counters included)")
    p.add_argument("--init-from", help="checkpoint to take weights from, starting a fresh run")
    return p.parse_args()


def main():
    args = parse_args()
    # resuming continues the run with its own settings (rollout, rewards, stage list...);
    # --config and --set still override them
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    cfg = Config.load(args.config, args.set, base=resume["config"] if resume else None)
    if args.unity_binary:
        cfg.unity_binary = args.unity_binary
    if args.ports:
        cfg.ports = args.ports
    if args.workers:
        cfg.workers = args.workers
    if args.mock:
        cfg.mock = True
    if args.run_name:
        cfg.run_name = args.run_name
    if cfg.ports:
        cfg.workers = len(cfg.ports)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if cfg.device == "auto" else cfg.device))

    run_dir = os.path.join(HERE, cfg.out_dir, cfg.run_name)
    for sub in ("checkpoints", "snapshots", "unity_logs"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    init = init_message(cfg.max_team, cfg.max_allies, cfg.max_contacts, cfg.max_zones, cfg.action_mode,
                        cfg.decision_period, cfg.sim_dt, cfg.reflexes)
    binary = cfg.unity_binary if cfg.unity_binary is None or os.path.isabs(cfg.unity_binary) \
        else os.path.abspath(cfg.unity_binary)
    print(f"[train] starting {cfg.workers} {'mock ' if cfg.mock else ''}environment(s) on {device}")
    workers = make_workers(cfg.workers, init, binary, cfg.base_port, cfg.ports,
                           log_dir=os.path.join(run_dir, "unity_logs"), mock=cfg.mock)
    spec = workers[0].spec
    with open(os.path.join(run_dir, "spec.json"), "w") as f:
        json.dump(spec.to_json(), f, indent=2)

    league = League(cfg, HERE, seed=cfg.seed)
    algo = MAPPO(cfg, spec, workers, league, device)
    if args.resume:
        algo.load_state_dict(resume)
        print(f"[train] resumed from {args.resume} at update {algo.update}, stage {league.stage}")
    elif args.init_from:
        algo.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=False), weights_only=True)
        print(f"[train] initialised weights from {args.init_from}")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    metrics_f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    episodes_f = open(os.path.join(run_dir, "episodes.jsonl"), "a")
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(run_dir, "tb"))
    except Exception:
        pass

    export_path = cfg.export_path if os.path.isabs(cfg.export_path) else os.path.join(HERE, cfg.export_path)
    algo.reset_all()
    t_start = time.time()
    try:
        while algo.update < cfg.total_updates and not stop["now"]:
            collect_s = algo.collect()
            t_learn = time.time()
            stats = algo.learn()
            learn_s = time.time() - t_learn

            eps = algo.completed
            algo.completed = []
            for e in eps:
                episodes_f.write(json.dumps({"update": algo.update, **e}) + "\n")
            episodes_f.flush()

            promoted = league.maybe_promote()
            stats.update({
                "update": algo.update, "env_steps": algo.env_steps, "stage": league.stage,
                "stage_name": cfg.stages[league.stage]["name"], "winrate_vs_rule": league.winrate(),
                "episodes": len(eps), "decisions_per_s": cfg.rollout * cfg.workers / max(1e-6, collect_s),
                "collect_s": collect_s, "learn_s": learn_s, "elapsed_min": (time.time() - t_start) / 60,
                "snapshots": len(league.snapshots),
                **algo.collect_stats,
            })
            if eps:
                stats["learner_score"] = float(np.mean([e["learner_score"] for e in eps]))
                for k in ("learner_focus_fire", "learner_spotting_share", "learner_dd_concealed_in_gun_range",
                          "opponent_focus_fire", "opponent_spotting_share"):
                    vals = [e[k] for e in eps if k in e and e[k] >= 0]
                    if vals:
                        stats[k] = float(np.mean(vals))
            metrics_f.write(json.dumps(stats) + "\n")
            metrics_f.flush()
            if tb:
                for k, v in stats.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        tb.add_scalar(k, v, algo.update)

            if algo.update % cfg.log_every == 0:
                print(f"[{algo.update:5d}] stage {league.stage} {cfg.stages[league.stage]['name']:<16} "
                      f"win {stats['winrate_vs_rule']:.2f}  eps {len(eps):3d}  "
                      f"pl {stats.get('policy_loss', 0):+.3f} vl {stats.get('value_loss', 0):.3f} "
                      f"ent {stats.get('entropy', 0):.2f} kl {stats.get('approx_kl', 0):.4f}  "
                      f"{stats['decisions_per_s']:.0f} dec/s", flush=True)
            if promoted:
                print(f"[train] promoted to stage {league.stage}: {cfg.stages[league.stage]['name']}", flush=True)

            if league.stage >= cfg.selfplay_from_stage and algo.update % cfg.snapshot_every == 0:
                path = os.path.join(run_dir, "snapshots", f"actor_{algo.update:06d}.pt")
                torch.save(algo.actor.state_dict(), path)
                league.add_snapshot(path, algo.update)
            if algo.update % cfg.checkpoint_every == 0:
                save(algo, run_dir)
            if algo.update % cfg.export_every == 0:
                export_policy(algo.actor, algo.critic if cfg.algo == "mappo" else None, spec, cfg, export_path, algo.value_norm)
    finally:
        save(algo, run_dir)
        try:
            export_policy(algo.actor, algo.critic if cfg.algo == "mappo" else None, spec, cfg, export_path, algo.value_norm)
            print(f"[train] exported policy to {export_path}")
        except Exception as e:  # never lose the checkpoint over an export problem
            print(f"[train] export failed: {e}")
        for w in workers:
            w.close()
        metrics_f.close()
        episodes_f.close()


def save(algo, run_dir):
    """latest.pt every time; a numbered copy every 10 checkpoints so disk use stays bounded."""
    state = algo.state_dict()
    torch.save(state, os.path.join(run_dir, "checkpoints", "latest.pt"))
    if algo.update % (algo.cfg.checkpoint_every * 10) == 0:
        torch.save(state, os.path.join(run_dir, "checkpoints", f"update_{algo.update:06d}.pt"))


if __name__ == "__main__":
    main()
