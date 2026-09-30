"""Share a training checkpoint with the team through git.

    # pack a run's latest checkpoint into checkpoints/<run>/ (then commit and push that folder)
    python share_checkpoint.py publish full_mappo --note "stage 3, 60% vs Elite"

    # any checkpoint file, under a name of your choice
    python share_checkpoint.py publish runs/bc/bc.pt --name bc

    # what the team has shared
    python share_checkpoint.py list

A teammate continues a shared checkpoint with
    python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --resume checkpoints/full_mappo/latest.pt
"""

from __future__ import annotations

import argparse
import os

from naval_rl import share

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SHARED = os.path.join(HERE, share.SHARED_DIR)


def cmd_publish(args) -> None:
    source = args.source
    if not os.path.isfile(source):
        source = os.path.join(RUNS, args.source, "checkpoints", "latest.pt")
        if not os.path.isfile(source):
            raise SystemExit(f"no checkpoint file '{args.source}', and no run '{args.source}' in runs/")
    name = args.name or share.default_name(source, RUNS)
    if not name:
        raise SystemExit("pass --name: the checkpoint is not inside runs/, so there is no run name to use")

    dest = os.path.join(SHARED, name)
    info = share.publish(source, dest, note=args.note, force=args.force)
    rel = os.path.relpath(dest, os.getcwd())
    print(f"published '{name}': update {info['update']}"
          + (f", stage {info['stage']} {info['stage_name']}" if "stage" in info else "")
          + (f", win rate vs rule AI {info['winrate_vs_rule']}" if info.get("winrate_vs_rule") is not None else "")
          + f"  ->  {rel}/")
    print("\nShare it with the team:")
    print(f"  git add {rel}")
    print(f"  git commit -m \"Share checkpoint {name} at update {info['update']}\"")
    print("  git push")
    print(f"\nThey continue it with:  python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 "
          f"{info['use_with']} {share.SHARED_DIR}/{name}/latest.pt")


def cmd_list(_args) -> None:
    rows = share.list_shared(SHARED)
    if not rows:
        print(f"no shared checkpoints in {os.path.relpath(SHARED, os.getcwd())}/")
        return
    print(f"{'name':<16} {'update':>6}  {'stage':<18} {'win vs rule':>11}  {'use with':<11} {'by':<14} published")
    for r in rows:
        stage = f"{r['stage']} {r['stage_name']}" if r.get("stage") is not None else "-"
        win = f"{r['winrate_vs_rule']:.2f}" if r.get("winrate_vs_rule") is not None else "-"
        print(f"{r['name']:<16} {r['update']:>6}  {stage:<18} {win:>11}  {r['use_with']:<11} "
              f"{r['author']:<14} {r['published']}")
        if r.get("note"):
            print(f"{'':<16} {r['note']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pub = sub.add_parser("publish", help="copy a checkpoint into checkpoints/<name>/ for git")
    pub.add_argument("source", help="a run name in runs/ (uses its checkpoints/latest.pt) or a checkpoint file")
    pub.add_argument("--name", help="shared name (default: the run name)")
    pub.add_argument("--note", default="", help="a line for the team: what changed, how it plays")
    pub.add_argument("--force", action="store_true", help="replace the shared checkpoint even if that "
                     "would discard someone else's newer training")
    pub.set_defaults(func=cmd_publish)
    sub.add_parser("list", help="show the shared checkpoints").set_defaults(func=cmd_list)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
