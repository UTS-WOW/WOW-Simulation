"""Prints a run's metrics.jsonl in the same format as the training terminal.

Usage (from Training/):
    python3 show_log.py runs/mappo | less
    python3 show_log.py runs/mappo_from2750_fix1 | tail -n 50
"""

import json
import sys

run = sys.argv[1] if len(sys.argv) > 1 else "runs/mappo"
with open(f"{run.rstrip('/')}/metrics.jsonl") as f:
    for line in f:
        x = json.loads(line)
        print(f"[{x['update']:5d}] stage {x['stage']} {x['stage_name']:<16} "
              f"win {x['winrate_vs_rule']:.2f}  eps {x['episodes']:3d}  "
              f"pl {x.get('policy_loss', 0):+.3f} vl {x.get('value_loss', 0):.3f} "
              f"ent {x.get('entropy', 0):.2f} kl {x.get('approx_kl', 0):.4f}  "
              f"{x['decisions_per_s']:.0f} dec/s")
