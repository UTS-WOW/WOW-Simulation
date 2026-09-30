#!/usr/bin/env bash
# Behaviour cloning -> evaluation -> install in the game -> MAPPO fine-tuning (+ automatic evaluation).
# Waits for a running `bc.py collect` to finish (runs/bc_collect2.log) if its demos are not there yet.
# Usage: Training/pipeline_bc_mappo.sh [run_name]
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$HOME/anaconda3/bin/python}"
RUN="${1:-full_mappo}"
BIN=../Builds/NavalTrainer/NavalTrainer.x86_64

until grep -q "^saved" runs/bc_collect2.log 2>/dev/null && [ -f runs/bc/demos.npz ]; do sleep 15; done
sleep 5
echo "== behaviour cloning"; "$PY" -u bc.py train --epochs 8 || exit 1
echo "== evaluating the cloned policy (6v6 domination vs Elite)"
"$PY" -u evaluate.py --checkpoint runs/bc/bc.pt --stage 3 --difficulty 2 --episodes 48 --workers 8 \
      --out runs/eval_bc2_stage3_elite.json 2>/dev/null | tail -1
echo "== installing it in the game"
"$PY" - <<'PYEOF'
import sys, torch, shutil
sys.path.insert(0, ".")
from naval_rl.config import Config
from naval_rl.spec import Spec
from naval_rl.model import Actor
from naval_rl.export import export_policy
ck = torch.load("runs/bc/bc.pt", map_location="cpu", weights_only=False)
cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
spec = Spec.from_json(ck["spec"])
a = Actor(spec, cfg.d_model, cfg.heads, cfg.layers, cfg.hidden); a.load_state_dict(ck["actor"])
export_policy(a, None, spec, cfg, "runs/bc/bc_policy.bin")
shutil.copyfile("runs/bc/bc_policy.bin", "../Assets/StreamingAssets/RL/naval_policy.bin")
print("installed runs/bc/bc_policy.bin as the game's policy")
PYEOF
echo "== MAPPO fine-tuning ($RUN)"
WIN=$("$PY" -c "import json; s=json.load(open('runs/eval_bc2_stage3_elite.json'))['summary']; print(json.dumps({'update':0,'win_rate':s['win_rate'],'stderr':s['stderr'],'damage_dealt':s['damage_dealt'],'damage_taken':s['damage_taken'],'episodes':s['episodes'],'minutes':0,'note':'behaviour cloning only'}))")
mkdir -p "runs/$RUN/evals"; echo "$WIN" > "runs/$RUN/evals/summary.jsonl"
setsid nohup "$PY" -u train.py --unity-binary "$BIN" --workers 8 --run-name "$RUN" --init-from runs/bc/bc.pt \
    --set start_stage=3 --set actor_freeze_updates=10 --set lr_actor=5e-5 --set ent_coef=0.002 \
    --set ent_coef_final=0.0005 --set gamma=0.995 --set minibatches=16 --set total_updates=300 \
    > "runs/$RUN.log" 2>&1 < /dev/null &
sleep 60
setsid nohup "$PY" -u watch_checkpoints.py --run "$RUN" --stage 3 --episodes 48 > "runs/${RUN}_evals.log" 2>&1 < /dev/null &
echo "== started: training log runs/$RUN.log, evaluations runs/${RUN}_evals.log"
