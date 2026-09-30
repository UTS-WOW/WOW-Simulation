#!/usr/bin/env bash
# Runs every training path briefly against the real game and fails loudly if any breaks:
# each curriculum stage, self-play against past snapshots, and each ablation mode.
# Usage: Training/check_paths.sh [path/to/NavalTrainer.x86_64]
set -uo pipefail
cd "$(dirname "$0")"
BIN="${1:-../Builds/NavalTrainer/NavalTrainer.x86_64}"
PY="${PYTHON:-python}"
PORT=5100
FAILED=0
QUICK_STAGE='[{"name":"quick3v3","procedural":{"mode":0,"preset":1,"density":0,"ships":[3,3],"time_limit":60,"world_reuse":100},"difficulty":1,"promote_winrate":2}]'

run() {
  local name="$1"; shift
  rm -rf "runs/check_$name"
  if timeout 600 "$PY" -u train.py --unity-binary "$BIN" --workers 2 --run-name "check_$name" \
       --set base_port=$PORT --set rollout=64 --set chunk=16 --set total_updates=3 \
       --set export_path="\"runs/check_$name/policy.bin\"" "$@" > "runs/check_$name.log" 2>&1; then
    local updates episodes exceptions
    updates=$(wc -l < "runs/check_$name/metrics.jsonl")
    episodes=$(wc -l < "runs/check_$name/episodes.jsonl" 2>/dev/null || echo 0)
    exceptions=$(cat runs/check_$name/unity_logs/*.log | grep -c "Exception" || true)
    if [ "$updates" -eq 3 ] && [ "$exceptions" -eq 0 ]; then
      echo "PASS  $name  ($updates updates, $episodes battles finished)"
    else
      echo "FAIL  $name  ($updates updates, $exceptions exceptions in the game logs)"; FAILED=1
    fi
  else
    echo "FAIL  $name  (trainer exited with an error, see runs/check_$name.log)"; FAILED=1
    tail -5 "runs/check_$name.log"
  fi
  PORT=$((PORT + 10))
}

run stage1_koth       --set start_stage=1
run stage2_archipelago --set start_stage=2
run stage3_domination --set start_stage=3
run stage4_open       --set start_stage=4
run selfplay_snapshots --set "stages=$QUICK_STAGE" --set selfplay_from_stage=0 --set snapshot_every=1 \
                       --set mix_latest=0.2 --set mix_snapshot=0.8 --set mix_rule=0.0
run lowlevel_actions  --set "stages=$QUICK_STAGE" --set action_mode='"lowlevel"'
run belief_critic     --set "stages=$QUICK_STAGE" --set critic_mode='"belief"'
run ippo              --set "stages=$QUICK_STAGE" --set algo='"ippo"'
run no_spotting_reward --set "stages=$QUICK_STAGE" --set spotting_reward=false
exit $FAILED
