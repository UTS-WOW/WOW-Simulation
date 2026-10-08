#!/usr/bin/env bash
# Packs everything SageMaker needs into one zip: the simple MAPPO code, the scenarios, the notebook
# and the headless Linux player (no Unity install is needed on SageMaker).
#
#   Training/build_player.sh            # once, after any change to the game's C# code
#   Training/package_sagemaker.sh       # writes wow_sagemaker.zip in the repository root
#
# Then upload wow_sagemaker.zip to SageMaker Studio and open Training/WOW-MAPPO-Simple.ipynb
# (its first cell unzips it to ~/wow).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-wow_sagemaker.zip}"

if [ ! -x Builds/NavalTrainer/NavalTrainer.x86_64 ]; then
  echo "no headless player in Builds/NavalTrainer - run Training/build_player.sh first" >&2
  exit 1
fi

rm -f "$OUT"
zip -qr "$OUT" \
  Training/simple_mappo Training/naval_rl Training/scenarios \
  Training/train_simple.py Training/WOW-MAPPO-Simple.ipynb Training/requirements.txt \
  Builds/NavalTrainer \
  -x '*/__pycache__/*' 'Builds/NavalTrainer/*_BurstDebugInformation_DoNotShip/*' 'Builds/NavalTrainer/runs/*'
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
