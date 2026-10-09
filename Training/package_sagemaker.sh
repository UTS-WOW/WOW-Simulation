#!/usr/bin/env bash
# Packs everything SageMaker (or another computer) needs into one zip: the commander MAPPO code, the
# scenarios, the notebooks, the headless Linux player (no Unity install is needed) and - when there is
# one - the run "fleet"'s imitation checkpoint, so Stage 0 can start without running StageP first.
#
#   Training/build_player.sh            # once, after any change to the game's C# code
#   Training/package_sagemaker.sh       # writes wow_sagemaker.zip in the repository root
#
# Then upload wow_sagemaker.zip to SageMaker Studio and open Training/StageP-Imitation.ipynb or
# Training/Stage0-Gunnery.ipynb (the first cell unzips it to ~/wow). On a computer: unzip it anywhere,
# cd <folder>/Training, pip install -r requirements.txt jupyter, and open the same notebooks.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-wow_sagemaker.zip}"

if [ ! -x Builds/NavalTrainer/NavalTrainer.x86_64 ]; then
  echo "no headless player in Builds/NavalTrainer - run Training/build_player.sh first" >&2
  exit 1
fi

# a version stamp, so the notebook can tell when a newer bundle has been uploaded than the one it runs
VERSION="$(date +%Y-%m-%d_%H%M)-$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "$VERSION" > Training/BUNDLE_VERSION
trap 'rm -f Training/BUNDLE_VERSION' EXIT

EXTRA=()
if [ -f Training/runs_simple/fleet/models/imitation.pt ]; then
  EXTRA+=(Training/runs_simple/fleet/models/imitation.pt)      # head start: the captain cloned from the Elite AI
fi

rm -f "$OUT"
zip -qr "$OUT" \
  Training/simple_mappo Training/naval_rl Training/scenarios \
  Training/train_simple.py Training/watch_simple.py Training/WOW-MAPPO-Simple.ipynb Training/Stage*.ipynb \
  Training/make_stage_notebooks.py Training/requirements.txt Training/BUNDLE_VERSION Training/tests \
  "${EXTRA[@]}" \
  Builds/NavalTrainer \
  -x '*/__pycache__/*' 'Builds/NavalTrainer/*_BurstDebugInformation_DoNotShip/*' 'Builds/NavalTrainer/runs/*'
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), version $VERSION"
