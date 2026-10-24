#!/usr/bin/env bash
# Packs everything into one zip: the Unity project (Assets, Packages, ProjectSettings - open the
# unzipped folder in Unity Hub; Unity rebuilds its Library the first time), the commander MAPPO code,
# scenarios and notebooks, the Linux player (headless training and the playable game, no Unity
# install needed) and - when there are - the run "fleet"'s imitation checkpoint, so Stage 0 can start
# without running StageP first, and a trained MAPPO checkpoint for the notebooks to continue from.
#
#   Training/build_player.sh            # once, after any change to the game's C# code
#   Training/package_sagemaker.sh       # writes wow_sagemaker.zip in the repository root
#   YOUR_NAME=luki Training/package_sagemaker.sh   # the notebooks come with YOUR_NAME = "luki" filled in
#   FLEET_MODEL=Training/runs_simple/fleet_sagemaker/models/model_latest.pt Training/package_sagemaker.sh
#                                       # ... with that checkpoint as runs_simple/fleet/models/model_latest.pt
#                                       # (default: runs_simple/fleet/models/model_latest.pt, if there is one)
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
EXTRA_DIR="$(mktemp -d)"
trap 'rm -f Training/BUNDLE_VERSION; rm -rf "$EXTRA_DIR"' EXIT

# the MAPPO run's checkpoint to continue from, packed as runs_simple/fleet/models/model_latest.pt
MODEL="${FLEET_MODEL:-Training/runs_simple/fleet/models/model_latest.pt}"
if [ -f "$MODEL" ]; then
  mkdir -p "$EXTRA_DIR/Training/runs_simple/fleet/models"
  cp "$MODEL" "$EXTRA_DIR/Training/runs_simple/fleet/models/model_latest.pt"
  echo "including $MODEL as Training/runs_simple/fleet/models/model_latest.pt"
fi

# your name in every notebook (YOUR_NAME = "your-name" otherwise): the packed copies replace the originals
if [ -n "${YOUR_NAME:-}" ]; then
  mkdir -p "$EXTRA_DIR/Training"
  for nb in Training/WOW-MAPPO-Simple.ipynb Training/Stage*.ipynb; do
    sed 's/YOUR_NAME = \\"your-name\\"/YOUR_NAME = \\"'"$YOUR_NAME"'\\"/' "$nb" > "$EXTRA_DIR/$nb"
  done
  echo "notebooks with YOUR_NAME = \"$YOUR_NAME\""
fi

EXTRA=()
if [ -f Training/runs_simple/fleet/models/imitation.pt ]; then
  EXTRA+=(Training/runs_simple/fleet/models/imitation.pt)      # head start: the captain cloned from the Elite AI
fi

rm -f "$OUT"
zip -qr "$OUT" \
  Training/simple_mappo Training/naval_rl Training/scenarios \
  Training/train_simple.py Training/watch_simple.py Training/export_simple.py Training/WOW-MAPPO-Simple.ipynb Training/Stage*.ipynb Training/Versus-MAPPO-vs-PQN.ipynb \
  Training/make_stage_notebooks.py Training/requirements.txt Training/BUNDLE_VERSION Training/tests \
  "${EXTRA[@]}" \
  Builds/NavalTrainer \
  Assets Packages ProjectSettings \
  -x '*/__pycache__/*' 'Builds/NavalTrainer/*_BurstDebugInformation_DoNotShip/*' 'Builds/NavalTrainer/runs/*' \
     'Assets/Screenshots/*' 'Assets/Screenshots.meta' 'Assets/StreamingAssets/RL/*.bak' 'Assets/StreamingAssets/RL/*.bak.meta' '*.tmp'
if [ -d "$EXTRA_DIR/Training" ]; then
  case "$OUT" in /*) ZIP="$OUT" ;; *) ZIP="$(pwd)/$OUT" ;; esac
  (cd "$EXTRA_DIR" && zip -qr "$ZIP" Training)
fi
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), version $VERSION"
