#!/usr/bin/env bash
# Builds the headless training player into Builds/NavalTrainer/.
#
# Unity refuses to open a project that is already open in the editor, so the build runs on a
# synced copy in ~/.cache/naval-rl-build (Library included, so there is no full reimport) and the
# player is written back into this project's Builds/ folder. Your open editor is not touched.
#
# Usage: Training/build_player.sh [path/to/Unity]
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT="$(pwd)"
COPY="${NAVAL_BUILD_COPY:-$HOME/.cache/naval-rl-build}"
UNITY="${1:-${UNITY:-$HOME/Unity/Hub/Editor/$(sed -n 's/m_EditorVersion: //p' ProjectSettings/ProjectVersion.txt)/Editor/Unity}}"

# The copy keeps the downloaded packages but not the import database: copying that while the editor
# is writing to it gives a build whose scene no longer finds its scripts. Unity reimports the copy
# once (a few minutes) and reuses its own database on later builds.
mkdir -p "$COPY/Library"
rsync -a --delete --exclude /Temp/ --exclude /Logs/ --exclude /Builds/ --exclude /Training/ \
      --exclude /obj/ --exclude /bin/ --exclude /.git/ --exclude /Library/ --exclude /UserSettings/ \
      "$PROJECT/" "$COPY/"
rsync -a "$PROJECT/Library/PackageCache/" "$COPY/Library/PackageCache/"
mkdir -p "$PROJECT/Logs"
"$UNITY" -batchmode -nographics -projectPath "$COPY" \
  -executeMethod Naval.RL.EditorTools.RLBuild.BuildLinuxTrainer \
  -rlBuildPath "${NAVAL_BUILD_OUT:-$PROJECT/Builds/NavalTrainer/NavalTrainer.x86_64}" -logFile "$PROJECT/Logs/rl_build.log"
echo "built: ${NAVAL_BUILD_OUT:-Builds/NavalTrainer/NavalTrainer.x86_64} (log: Logs/rl_build.log)"
