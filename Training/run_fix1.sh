#!/usr/bin/env bash
# Resume the original run's update 2750 (start of stage 2) into a new run folder,
# runs/mappo_from2750_fix1, with the settings meant to fix the stage 2 plateau.
cd "$(dirname "$0")"
python3 train.py \
  --unity-binary ../NavalTrainerServer/NavalTrainerServer.x86_64 \
  --resume runs/mappo/checkpoints/update_002750.pt \
  --run-name mappo_from2750_fix1 \
  --set gamma=0.995 \
  --set ent_coef_final=0.003 \
  --set shaping_floor=0.3 \
  --set winrate_window=200 \
  --set total_updates=10000
