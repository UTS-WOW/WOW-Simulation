# Training Progress Report — 4th October, 7:20 pm

Times are Sydney time. Daylight saving started at 3 am on 4 October, so times up to then are **AEST (UTC+10)** and
times after are **AEDT (UTC+11)**.

Figures come from `metrics.jsonl` and `episodes.jsonl` in each run's folder under `Training/runs/`.

The previous report is `Progress_Report_3rd_October_11-50pm.md`. The full baseline settings are in
`Training_Settings_4th_October_5-44pm.md`.

## Summary

- **The original run `runs/mappo` was stopped at update 4759** (Sun 4 Oct, 6:43 pm AEDT), after **43.5 h** of
  training.
  - It plateaued on stage 2 (`archipelago_3v3`). Its win rate fell from about 0.62 at the start of the stage to
    about 0.22 at the end.
  - It kept drifting into a passive "hold fire" mode.
- **A new run `runs/mappo_from2750_fix1` started at 7:01 pm AEDT.**
  - It resumes from update 2750, the start of stage 2, when the policy was at its best.
  - It has four setting changes aimed at the plateau.
- **Early status of the new run** (updates 2751–2780): healthy.
  - Win rate about 0.57–0.60 against the Veteran AI.
  - Holds fire 5% of the time.
  - Entropy higher, as intended.
  - It's too early to judge the fix: the original run only started sliding around update 2860.

## Original run `mappo`: final state

| Stage | Updates | Update time | Battles | Win rate (whole stage) | Result |
|---|---|---|---|---|---|
| 0 `bb_duel` (Recruit) | 1–755 | 4.2 h | 35,606 | 0.338 | Promoted at 756 |
| 1 `koth_3v3` (Veteran) | 756–2730 | 19.6 h | 46,723 | 0.185 | Promoted at 2731 |
| 2 `archipelago_3v3` (Veteran) | 2731–4759 | 19.8 h | 36,298 | 0.331 | Plateau, stopped |
| **Total** | **4759** | **43.5 h** | **118,627** | | 39.0 M decisions |

**Stage 2 over time** (250-update blocks, against the Veteran AI):

| Updates | Win rate | Holds fire | Damage dealt vs taken (hulls) |
|---|---|---|---|
| 2731–2980 | 0.49 (0.62 in the first 50) | 26% | 3.08 vs 2.82 |
| 2981–3230 | 0.28 | 40% | 2.76 vs 2.90 |
| 3231–3480 | 0.23 | 48% | 2.67 vs 3.05 |
| 3481–3730 | 0.39 | 26% | 2.98 vs 2.77 |
| 3731–3980 | 0.37 | 30% | 2.97 vs 2.92 |
| 3981–4230 | 0.29 | 28% | 2.81 vs 3.00 |
| 4231–4480 | 0.38 | 51% | 2.99 vs 3.03 |
| 4481–4759 | 0.22 | 56% | 2.65 vs 3.14 |

**Session times:**

| Session | Updates | Time |
|---|---|---|
| 1 | 1–220 | Fri 2 Oct 22:01 – 23:15 AEST |
| 2 | 221–2907 | Fri 2 Oct 23:23 AEST – Sat 3 Oct 23:43 AEST |
| 3 | 2908–4759 | Sat 3 Oct 23:44 AEST – Sun 4 Oct 18:43 AEDT, about 18 h, with `total_updates` raised to 10,000 |

All checkpoints are kept in `runs/mappo/checkpoints/`: `update_000250.pt` to `update_004750.pt`, plus `latest.pt`
at update 4759.

## Diagnosis of the plateau

- **The win rate follows a passive mode.** The correlation between hold-fire share and win rate over
  25-update blocks was **−0.93** in early stage 2 and **−0.61** in stage 1.
  - When the policy holds fire about 10% of the time, it out-damages the AI and wins about 60%.
  - When it holds fire 45–73% of the time, it wins 7–30%.
- **The schedules had already bottomed out.**
  - The exploration (entropy) bonus reached its floor of 0.001 at update 300.
  - The shaping rewards (damage, kills, spotting) reached their 10% floor at update 1500.
  - So stage 2 trained with almost no exploration and mostly sparse rewards.
- **The planning horizon was short.** `gamma` 0.99 looks about 100 s ahead, while stage 2 battles last
  8 minutes.
- **Promotion was noisy.** A single 100-battle window decided promotion, and both promotions happened at a true
  win rate of about 0.45.
- **The code itself is fine.** The MAPPO implementation was reviewed with no bugs found, and the 10 related tests
  pass.

## New run `mappo_from2750_fix1`: what changed

**Starting point:** `runs/mappo/checkpoints/update_002750.pt`.

- That's stage 2, update 2750, with the policy in aggressive mode: win rate about 0.62, holding fire 4%.
- `update_003000.pt` was not used because the slide had already started there (win rate 0.32, holding fire 49%).

**Settings changed** (everything else is identical to the original run):

| Setting | Original | New | Why |
|---|---|---|---|
| `ent_coef_final` | 0.001 | **0.003** | 3× higher exploration floor, so the policy doesn't lock into one mode |
| `shaping_floor` | 0.1 | **0.3** | Damage, kill and spotting rewards stay at 30%, a steady incentive to engage |
| `gamma` | 0.99 | **0.995** | About a 200 s horizon instead of 100 s, so points and winning outweigh short-term safety. Matches the README recipe. |
| `winrate_window` | 100 | **200** | Promotion needs a strong 200-battle stretch, so lucky promotions are less likely |
| `total_updates` | 10,000 | 10,000 | Unchanged |

**Command used to start it** (from `Training/`). It's saved as `Training/run_fix1.sh` and was started with:

```
bash run_fix1.sh
```

which runs:

```
python3 train.py \
  --unity-binary ../NavalTrainerServer/NavalTrainerServer.x86_64 \
  --resume runs/mappo/checkpoints/update_002750.pt \
  --run-name mappo_from2750_fix1 \
  --set gamma=0.995 \
  --set ent_coef_final=0.003 \
  --set shaping_floor=0.3 \
  --set winrate_window=200 \
  --set total_updates=10000
```

**To resume this run later,** use its own checkpoint. Don't re-run the script, which always starts from 2750:

```
python3 train.py --unity-binary ../NavalTrainerServer/NavalTrainerServer.x86_64 --resume runs/mappo_from2750_fix1/checkpoints/latest.pt
```

**Output folder:** `Training/runs/mappo_from2750_fix1/`, containing `metrics.jsonl`, `episodes.jsonl`,
`config.json` (with the settings above) and `checkpoints/`. The original run's folder is untouched.

## New run: early results

Updates 2751–2780, 30 updates, 0.3 h, 572 battles.

| Updates | Run | Win rate | Holds fire | Damage dealt vs taken | Entropy | Value loss |
|---|---|---|---|---|---|---|
| 2751–2775 | **New** | 0.599 | 5% | 3.23 vs 2.70 | 4.60 | 1.37 |
| 2751–2775 | Original | 0.646 | 4% | 3.26 vs 2.58 | 4.04 | 0.15 |
| 2776–2780 | **New** | 0.575 | 5% | 3.26 vs 2.73 | 4.83 | 1.11 |
| 2776–2780 | Original | 0.673 | 8% | 3.23 vs 2.54 | 4.62 | 0.15 |

- **Checks:**
  - All four new settings are active: exploration coefficient 0.003, shaping factor 0.30, window 200 battles.
  - Speed is unchanged at 36.5 s per update (268 decisions/s).
- **Win rate:** slightly below the original over the same updates. The gap is about 2 standard errors on roughly
  500 battles, so it isn't conclusive. Extra exploration can cost a little in the short term.
- **Entropy:** higher than the original, as intended.
- **Value loss** jumped after resuming and is coming down (1.86 → 1.11).
  - The same jump happened after both plain resumes of the original run: 0.26 → 2.51 at update 221, and
    0.15 → 1.65 at update 2908.
  - Each time it settled within 50–100 updates, so it comes from resuming, not from the new settings.
- **How battles ended:** 46% on the time limit, 31% with the enemy fleet sunk, 23% with our fleet lost. That's
  about the same as the original over these updates.

## What to watch next

| When | Check |
|---|---|
| Updates 2850–3000 (about 1–2 h from now) | Where the original slid. The fix is working if the win rate holds about 0.5 or above and hold-fire stays below 0.3. |
| About update 2950 | Value loss should be back below about 0.8 |
| Around update 3050 (about 3 h) | If it has slid like the original, try `fix2` with a lower actor learning rate (`--set lr_actor=1e-4`), or switch to the README route (`bc_terrain` → stage 3) |
| Promotion to stage 3 | Needs 0.65 over 200 battles. Expect a large drop against the Elite AI and self-play. |

**Estimate to the 10,000 limit:** about 7,200 updates. That's about 3 days if it stays on stage 2, or 5–6 days if
most of it is spent on stage 3.

## Housekeeping

- **New helper files:**
  - `Training/run_fix1.sh`: the start command for this run.
  - `Training/show_log.py`: prints a run's updates in terminal format. For example,
    `python3 show_log.py runs/mappo_from2750_fix1 | tail -n 50`, or `python3 show_log.py runs/mappo | less`.
- **Pasting commands:** long one-line commands broke when copied into the terminal (a line break after `--set`).
  Short scripts or `\`-continued lines avoid this.
- **Idle shutdown:** keep the training terminal (JupyterLab terminal 4) open. Its output keeps the space from the
  60-minute idle shutdown.
- **Policy export:** the new run overwrites `Assets/StreamingAssets/RL/naval_policy.bin` every 25 updates. Commit
  it only after choosing a good checkpoint with `evaluate.py`.
- **Git:** branch `Sagemaker` is at `8dab75e` on GitHub. These reports, the settings file and the two helper files
  are not committed yet.
