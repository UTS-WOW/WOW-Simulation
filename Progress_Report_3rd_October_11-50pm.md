# Training Progress Report — 3rd October, 11:50 pm

All times are AEST (UTC+10). Figures come from `Training/runs/mappo/metrics.jsonl` (one line per update)
and `Training/runs/mappo/episodes.jsonl` (one line per battle), read at update 2918.

## Summary

- **Run:** `runs/mappo`, MAPPO learned from scratch, curriculum stages 0 → 4.
- **Progress:** update **2918**, now on **stage 2 (`archipelago_3v3`)**, two promotions so far.
- **Training time so far:** **25.7 h** over three sessions, **23.9 M decisions**, **85,851 battles**.
- **Limit:** `total_updates` raised from 3000 to **10,000** at update 2907. The run is still going.
- **Opponent so far:** only the rule-based AI, at Recruit (stage 0) and Veteran (stages 1–2). The Elite AI and
  self-play start at stage 3.
- **Current form:** stage 2 started strongly (0.60–0.70 win rate) but has slipped to about 0.35 over the last
  60 updates. A similar swing happened in stage 1 and recovered.

## Setup

| | |
|---|---|
| Machine | SageMaker `ml.g4dn.2xlarge` (8 vCPU, 32 GB RAM, Tesla T4), space `TeamWow` |
| Game | Headless build `NavalTrainerServer/NavalTrainerServer.x86_64`, 8 copies in parallel (ports 5005–5012) |
| Rollout | 1024 decisions per game per update → 8192 decisions per update |
| Time step | 1 decision = 1 s of game time (50 physics frames of 0.02 s), simulated as fast as the CPU allows |
| Algorithm | Recurrent MAPPO: transformer entity encoder + GRU actor, centralised transformer critic |
| Model size | Actor 587,560 parameters, critic 399,234 parameters |
| Key settings | lr 2e-4 / 5e-4, gamma 0.99, lambda 0.95, clip 0.2, 4 epochs × 4 minibatches, GRU chunk 32 |
| Schedules | Entropy bonus 0.01 → 0.001 over 300 updates; shaping rewards 1.0 → 0.1 over 1500 updates |

Command currently running (from `Training/`):

```
python3 train.py --unity-binary ../NavalTrainerServer/NavalTrainerServer.x86_64 --resume runs/mappo/checkpoints/latest.pt --set total_updates=10000
```

## Timeline

| Session | Updates | Started | Ended | Duration |
|---|---|---|---|---|
| 1 | 1–220 | Fri 2 Oct 22:01 | Fri 2 Oct 23:15 (stopped with Ctrl+C) | 1.2 h |
| 2 | 221–2907 | Fri 2 Oct 23:23 | Sat 3 Oct 23:43 (stopped to raise the limit) | 24.3 h |
| 3 | 2908– | Sat 3 Oct 23:44 | running | — |

Both stops were clean: a checkpoint was saved, and the resumed run continued at the next update number with
no gaps or duplicate updates in the logs.

| Event | Update | Time |
|---|---|---|
| Promoted to stage 1 (`koth_3v3`) | 756 | Sat 3 Oct 02:22 |
| Promoted to stage 2 (`archipelago_3v3`) | 2731 | Sat 3 Oct 21:56 |

## Stage by stage

| Stage | Opponent | Updates | Update time | s / update | dec/s | Battles | Battles / update | Typical battle | Win rate (whole stage) | Promotion bar |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 `bb_duel` (1v1 battleships) | Recruit | 1–755 (755) | 4.2 h | 20.1 | 459 | 35,606 | 47 | 180 s (limit 180 s) | 0.338 | 0.55 / 100 battles |
| 1 `koth_3v3` (capture a zone) | Veteran | 756–2730 (1975) | 19.6 h | 35.7 | 277 | 46,723 | 24 | 360 s (limit 360 s) | 0.185 | 0.65 / 100 battles |
| 2 `archipelago_3v3` (islands, smoke, radar) | Veteran | 2731–2918 (188 so far) | 1.9 h | 36.1 | 272 | 3,522 | 19 | 480 s (limit 480 s) | 0.555 | 0.65 / 150 battles |

Results (wins / draws / losses):

| Stage | W / D / L | Time limit | Enemy fleet sunk | Own fleet lost |
|---|---|---|---|---|
| 0 | 11,745 / 601 / 23,260 | 82% | 9% | 9% |
| 1 | 8,632 / 7 / 38,084 | 70% | 14% | 16% |
| 2 | 1,952 / 4 / 1,566 | 51% | 28% | 21% |

### Stage 0 — `bb_duel` (4.2 h)

Win rate per 100 updates:

| 1–100 | 101–200 | 201–300 | 301–400 | 401–500 | 501–600 | 601–700 | 701–755 |
|---|---|---|---|---|---|---|---|
| 0.32 | 0.33 | 0.34 | 0.33 | 0.30 | 0.29 | 0.39 | 0.45 |

- It stayed around random level (about 0.30) for 600 updates, then improved quickly.
- It was promoted when one lucky 100-battle stretch reached 0.55. Its true win rate was about 0.45.

### Stage 1 — `koth_3v3` (19.6 h)

Win rate over the stage (50-update blocks, grouped):

| Updates | Win rate | |
|---|---|---|
| 756–1005 | 0.08–0.10 | Big drop on arrival: new task (3 ships, capture zone) |
| 1006–1255 | 0.06–0.07 | Low point |
| 1256–1405 | 0.10 → 0.23 | First real learning |
| 1406–1605 | 0.13–0.16 | Plateau |
| 1606–1805 | 0.22–0.28 | |
| 1806–1955 | 0.16–0.22 | |
| 1956–2105 | 0.33–0.40 | Peak |
| 2106–2255 | 0.19 → 0.07 | Collapse |
| 2256–2455 | 0.12–0.19 | Recovery |
| 2456–2705 | 0.23–0.35 | |
| 2706–2730 | 0.46 | Promoted at 2731 (true level about 0.45 against a 0.65 bar) |

Behaviour, learner vs rule AI (averages per battle):

| | Learner | Rule AI |
|---|---|---|
| Damage dealt (hulls) | 2.82 | 2.92 |
| Time of first zone capture | 349 s | 179 s |
| Radar uses | 2.7 | 0.8 |
| Destroyers concealed within gun range | 53% | 27% |
| Share of time holding fire | 38% | 12% |

It fought about evenly but captured the zone much later than the rule AI, so most battles were lost on points
at the time limit.

### Stage 2 — `archipelago_3v3` (1.9 h so far)

| Updates | 2731–2780 | 2781–2830 | 2831–2860 | 2861–2890 | 2891–2918 |
|---|---|---|---|---|---|
| Win rate | 0.62 | 0.55 | 0.65 (peak 0.70) | 0.51 | 0.39 |

- **Strong start:** unlike stage 1 there was no collapse on arrival. The 3v3 skills carried over to the island
  maps.
- **Recent slide:** it started at about update 2861, roughly 50 updates *before* the restart at 2908, so the
  restart is not the cause. It looks like the same kind of swing as stage 1's collapse at 2106–2255.
- **Behaviour:** better than the rule AI at the fighting itself.

  | | Learner | Rule AI |
  |---|---|---|
  | Damage dealt (hulls) | 3.18 | 2.75 |
  | Radar uses | 2.8 | 0.2 |
  | Destroyers concealed within gun range | 81% | 42% |
  | Groundings | ~0 | 0 |

## Checkpoints

In `Training/runs/mappo/checkpoints/` (about 12 MB each):

- Numbered every 250 updates: `update_000250.pt` … `update_002750.pt` (11 files), with `update_003000.pt` next.
- `latest.pt` is overwritten every 25 updates.
- Useful ones:
  - **750:** end of stage 0.
  - **2000:** stage 1 peak period.
  - **2750:** start of stage 2.

Each checkpoint holds the actor, critic, optimisers, league state (stage, win window) and config. Any of them
can be resumed into a **new** run folder with `--run-name`, or evaluated with `evaluate.py` / `watch.py`.
The policy is also exported to `Assets/StreamingAssets/RL/naval_policy.bin` every 25 updates.

## Code review

- **MAPPO implementation:** reviewed line by line, no bugs found (advantage calculation, PPO and value losses,
  death masking, GRU state handling, masks, data indexing).
- **Tests:** the 10 related tests pass (advantages, model, rewards, league, end-to-end mock training).

## Observations and risks

1. **Promotion is noisy.** It triggers the first time any 100-battle window reaches the bar. Both promotions
   so far came at a true win rate of about 0.45, well below the bar (0.55 and 0.65).
2. **Large swings within a stage.** Stage 1 went from 0.40 down to 0.07 and back. Stage 2 has gone from 0.70
   to 0.35 so far. Worth watching rather than acting on immediately.
3. **Exploration and shaping already at their minimum.** The schedules run on the global update count, not
   per stage. The entropy bonus reached its floor at update 300 and the shaping rewards reached their 10% floor
   at update 1500. So the harder stages get little exploration and mostly sparse rewards (win/loss and score).
4. **Short planning horizon.** gamma 0.99 is about 100 s ahead. Battles are now 8 minutes, and 15 minutes from
   stage 3; the config itself suggests 0.995 for those.
5. **Stage 3 is the main risk:** 6v6, random maps, the **Elite** AI, a 0.60 bar, and only about 12% of battles
   counting towards promotion (self-play takes 80%). For comparison, the team's behaviour-cloned policy reached
   0.38 against Elite.

## Estimates

Rough estimates, not measurements:

- **Update speed at stage 3:** about 60–80 s per update, estimated from how time grew between stages 0 and 1.
- **Updates left to the 10,000 limit:** about 7,080.
  - If most of them are at stage 3: about 5–6 days, finishing around **Thu 8 – Fri 9 Oct**.
  - If it stays at stage 2: about 3 days.
- **Reaching stage 4 within the two-week window:** plausible, if stage 3 keeps improving.

Decision points:

| By about | Should be |
|---|---|
| Mon 5 Oct | In stage 3 |
| Thu 8 Oct | Stage 3 win rate against Elite clearly rising |

If not, consider either of these:

- Resetting the entropy and shaping schedules at each promotion.
- Switching to the README's main line: resume `bc_terrain` (on `main`) at stage 3.

## Housekeeping

- **Idle shutdown:** the space shuts down after 60 min with no JupyterLab terminal or notebook activity.
  Training output in its terminal counts as activity, so keep that terminal open and don't run training
  with `nohup` in the background.
- **Git:** branch `Sagemaker`, commit `8dab75e`, pushed. It contains:
  - the trainer build
  - `STRUCTURE.md`
  - the LineDrawer change
  - the deletion of the `bc`, `bc_terrain`, `full_mappo` and `mappo_terrain` checkpoints (still on `main`)

  `naval_policy.bin` changes with every export. `.ipynb_checkpoints/` folders and this report are untracked.
- **Monitoring:**
  - `tail -n 3 Training/runs/mappo/metrics.jsonl` shows the latest updates.
  - Any 50-update win-rate block can be recalculated from `episodes.jsonl`.
