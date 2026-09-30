# Training — MAPPO for the naval simulation

A recurrent MAPPO trainer for the fleets in this Unity project. The Unity side lives in
`Assets/Scripts/RL/`, the Python side here. For how it all works (observations, actions, rewards,
networks, the algorithm, curriculum, export), with diagrams, see [../RL_README.md](../RL_README.md).

## Quick start

```bash
# 1. build the headless training player (works with the editor open: it builds a synced copy)
Training/build_player.sh            # or in the editor: Naval > RL > Build Training Player (Linux)

# 2. train with 8 parallel players
cd Training
python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --workers 8 --run-name first

# 3. watch it learn in the editor: tick GameBootstrap > Rl Training Server, press Play, then
python train.py --ports 5005 --run-name watch

# the recipe that works for the full game: imitate the rule-based AI first, then MAPPO
python bc.py collect --battles 120          # rule AI vs rule AI, every decision labelled (~7 min)
python bc.py train                          # behaviour cloning (~3 min on a GPU)
python train.py --unity-binary ../Builds/NavalTrainer/NavalTrainer.x86_64 --run-name full_mappo \
    --init-from runs/bc/bc.pt --set start_stage=3 --set actor_freeze_updates=10 --set lr_actor=5e-5 \
    --set ent_coef=0.002 --set ent_coef_final=0.0005 --set gamma=0.995 --set minibatches=16

# check the trainer without Unity
python train.py --mock --workers 4 --set total_updates=50
python -m pytest tests
```

## Checking and watching

| Command | What it does |
|---|---|
| `python -m pytest tests` | trainer tests, plus a parity test showing the in-game C# policy matches PyTorch to about 1e-7 |
| `./check_paths.sh` | runs every curriculum stage, self-play against snapshots and every ablation mode briefly against the real game |
| `python probe_actions.py` | issues hand-written orders in the archipelago battle and checks each one works: move through islands without grounding, radar, smoke, torpedoes, hold fire. Also films it to `runs/probe/battle.mp4` |
| `python watch.py --checkpoint runs/<run>/checkpoints/latest.pt` | a trained policy plays one battle, filmed to `runs/watch/battle.mp4` and `.gif` |
| `python bc.py collect` / `python bc.py train` | records rule-AI battles with every decision labelled in the policy's action heads, then trains the actor to imitate them (the warm start for MAPPO) |
| `python evaluate.py --checkpoint <ckpt> --stage 3` | win rate and damage against the rule AI over N battles; `--random` and `--scripted` give baselines |
| `python watch_checkpoints.py --run <run>` | evaluates every checkpoint of a running training run automatically (96 battles, same seed each time) and appends a row to `runs/<run>/evals/summary.jsonl` |
| `NavalTrainer.x86_64 -rlDemo both` | the game itself, straight into a 6v6 with the exported policy flying both fleets (`enemy` = you against it, `player` = it against the rule AI) |

## Playing against the trained policy

Every `export_every` updates the trainer writes `Assets/StreamingAssets/RL/naval_policy.bin`. In the
game's setup screen, **Enemy AI: Learned** makes it your opponent, and **Your fleet: Learned** lets
you watch it command your side. The policy runs in plain C# (`RLPolicy.cs`) with no extra package,
one decision per ship per second, with the forward passes spread over several frames.

Requires Python 3.10+, PyTorch and NumPy. TensorBoard is optional and used if it is installed, and
PyYAML is only needed for `--config file.yaml`. Override any value in `naval_rl/config.py` with
`--set key=value`, for example `--set workers=12 --set rollout=512`.

Each run writes to `runs/<name>/`:

| File | Contents |
|---|---|
| `metrics.jsonl` | one line per update: losses, entropy, KL, win rate against the rule AI, curriculum stage, throughput |
| `episodes.jsonl` | one line per battle: result, learner side, opponent, and behaviour metrics for both fleets |
| `checkpoints/latest.pt` | everything needed for `--resume` (weights, optimisers, value normaliser, league) |
| `snapshots/` | frozen past policies for the self-play league |
| `unity_logs/` | one player log per worker |

Every `export_every` updates the policy is written to `Assets/StreamingAssets/RL/naval_policy.bin`,
which the game loads for in-game inference.

## Results so far

Full game (6v6 domination against the **Elite** rule AI), 48 battles each, sampled actions:

| Player | Win rate | Damage dealt | Damage taken |
|---|---|---|---|
| random legal actions | 0.23 ± 0.06 | 4.77 | 3.58 |
| behaviour cloning v1 (movement labels from steering only) | 0.71 ± 0.07 | 5.27 | 3.45 |
| behaviour cloning v2 (zone orders labelled from the rule AI's assignments) | 0.60 ± 0.07 | 4.95 | 4.65 |

Behaviour cloning: 120 Elite-vs-Elite battles (stages 1-4, about 890k decisions, 80% usable labels).
Held-out accuracy per head: move 0.65, speed 0.96, target 0.93, fire 1.00, torpedo 1.00, ability 0.98.
v1 barely ever chose "go to zone" (the rule AI reaches zones by steering, so its labels said "move"), and
in a watched game it won the fighting but lost on points. v2 labels zone orders from the fleet commander's
assignments (72% of its movement labels in domination) and copies the rule AI's objective play more
faithfully; the two win rates are within their error bars. MAPPO fine-tuning starts from v2 (`runs/full_mappo`, evaluated every 25 updates in
`runs/full_mappo/evals/summary.jsonl`).

Battleship duel (stage 0) learned from scratch, for comparison:

Stage 0 (battleship duel against the Recruit AI), 96 battles each, sampled actions:

| Player | Win rate | Damage dealt | Damage taken |
|---|---|---|---|
| random legal actions | 0.30 ± 0.05 | 0.56 | 0.99 |
| trained, 66 updates (~540k decisions, about 25 minutes) | 0.42 ± 0.05 | 0.65 | 0.91 |
| scripted tactic (auto target, fire at will, broadside) | 0.55 ± 0.07 | 0.87 | 0.69 |

Those settings (rollout 1024, 4 minibatches, 4 epochs, actor learning rate 2e-4, gamma 0.99, win/loss
±1) are now the defaults. The earlier ones (256-step rollout, gamma 0.995, ±3 win bonus) left the duel
too noisy to learn from. Evaluate with sampled actions: early policies
sit near 50/50 on fire vs hold, and always taking the likelier option can mean never firing.

### Experiments that did not learn (kept for the record)

| Run | Change | Result (96 battles, sampled) |
|---|---|---|
| stage0_v3 resumed at update 66 | ammo cooldown added and entropy schedule accidentally reset | update 100: 0.30 ± 0.05, back to random level |
| stage0_v4 from scratch | decision every 2 s, gamma 0.995, lambda 0.98 | update 25: 0.28 ± 0.05, update 50: 0.23 ± 0.04 |

Learning the duel from random play is slow and fragile, which is why the next step is to imitate the
rule-based AI first (behaviour cloning) and then fine-tune with MAPPO.

## How each design problem is handled

| Problem | Where it is solved |
|---|---|
| The actions had no way to aim, and throttle and rudder alone are too hard to learn | `RLActions.cs`. Six masked heads: **move** (keep = carry on with the last order, 8 compass legs, close, broadside, open range, regroup, or one of the zones), **speed**, **target** (auto = the ship's own gunnery choice, or focus a contact), **fire / hold fire**, **torpedo** (only legal with a real firing solution), **ability**. The existing autopilot, gun lead and turret training carry them out. `--set action_mode=lowlevel` swaps in direct rudder and throttle as an ablation. |
| The critic was limited to what the team knows | `RLObservation.cs` sends the actor only fog-of-war-legal information. The critic also gets the true state of every enemy, next to what the team believes about it. `--set critic_mode=belief` removes the privileged columns (ablation). |
| A fixed 178-number vector can't cover fleets of 1 to 30 | Entities (self, allies, contacts, zones) go through a transformer, and targets and zones are chosen with pointer heads (`model.py`). No layer depends on how many ships there are. |
| The reward was never defined | `RLRewardTracker.cs` reports raw components: score margin, damage dealt and taken, zones, kills, friendly fire, result, plus per-ship damage, **spotting damage** and friendly fire. `rewards.py` holds the weights. The team part is zero-sum, and shaping fades out over training. |
| Updating after each episode doesn't fit 20-minute battles | Fixed-length rollouts (`rollout` decisions per worker) with automatic resets (`mappo.py`). The game's own time limit counts as a real ending because time remaining is in the observation. A rollout cut mid-battle is bootstrapped from the critic. |
| Game logic ran on frame-rate-dependent `Time.deltaTime` | Training sets `Time.captureDeltaTime`: every frame is exactly `sim_dt` (20 ms) of game time and physics steps once per frame, the same dt as playing at 50 fps. A decision is `decision_period / sim_dt` frames. |
| Global singletons prevent several battles in one scene | One battle per Unity process. `env.py` launches N headless players on consecutive ports and steps them in parallel (send to all, then receive from all). |
| `ShipAI` assumed the Player fleet is always human; damage events dropped the attacker | `Ship.Controller` is `Human`, `RuleAI` or `Learned`, and either fleet can have any of them (`GameManager.SetTeamController`). `OnShipDamaged` now carries the attacker. `Contact.spotter` records which ship produced each contact, so spotting damage can be credited. |
| Partial observability | GRU actor, trained on 32-step chunks and reset at episode starts. |
| Only ever training against the scripted AI | `league.py`: curriculum stages against Recruit → Veteran → Elite, promotion on rolling win rate, then self-play (latest self, prioritised past snapshots, and the rule AI kept as an anchor). The learner's side is randomised every episode. |

## Curriculum

| Stage | Battle | Opponent | Promote at |
|---|---|---|---|
| 0 `bb_duel` | `scenarios/stage0_bb_duel.json`: 1 battleship a side, open sea, 3 min | Recruit | 55% win (random play: 31%, a simple scripted tactic: 55%) |
| 1 `koth_3v3` | `scenarios/stage1_koth_3v3.json`: BB+CA+DD, one central cap, 6 min | Veteran | 65% |
| 2 `archipelago_3v3` | `scenarios/stage2_archipelago_3v3.json`: islands, three caps, 8 min | Veteran | 65% |
| 3 `domination_6v6` | procedural domination, random weather, 15 min; self-play starts here | Elite + league | 60% |
| 4 `open` | random battlefield, 4–8 a side, 20 min | Elite + league | — |

The scenario files use the Scenario editor's JSON format, so any battle saved in the editor can be
used as a stage. Every episode jitters ship positions and headings slightly, and the terrain is only
rebuilt when the stage changes, so resets are fast.

## Wire protocol

Lockstep TCP, one connection per player (`naval_rl/protocol.py` ⇄ `RLWire.cs`). A frame is a uint32
length, a uint32 JSON length, a JSON header, and then raw little-endian float32 or int32 arrays
described by the header's `"arrays"` list.

| Trainer sends | Environment replies |
|---|---|
| `init` (padding caps, action mode, decision period, sim dt) | `spec`: every dimension, feature name, action head and reward component name |
| `reset` (scenario or procedural battle, which sides are learned, opponent difficulty) | first `obs` |
| `step` with `actions` int32 `[2, max_team, 6]` | `obs`: arrays with a leading team axis of 2, rewards, terminal flag, and on the final step the battle stats |
| `close` | the player quits |

Nothing is hard-coded on the Python side; all sizes come from the `spec`.
