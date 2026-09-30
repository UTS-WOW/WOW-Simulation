# The Environment

Everything about the world a match is fought in: coordinates and units, how the battlefield is
generated, what the terrain does to ships, what the weather and fog of war do to information, how the
objectives score, and how a match starts and ends.

This document describes the *environment* — the simulated world and its rules. For ship statistics and
how the real Tier 10 data was converted see [SHIPS.md](SHIPS.md); for controls see
[README.md](README.md).

## Contents

1. [Coordinate system and units](#1-coordinate-system-and-units)
2. [Terrain](#2-terrain)
3. [Battlefield configuration](#3-battlefield-configuration)
4. [Deployment geometry](#4-deployment-geometry)
5. [Objectives](#5-objectives)
6. [Weather](#6-weather)
7. [Fog of war and information](#7-fog-of-war-and-information)
8. [Match flow](#8-match-flow)
9. [Determinism and seeding](#9-determinism-and-seeding)
10. [Reading the environment from code](#10-reading-the-environment-from-code)
11. [Notes for reinforcement learning](#11-notes-for-reinforcement-learning)

---

## 1. Coordinate system and units

| Quantity | Convention |
|---|---|
| World | 2D, XY plane. Z is used only for draw ordering (submerged hulls sit at `z = 0.2`) |
| Origin | `(0, 0)` is the centre of the map, and the centre of the contested line |
| Extent | `GameConfig.WorldSize = 4000` units, spanning `[-2000, +2000]` on both axes |
| Scale | **1 unit ≈ 10 m**, so the battlefield is about **40 km × 40 km** |
| Heading | Compass degrees, `0 = north`, increasing clockwise. Sprites point up, so `transform.z = -heading` |
| Speed | Units per second. Displayed as knots: `Ship.SpeedKnots = abs(speed) × 19.4` |
| Height | `h > 0` is land, `h < 0` is water. `Depth = -h`, so `0` is the shoreline and `1` is the abyss |

Gravity is zeroed at boot (`Physics2D.gravity = Vector2.zero`); ships are dynamic `Rigidbody2D`
bodies with capsule hull colliders that collide only with each other.

---

## 2. Terrain

The battlefield is a **height field**, generated procedurally from a seed. The same field drives the
water shader, the navigation grid and the AI's terrain reasoning — there is no separate collision
geometry for land.

- Resolution is **512 × 512** samples across 4000 units, so one height cell is **7.8 units (~78 m)**.
- `WorldMap.SampleHeight(v)` bilinearly interpolates the field at any world point.
- The field is uploaded to a `RFloat` texture for the shader. Texture uploads clamp to `0..1`, so
  height is stored biased as `h × 0.5 + 0.5` and decoded back in `NavalOcean.shader`.

### Height profile

**Above water** — islands are domes: `h = clamp01(land)^0.65`, modulated by a 4-octave FBM roughness
term, with the dome radius itself warped by low-frequency noise so coastlines are irregular rather
than circular.

**Below water** — depth is built from three terms:

| Term | Effect |
|---|---|
| Continental shelf | `clamp01(nearestShore / 240)^0.85` — water deepens over the first 240 units (2.4 km) off any beach |
| Basin | Low-frequency FBM scaling the shelf between `0.78×` and `1.1×` |
| Offshore banks | A separate FBM; where it exceeds `0.78` the depth is capped at `0.17`, creating broad **shoals** that only light hulls can cross |

Beyond 86% of the half-width the map fades to open ocean (depth `1.0`), so the borders are always
navigable and no fleet can be trapped against an edge.

### Draft and grounding

A ship can enter water where `depth ≥ draft × 0.36` (`WorldMap.DraftToDepth`).

| Class | Draft | Required depth | Clears the shelf at roughly |
|---|---|---|---|
| Submarine | 0.28 | 0.101 | ~17 units (170 m) off the beach |
| Destroyer | 0.30 | 0.108 | ~17 units (170 m) |
| Cruiser | 0.52 | 0.187 | ~35 units (350 m) |
| Transport | 0.60 | 0.216 | ~42 units (420 m) |
| Battleship | 0.72 | 0.259 | ~50 units (500 m) |

This is the single most important environmental asymmetry: **a destroyer can cut inside an island
chain that a battleship has to sail around**, and shoals in open water are usable cover for light
hulls. `WorldMap.DepthGradient(v)` returns the outward slope of the sea floor and is what ships use
to sheer away from shallows.

### Navigation grid

`NavGrid` rasterises the height field into a pathfinding grid at `GameConfig.NavCellSize = 16`
units (160 m), giving a **250 × 250** cell grid. Each cell samples its centre *and* its corners and
keeps the shallowest result, so a thin spit of land still blocks the cell rather than falling between
samples. It stores per-cell depth and a coastline distance, and runs **draft-aware A\***: the same
grid serves every class, but each query rejects cells too shallow for that hull. Path requests are
queued and throttled to `GameConfig.PathThrottlePerFrame = 3` solves per frame.

`NavGrid.NearestNavigable(pos, draft)` snaps any point to water that hull can actually float in — it
is what keeps spawns and move orders off the beach.

---

## 3. Battlefield configuration

A match is configured by a `MapConfig` chosen on the setup screen. Three presets set coherent
defaults; every field can then be overridden individually.

| | Ocean Archipelago | Open Sea | Strait Clash |
|---|---|---|---|
| Island density | Medium (`1.0×`) | Low (`0.35×`), large islands forced to **zero** | High (`1.7×`), scattered islands suppressed |
| Flag layout | Three Point Domination | King of the Hill | Two Flag Assault |
| Capture radius | 150 units (1500 m) | 190 units (1900 m) | 130 units (1300 m) |
| Spawn distance | `0.375` of half-width | `0.36` | `0.55` |
| Character | Cover, flanks and ambush | No cover at all — gunnery and angling decide it | Everything funnels through one channel |

**Island density** multiplies three base counts: 6 large, 9 small, 14 rocks.

| Density | Multiplier | Large (120–235u) | Small (55–105u) | Rocks (14–34u) |
|---|---|---|---|---|
| Low | 0.35 | 2 | 3 | 5 |
| Medium | 1.0 | 6 | 9 | 14 |
| High | 1.7 | 10 | 15 | 24 |
| Procedural | random 0.3–1.8 | varies | varies | varies |

Each island gets a **hazard shelf** of `radius + 110` units (rocks: `+45`) — the shallow ring around
it that a deep hull must respect.

Placement makes 60 attempts per island and rejects any position that comes within:

- `DeployRadius + radius + 150` of **any of the six squadron spawns**,
- `190 + radius` of **any capture zone** (caps are always open water),
- `sum of radii + 130` of **another island** (rocks: `+40`).

**Open Sea** forces large and small island counts to zero and caps rocks at 4. **Strait Clash**
instead places two fixed landmasses of radius `Half × 0.46` on the flanks, leaving a channel whose
half-width is `max(Half × 0.30, DeployRadius + Half × 0.11 + 90)` — it widens automatically so a
large fleet still fits through.

---

## 4. Deployment geometry

Both fleets deploy as **three squadrons — left flank, centre, right flank** — facing each other
across the centre line. Classes are dealt round-robin, so each squadron is a balanced task force
rather than a wing of battleships. Player squadrons are pre-bound to control groups `1`, `2`, `3`.

- Base line: `y = ±Half × spawnDistance`
- Flank offset: `x = ±Half × 0.55`, except **Strait Clash** which uses `±Half × 0.11` so the flank
  squadrons deploy inside the channel instead of hard aground on the landmasses.

`DeployRadius = clamp(150 + ceil(fleetSize / 3) × 34, 200, 560)`, computed **before** generation so
island placement leaves the right amount of sea room:

| Fleet size | Ships per squadron | Deploy radius |
|---|---|---|
| 6 | 2 | 218 |
| 12 | 4 | 286 |
| 18 | 6 | 354 |
| 24 | 8 | 422 |
| 30 | 10 | 490 |

Squadrons of 6 or fewer form a wedge; larger ones pack into a block. Every spawn is snapped through
`NavGrid.NearestNavigable`, and snapped positions are jittered by up to 14 units so two hulls never
land on the same cell centre and shove each other apart when physics starts.

Resulting approach distances (centre squadron to its nearest cap):

| Preset | Base line | Fleet separation | Nearest cap | Approach | DD / BB arrive |
|---|---|---|---|---|---|
| Ocean Archipelago | 750 | 15.0 km | B at `(0, 0)` | **750 units (7.5 km)** | 6:13 / 9:00 |
| Open Sea | 720 | 14.4 km | the single cap at `(0, 0)` | **720 units (7.2 km)** | 5:58 / 8:38 |
| Strait Clash | 1100 | 22.0 km | own flag at `(0, -605)` | **495 units (4.95 km)** | 4:06 / 5:56 — the enemy flag is 1705 away |

Ships run at real speeds (a Yamato makes 27 knots, or 1.39 units/second), so the start lines are set
just outside mutual battleship spotting range — 15.0 km against a 14.1 km battleship concealment.
Neither battle line can see the other at the moment the clock starts, which keeps the approach a real
phase of the battle instead of an immediate gun duel.

---

## 5. Objectives

### Capture zones

A zone is a `CircleCollider2D` trigger evaluated **4 times a second**. Transports never count toward
capture, and a submarine must be at surface or periscope depth to hold ground.

| Occupancy | Behaviour |
|---|---|
| Both fleets present | **Contested** — the meter freezes exactly where it is |
| One fleet present | Meter moves at `sqrt(abs(netHulls)) / 40` per second |
| Empty, never captured | Bleeds back to neutral over 120 s |
| Empty, already owned | Returns to *fully held* in 32 s — the owner keeps it |

`Progress` runs `-1` (fully enemy) to `+1` (fully player). `BaseCaptureTime = 40 s`, so one ship
flips a neutral zone in 40 s, two in ~28 s, four in 20 s — extra hulls help with diminishing returns.

**Capture is permanent.** Ownership transfers *only* when a meter completes; it is never given up
passively. Once taken, a zone keeps scoring whether or not anyone stays behind, and the only way to
lose it is for the other side to sail in and complete a capture of their own. This frees a fleet to
move on after taking a point instead of parking on it, and makes the objective a clean discrete
achievement rather than a value that quietly bleeds away.

Zones expose a coarse discrete `ZoneState` — `Neutral`, `Capturing`, `Captured`, `Contested` —
alongside the continuous `Progress`, plus `FullyCaptured` and `UnderAttack`.

### Flag layouts

| Layout | Zones | Positions |
|---|---|---|
| Three Point Domination | 3 | `(-1040, 0)`, `(0, 0)`, `(1040, 0)` — one per squadron front |
| King of the Hill | 1 | `(0, 0)` — everything converges |
| Two Flag Assault | 2 | `(0, ±baseLine × 0.55)` — one in front of each base |

Two Flag Assault is the interesting one: you win by taking *theirs*, so both sides must decide how
much to commit forward and how much to leave at home.

### Scoring

```text
ZonePointsPerSecond = FullMapPointsPerSecond (3.6) / zoneCount
ScoreToWin          = 1000
KillPoints          = 12   (awarded per enemy hull sunk)
```

The per-zone rate is **divided by the number of zones** so that holding the whole map wins in about
the same time on every layout. Without this a one-flag map cannot reach 1000 points inside the time
limit and every King of the Hill match ends on the clock.

| Layout | Zones | Points/s per zone | Holding everything → 1000 |
|---|---|---|---|
| King of the Hill | 1 | 3.60 | 278 s |
| Two Flag Assault | 2 | 1.80 | 278 s |
| Three Point Domination | 3 | 1.20 | 278 s |
| Capture and Control | 5 | 0.72 | 278 s |

### Ports

Each side has one harbour, tucked 170 units behind its deployment line, away from the contested
middle. A ship inside the **90 unit service radius** is refuelled (9% capacity/s), rearmed
(16%/s) and given heavy repair. Ports have 5000 HP and can be destroyed.

---

## 6. Weather

Weather is global, changes the sea state and — more importantly — **changes what everyone can see**.

| | Visibility | Gun dispersion | Sea state | Fog overlay |
|---|---|---|---|---|
| Clear | 1.00× | 1.00× | 1.0 | none |
| Fog | **0.42×** | 1.25× | 0.55 | 0.24 |
| Rain | 0.74× | 1.15× | 1.35 | 0.12 |
| Storm | 0.55× | **1.45×** | 2.1 | 0.18 |

The starting weather is chosen on the setup screen, then it evolves: every **80–170 s** the system
rolls a transition from a per-state table (Clear tends to stay clear; Storm decays to Rain), and
blends to the new profile over **18 s**. Rain and storm precipitation is spawned on
`unscaledDeltaTime`, so rainfall does not visually accelerate with time compression, and is skipped
entirely above 620 orthographic size.

Wind direction drifts continuously and drives the ocean shader's wave motion.

---

## 7. Fog of war and information

This is the part of the environment that matters most tactically. Detection is evaluated **8 times a
second** for both teams independently. Neither side — including the AI — is given information it has
not earned.

### Can I see you?

For each observer/target pair within a hard 2800-unit (28 km) cutoff — it has to exceed the longest
gun on the map, or a battleship could shell targets it was structurally unable to see:

```text
spotted  if  distance ≤ min(target.Detectability, observer.EffectiveSpotRange)
             and line of sight is clear
```

**Line of sight** is sampled along the ray in steps of ~26 units. **Land always blocks** (any
height above `0.06`). **Smoke blocks optical detection only** — hydrophones and close-range lookouts
see straight through it.

### How far can I see?

```text
EffectiveSpotRange  = spotRange × weatherVisibility × sensorIntegrity + abilityBonus
EffectiveSonarRange = (sonarRange + abilityBonus × 0.6) × sensorIntegrity
EffectiveHydroRange = hydroRange × sensorIntegrity + abilityBonus
```

Sensor integrity scales from `1.0` down to `0.4` as the sensor system takes damage.

### How far away can I be seen?

Base detectability is modified multiplicatively:

| Condition | Multiplier |
|---|---|
| Haze (any weather below clear) | up to **0.62×** — bad weather hides everyone |
| Speed | `0.82×` stopped → `1.12×` at flank speed (wake) |
| Fired guns in the last 12 s | **1.55×** — muzzle flash and smoke give the position away |
| On fire | `1.35× + 0.15×` per fire stack |
| Submerged | periscope `0.45×`, submerged `0.2×`, deep `0.1×`, and a further `0.7×` when running silent |
| Inside smoke | **0.12×**, but only `0.55×` if you are firing out of it |
| Held by an active sonar ping | floor of **420 units** — you cannot hide |

Detectability never drops below **12 units**.

Note the interaction that drives destroyer play: sitting still in smoke and not firing makes you
nearly invisible; opening fire from that same smoke roughly quadruples your signature.

### Assured detection

Concealment can be defeated outright. Surveillance radar (10 km), hydroacoustic search (5 km), the
submarine's hydrophone (7 km) and submarine surveillance (9 km) all acquire a contact inside their
radius **regardless of how good its concealment is**. Radar reaches through islands as well as smoke;
the acoustic sets only see through smoke, but they also hear submerged boats. Every surface ship also
has a 2 km guaranteed acquisition range — get that close and you are seen whatever you do.

This is the counter to a destroyer sitting invisible on a cap, and it is why a cruiser's radar charge
is often worth more than its guns.

### Submarines

A submerged boat is invisible to normal spotting and can only be found by **active sonar**, itself
reduced by `0.7×` against a deep boat and `0.6×` against one running silent. A boat at periscope
depth leaves a feather that a sharp lookout can still see optically.

### Contact memory

Each team keeps a `Contact` record per enemy hull, holding `lastKnownPosition`, `lastKnownHeading`,
`lastSeenTime`, whether the class was identified, and a state:

- **Confirmed** — visually held right now
- **Unknown** — sonar contact only, position known but not identity
- **LastKnown** — lost; the memory persists for **55 s** before being discarded

The AI dead-reckons dark contacts from `lastKnownPosition + heading × age` with a growing uncertainty
radius, which is how it keeps hunting a destroyer that has slipped into smoke instead of instantly
forgetting it.

---

## 8. Match flow

```text
Menu  →  Deployment  →  Battle  →  Victory / Defeat
```

During **Deployment** the player can drag squadrons around inside their deploy circles and pick a
starting formation. **Battle** begins on command and runs the clock.

### Modes and time limits

| Mode | Objective | Time limit |
|---|---|---|
| Domination | Hold zones to 1000 points, or sink the enemy fleet | 1200 s (20:00) |
| Skirmish | Destroy the enemy task force | 900 s |
| Fleet Battle | Highest fleet strength at the limit | 1080 s |
| Capture and Control | Five zones, 1000 points | 1200 s |
| Escort | Get a transport to the eastern anchorage | 1080 s |

### End conditions, in evaluation order

1. **Escort only** — a transport reaching the anchorage wins; losing every transport loses.
2. Either fleet reduced to **zero hulls**.
3. **Objective modes** — either side reaching **1000 points**.
4. **Time limit** — objective modes compare score (fleet strength breaks an exact tie); other modes
   compare fleet strength.

Fleet strength is `sum of fleetPointCost × 10 × healthFraction` over living hulls, so it accounts for
damage as well as losses.

### Time compression

Speed steps are `0, 1, 2, 4, 8`. `Time.timeScale` is set directly, and `Time.fixedDeltaTime` scales
as `0.02 × clamp(timeScale, 1, 4)` — larger physics steps at high compression rather than hundreds of
solver ticks per second. `Time.maximumDeltaTime` is pinned at `0.33` so a frame spike cannot
teleport the simulation. Pausing sets `timeScale = 0`; the phase itself gates it, so the world is
frozen outside `Battle`.

### Simulation rates

| System | Rate |
|---|---|
| Physics | 50 Hz at 1× (`fixedDeltaTime = 0.02`), stepping down to 12.5 Hz at 4× and above |
| Detection / fog of war | 8 Hz |
| Ship AI | 5 Hz |
| Capture zones | 4 Hz |
| Pathfinding | 3 solves per frame, queued |

---

## 9. Determinism and seeding

`WorldMap.Generate(seed, ...)` calls `Random.InitState(seed)` and `NavalMath.SetNoiseSeed(seed)`, so
**terrain, island placement, ports and zone positions are fully reproducible from the seed**.
`GameBootstrap.seed = 0` picks a random one; any non-zero value reproduces that battlefield exactly.

The *match* is not deterministic beyond generation: gun dispersion, fire chance, weather transitions
and AI tie-breaks all draw from the shared `Random` stream during play, and physics runs at a
variable frame rate. Same seed means same map, not same battle.

---

## 10. Reading the environment from code

The environment exposes a compact, stable surface. Everything below is queryable at any time without
touching rendering or UI.

**World**

```csharp
WorldMap.I.SampleHeight(v)          // > 0 land, < 0 water
WorldMap.I.SampleDepth(v)           // 0 at shoreline, 1 in the abyss
WorldMap.I.IsNavigable(v, draft)    // can this hull float here
WorldMap.I.DepthGradient(v)         // outward slope of the sea floor
WorldMap.I.Islands                  // centre, radius, hazard radius, isRock
WorldMap.I.Zones                    // capture zones
WorldMap.I.Ports                    // harbours
WorldMap.I.PlayerDeployCenters      // left / centre / right, and the enemy equivalents
NavGrid.I.CellDepth(x, y)           // rasterised depth
NavGrid.I.CoastDistance(x, y)       // cells to the nearest shore
```

**Objectives**

```csharp
zone.State            // Neutral | Capturing | Captured | Contested  (discrete)
zone.Progress         // -1 .. +1                                    (continuous)
zone.Owner            // Player | Enemy | Neutral
zone.FullyCaptured    // locked in for the owner
zone.UnderAttack      // someone is actively taking it off its owner
zone.PlayerShips      // qualifying hulls inside, per side
zone.EnemyShips
```

**Information state** — always query this rather than `ShipRegistry` directly if you want to respect
fog of war:

```csharp
DetectionSystem.I.IsVisible(ship, observerTeam)
DetectionSystem.I.GetContact(ship, observerTeam)   // state, lastKnownPosition/Heading, age
DetectionSystem.I.Contacts(observerTeam)
DetectionSystem.I.GatherLiveContacts(team, into)
```

**Match state**

```csharp
GameManager.I.Phase, .Mode, .BattleTime, .TimeRemaining, .TimeLimit
GameManager.I.PlayerScore, .EnemyScore, .PlayerKills, .EnemyKills
GameManager.I.FleetPoints, .EnemyFleetPoints      // damage-weighted fleet strength
GameManager.ZonePointsPerSecond                    // scales with zone count
BattleAssessment.For(team)                         // shared per-team situational picture
```

`BattleAssessment` is worth knowing about: it is recomputed about twice a second per team and holds
the derived picture the AI reasons over — posture, per-zone value and threat, local strength ratios,
and dead-reckoned positions for contacts that have gone dark.

---

## 11. Notes for reinforcement learning

Several environment choices were made deliberately to make the game legible to a learning agent:

- **Discrete objective states.** `ZoneState` gives an unambiguous four-value signal alongside the
  continuous meter.
- **Permanent capture.** A captured zone stays captured, so taking a point is a clean terminal
  achievement rather than a reward that silently decays and has to be re-earned by parking a hull.
- **Fair information.** Both fleets read the same fog-of-war layer, so a policy trained against the
  AI is not learning to beat an omniscient opponent.
- **Zone-count-normalised scoring**, so the reward scale of holding ground does not silently change
  when the map layout does.

Available reward signals: objective score (`PlayerScore` / `EnemyScore`), kills (`KillPoints`), and
damage-weighted fleet strength (`FleetPoints`), which moves continuously with damage rather than only
on hull loss. Natural episode termination is `GamePhase.Victory` / `GamePhase.Defeat`, with
`ResultSummary` describing the cause.

**The training environment is `Assets/Scripts/RL/`, and the MAPPO trainer is `Training/`** (see
[Training/README.md](Training/README.md)). The environment speaks a small lockstep TCP protocol
instead of ML-Agents. Every simulation system is a singleton, so one Unity process holds one battle,
and parallel environments are parallel headless players. Training fixes `Time.captureDeltaTime`, so
every frame is exactly 20 ms of game time no matter how fast the machine runs it, rather than relying
on time compression.
