# Naval Warfare — 2D Fleet Combat (Unity 6)

A top-down 2D naval combat game in the spirit of *World of Warships*.

- **1 to 30 ships per side**
- **Hybrid controls:** captain a single warship yourself, or command the whole fleet as an RTS
- **Three battlefields** with configurable objectives
- **Four combat ship classes:** destroyer, cruiser, battleship, submarine
- **Time compression up to 8x**
- **Fully procedural:** there are no art, audio or prefab assets. Ship sprites, turrets, the
  water/terrain shader, particle atlases and every sound effect are generated in code at runtime.

## Contents

- [Getting started](#getting-started)
- [How a match plays](#how-a-match-plays)
- [Controls](#controls)
- [Ships and combat](#ships-and-combat)
- [Battlefields and objectives](#battlefields-and-objectives)
- [Scenario editor](#scenario-editor)
- [Enemy AI](#enemy-ai)
- [Architecture](#architecture)
- [Tuning](#tuning)
- [Reinforcement learning](#reinforcement-learning)
- [Type-checking without the editor](#type-checking-without-the-editor)
- [Further documentation](#further-documentation)

---

## Getting started

1. Open the project in **Unity 6**.
2. Open `Assets/Scenes/SampleScene.unity`.
3. Press **Play**.

The scene contains a single `GameBootstrap` object that builds the entire game. Its inspector
fields:

| Field | Default | Meaning |
|---|---|---|
| `startMode` | `Domination` | `Domination`, `Skirmish`, `FleetBattle`, `CaptureAndControl` or `Escort` |
| `seed` | `0` | `0` = new map every run; any other value = reproducible map |
| `startWeather` | `Clear` | `Clear`, `Fog`, `Rain` or `Storm` |
| `skipMenu` | off | Skip fleet selection and go straight to deployment |
| `enemyDifficulty` | `Elite` | `Recruit`, `Veteran` or `Elite` — see [Enemy AI](#enemy-ai) |
| `heightmapResolution` | `512` | Terrain sampling resolution |
| `rlTrainingServer` | off | Start as an RL training environment instead of showing the menu |
| `rlPort` | `5005` | Port the training environment listens on |

---

## How a match plays

**Fleet selection → Deployment → Battle → Result**

1. **Battle setup**
   - Set each side's fleet size (1–30, default 6).
   - Choose how the enemy fleet is built: **Balanced**, **Custom slots** or **Mirror yours**.
   - Pick the battlefield.
   - Choose whether you start as **fleet commander** (default) or as **captain** of one ship.

2. **Deployment** (the simulation is paused)
   - Your fleet deploys on your baseline as **three squadrons — LEFT, CENTRE and RIGHT** — facing
     the enemy, with caps A/B/C along the centre line between you.
   - Drag ships to reposition them inside their squadron's area. Drag one across a boundary to hand
     it to the neighbouring squadron.
   - Pick a formation, then press **START BATTLE**.

3. **Battle**
   - You start in RTS fleet command. `Tab` takes the helm of the selected ship; `Tab` again hands
     it back.
   - The three squadrons are pre-bound to control groups, so `1` / `2` / `3` instantly select your
     left, centre and right squadrons.

---

## Controls

`Tab` switches between **Direct control** and **Fleet command**.

### Direct control (you are the captain)

| Input | Action |
|---|---|
| `W` / `S` | Engine telegraph — throttle ahead / astern (the ship has momentum; changes are not instant) |
| `A` / `D` | Rudder to port / starboard; releasing eases it back amidships |
| `Space` | All stop |
| Mouse | Trains the guns; the reticle auto-leads a target it is resting on |
| Left click / hold | Fire the main battery |
| Right click | Torpedo spread along the reticle bearing (arcs are drawn on screen) |
| `1` – `6` | Class consumables (see [Ship classes](#ship-classes)) |
| `X` | Submarine dive / surface |
| `E` | Damage control party |

### Fleet command (RTS)

**Selection**

| Input | Action |
|---|---|
| Left click | Select a ship |
| Drag | Box-select several ships |
| `Shift` + click | Add to selection |
| Double click | Select every ship of that class |
| `1` / `2` / `3` | Select the left / centre / right squadron |
| `Ctrl` + `1`–`9` | Bind the selection to a control group |
| `Ctrl` + `A` | Select the whole fleet |
| Click a ship in the task-force roster | Select it (or take its helm if you are in direct control) |

**Orders**

| Input | Order |
|---|---|
| Right click | Move / attack |
| `Shift` + right click | Queue a waypoint |
| `C` | Attack-move |
| `V` | Patrol |
| `B` | Escort |
| `T` | Focus fire |
| `Space` | Stop |
| `H` | Hold position |
| `R` | Astern |
| `G` | Retreat |
| `Y` | Return to port |
| `Q` | Smoke |

**Formations**

| Key | Formation |
|---|---|
| `F5` | Line ahead |
| `F6` | Line abreast |
| `F7` | Wedge |
| `F8` | Circle |
| `F9` | Defensive screen |
| `F10` | Break formation |

### Camera, time and system

| Input | Action |
|---|---|
| `WASD` / screen edge / middle-drag | Pan the camera (fleet command only — in direct control `WASD` is the helm) |
| Mouse wheel | Zoom (out far enough to see the whole map) |
| `F` | Follow the selected ship (press again to stop) |
| `` ` `` / `Home` | Frame the whole fleet |
| `+` / `-` | Cycle time compression: 1x → 2x → 4x → 8x (also on the status bar) |
| `P` | Pause |
| `F1` | Debug draw (includes the AI's reasoning — see [Enemy AI](#enemy-ai)) |
| `F2` | Reveal the map |
| `F3` | Show the navigation grid |
| `F4` | Command reference |
| `F6` | Dev view |

> **Note:** `F6` is currently bound to both the dev view and the *line abreast* formation, so in
> fleet command it also puts the selected ships into line abreast.

**Dev view (`F6`)** keeps hulls readable at strategic zoom, turns on the `F1` debug layer, and draws:

- every turret's firing arc and blind sector — turrets that can bear on the current target are drawn
  in the team colour, those that cannot are red;
- rings around the selected ship showing its concealment, spotting, gun and torpedo ranges.

---

## Ships and combat

### Ship classes

Every hull is a real Tier 10 ship, and every number below comes from its actual statistics. See
[SHIPS.md](SHIPS.md) for the full conversion.

| Class | Ship | HP | Speed | Concealment | Gun range |
|---|---|---|---|---|---|
| **DD** | Shimakaze | 17 900 | 39 kn | **5.6 km** | 11.4 km |
| **CA** | Des Moines | 50 600 | 33 kn | 10.9 km | 15.8 km |
| **BB** | Yamato | 97 200 | 27 kn | 14.1 km | **26.6 km** |
| **SS** | Balao | 20 200 | 30 kn | 5.9 km surfaced, 2.3 km at periscope depth | 4.0 km (deck gun) |

**Consumables** (keys in direct control):

| Class | `1` | `2` | `3` | `4` | `5` | `6` | `X` |
|---|---|---|---|---|---|---|---|
| **DD** | HE | AP | Torpedoes | Smoke | Engine boost | Damage control | — |
| **CA** | HE | AP | Radar | Hydro | Repair | Damage control | — |
| **BB** | HE | AP | Damage control | Repair | Spotter plane | — | — |
| **SS** | HE | Homing torpedoes | Ping | Hydrophone | Surveillance | Damage control | Dive |

### Spotting decides the battle

Guns out-range eyes by a wide margin: a Yamato shoots 26.6 km but is only *seen* at 14.1 km. A
destroyer that stays hidden is what lets its battle line shoot at all — and radar is what takes that
away from it.

- **Radar** (10 km) spots everything inside its circle, through smoke *and* through islands.
- **Hydroacoustic search** (5 km) and the submarine's **hydrophone** (7 km) see through smoke but not
  islands, and both also detect submerged submarines.

These are the counters to a destroyer sitting invisible on a cap.

### Reading the battlefield

Every ship carries an **overhead class symbol**, drawn at a constant screen size so it stays readable
through smoke, weather and at any zoom:

| Class | Symbol |
|---|---|
| DD | Two diamonds |
| CA | Diamond with one slash |
| BB | Diamond with two slashes |
| SS | Chevron |

Colours: **cyan** = friendly, **crimson** = hostile, **amber** = neutral. Each hull also has an
elongated team-coloured aura along its waterline, so you can see which way a contact is pointing.

### Shells: HE vs AP

| | HE | AP |
|---|---|---|
| Damage | Lower | Full |
| Penetration | Lower | Higher |
| Fire chance | Much higher | Lower |
| Citadel hits | Never | Possible against a broadside target |
| Over-penetration | — | Possible against light hulls |

### Angling and overmatch

A shell that strikes more than **~45°** off the plate normal can ricochet, and **always** ricochets
past **60°** — so pointing your bow at the enemy is how you survive. Two things defeat angling:

- **Improved angles** — Des Moines super-heavy AP only starts ricocheting at 60° and only always
  ricochets at 75°.
- **Overmatch** — a shell defeats any plate thinner than *calibre ÷ 14.3*, **regardless of angle**.
  Yamato's 460 mm guns overmatch 32 mm, which is exactly a cruiser's bow: angling saves a cruiser
  from everything *except* a Yamato.

Results measured over 400 shells per case:

| Shot | Result |
|---|---|
| Yamato AP → angled Des Moines bow | 100% penetration (overmatched) |
| Yamato AP → broadside Des Moines | 34% citadel, 66% penetration |
| Yamato AP → Shimakaze | 100% over-penetration (nothing thick enough to arm the fuse) |
| Des Moines AP → angled Yamato bow | 100% ricochet |
| Des Moines AP → broadside Yamato | 100% penetration, **never a citadel** (450 mm cannot beat a 410 mm belt) |
| Des Moines AP → broadside Des Moines | 37% citadel |

### Turret arcs

Each turret sits at its own point along the hull and can only train within its own arc, so how much
firepower you bring depends on your heading. Turrets track to their arc limit and wait there rather
than returning to centre, so they are ready the moment the hull comes round.

| Yamato heading relative to target | Barrels that can fire |
|---|---|
| 0–30° (bow-on) | 6 of 9 |
| 40–120° (broadside) | **9 of 9** |
| 150° | 6 of 9 |
| 180° (stern-on) | 3 of 9 |

This is the cost of angling: the heading that bounces shells also silences a third of your guns.

### Submarines: ping → homing torpedoes

A sonar **ping** marks a target for about 25 seconds. Torpedoes fired while the mark holds steer
onto it, and the marked ship stays visible on the plot until the lock decays.

---

## Battlefields and objectives

| Preset | Terrain | Default objective |
|---|---|---|
| **Ocean Archipelago** | Scattered islands giving cover and torpedo chokepoints | Three-point domination (A / B / C) |
| **Open Sea** | No cover at all — pure gunnery and angling | King of the Hill (one large central point) |
| **Strait Clash** | Two landmasses squeezing a narrow central channel | Two-flag assault (a flag in front of each base) |

Island density, weather and capture radius (500–2000 m) are adjustable on the setup screen. Each
preset sets its own spawn-to-cap distance so first contact happens early rather than after a
five-minute sail.

### Capturing zones

Zones are `CircleCollider2D` triggers. Ships inside a zone fill its capture meter — about 40 seconds
for one ship, faster with more (with diminishing returns). **If both fleets are inside, the meter
freezes (contested).**

**Captures are permanent.** Once the meter completes, the zone is yours and keeps scoring whether or
not anyone stays on it. Any partial progress the enemy made before leaving decays away. The only way
to lose a zone is for the enemy to sail in and complete a capture of their own. This means:

- taking a zone frees your ships to move on instead of parking on it;
- every zone is always in one clear state: `Neutral`, `Capturing`, `Captured` or `Contested`.

### Scoring and victory

| Source | Points |
|---|---|
| Each held zone | `3.6 ÷ zoneCount` per second |
| Each kill | 12 |

Zone income is divided by the number of zones on the map, so holding every zone wins in the same
~278 seconds whether the map has one flag or five.

**You win by:**

- reaching **1000 points**, or
- sinking the entire enemy fleet, or
- leading on points when the **20:00** clock runs out.

Domination runs for 20 minutes because ships move at real speeds — a Yamato makes 27 knots, and the
nearest cap is 7.5 km from the start line.

Terrain generation, draft and grounding, deployment geometry, weather, fog of war and the full
objective rules are documented in [ENVIRONMENT.md](ENVIRONMENT.md).

---

## Scenario editor

Click **SCENARIO EDITOR** on the battle-setup screen to open the authoring screen. Procedural matches
are fine for play, but a reinforcement-learning agent needs the *same* situation over and over (and
needs it again next week), so scenarios are built by hand and saved to disk.

| Tool | What it does |
|---|---|
| **Select / Move** | Click to pick a ship, zone or island. Drag its middle to move it; drag its rim to resize. Right-drag rotates a ship. `Delete` removes it. |
| **Place ship** | Pick a class and side, then click to place a hull. |
| **Place zone** | Click to place a capture zone and drag out its radius (500–2000 m). |
| **Place island** | Click to place an island and drag out its radius. The height field is rebuilt when you play. |

The right-hand panel sets weather, battlefield preset, enemy skill, time limit, map seed and fog of
war.

| Button | Action |
|---|---|
| **SAVE** | Writes the scenario as JSON to `<persistentDataPath>/Scenarios/` |
| **LOAD** | Cycles through saved scenarios |
| **PLAY SCENARIO** | Launches exactly what is on screen |

A zone can be given a **starting owner**, so a scenario can open with a flag already held — useful
for training a decap or a defence on its own instead of always starting from a neutral map.

Example scenario file:

```json
{
  "scenarioName": "RL Training Alpha",
  "seed": 4242,
  "preset": 1,
  "weather": 1,
  "timeLimit": 900.0,
  "aiDifficulty": 1,
  "fogOfWar": true,
  "ships":   [ { "cls": 2, "team": 0, "x": -200.0, "y": -500.0, "heading": 15.0 } ],
  "zones":   [ { "name": "A", "x": -400.0, "y": 120.0, "radius": 175.0, "owner": 1 } ],
  "islands": [ { "x": 60.0, "y": 40.0, "radius": 145.0, "isRock": false } ]
}
```

---

## Enemy AI

The AI plays under **the same fog of war you do** — it only knows what its team has actually
detected — but it reasons carefully about what it does know.

### Strategy: reading the match

`BattleAssessment` is rebuilt about twice a second per team. It reads the game mode, the score, the
clock and the points per second each side is earning, then projects who wins if nothing changes.
That projection sets the fleet's **posture**:

| Posture | When | Behaviour |
|---|---|---|
| `LandGrab` | Early, zones still neutral | Take the easy caps fast |
| `Press` | Losing on projection | Force fights and flip zones |
| `Hold` | Winning on projection | Guard the zones that win the match; trade only when safe |
| `CloseOut` | Winning, clock running out | Disengage and run out the clock |
| `Desperate` | Losing, clock running out | Everything onto one zone |

In **Skirmish** and **Fleet Battle** there are no points, so the AI ignores zones entirely, **masses
into one force** and hunts the weakest isolated enemy group.

### Fleet-level behaviour

- **Allocation is a draft, not a fixed split.** Ships are assigned one at a time to whichever zone
  needs help most, and a zone's need drops as it receives ships. A losing flank therefore keeps
  pulling reinforcements automatically, with no special-case code.
- **It actually captures zones.** The first ships sent to a zone are *cap sitters*: they must stay
  inside the ring and circle within it rather than drifting off to shoot at something.
- **Focus fire without overkill.** Ships are committed to a target only until their combined
  firepower covers its remaining HP; the rest move on to the next target. Securing a kill overrides
  everything — a target that will die to the next salvo gets shot first.

### Ship-level tactics

These habits apply to **both fleets**, so your own uncommanded ships fight well too:

- **Armour angling** — bow-on while reloading, broadside when the salvo is ready.
- **Support discipline** — never push more than ~300 units past the nearest friendly heavy ship.
- **Terrain cover** — when disengaging, prefer a position that actually breaks line of sight (tested
  with the same function the detection system uses).
- **Regroup, don't retreat** — damaged ships fall back behind friends and repair instead of sailing
  home.
- **Local strength gating** — push when winning its part of the fight; open the range when not.
- **Inference** — a destroyer that disappears inside torpedo range makes ships weave instead of
  sailing predictably, and cruisers will radar a zone they believe a hidden destroyer is sitting on.

### Difficulty

Set with `GameBootstrap.enemyDifficulty`:

| Level | Behaviour |
|---|---|
| `Recruit` | Slow reactions, no inference, no use of cover |
| `Veteran` | In between |
| `Elite` (default) | Fastest reactions, full inference and cover |

### Watching it think

Press **`F1`** in game to see:

- lines to each ship's assigned station, coloured by assignment;
- which ships are committed to which cap;
- local strength bars;
- both commanders' current postures and projections.

---

## Architecture

All code lives in `Assets/Scripts/` under a single namespace, `Naval`.

- **Ships** are `MonoBehaviour` hosts with a `Rigidbody2D` and a hull collider.
- **Every gameplay subsystem** is a plain C# class ticked in a fixed order, so the simulation never
  depends on Unity's component execution order.
- **Per-frame work** (AI, navigation, gunnery, visuals) runs in `Update`; **forces** run in
  `FixedUpdate`.

```text
Assets/
├── Scripts/
│   ├── Core/        NavalTypes, ShipStats, ShipDatabase, GameEvents, ShipRegistry,
│   │                GameManager (fleet setup, domination, time compression), GameBootstrap
│   ├── World/       WorldMap (height field, islands, ports, zones), MapConfig (presets/layouts),
│   │                NavGrid (draft-aware A*), OceanRenderer, FogOfWarRenderer, WeatherSystem,
│   │                CaptureZone, NavalPort
│   ├── Ships/       Ship, ShipMovement (Rigidbody2D), ShipNavigation, ShipDamage, ShipDetection,
│   │                ShipWeapons, ShipAbilities, SubmarineSystem, ShipResources, ShipVisual
│   ├── Detection/   DetectionSystem (contact memory), SmokeScreen
│   ├── Combat/      ProjectileSystem (shells, torpedoes, homing, depth charges)
│   ├── AI/          BattleAssessment (shared situational picture), FleetCommander (strategy),
│   │                ShipAI (per-ship state machine, tactics, consumables)
│   ├── Player/      ControlModeManager, DirectShipController, RTSCamera, SelectionManager,
│   │                CommandSystem, FormationManager
│   ├── Scenario/    Scenario (data + JSON), ScenarioEditor
│   ├── UI/          UIManager (fleet menu, HUD, action bar), Minimap, WorldOverlay,
│   │                DebugOverlay (F1–F3), DevOverlay (F6)
│   ├── FX/          ParticleFX (batched CPU particles), LineDrawer (batched world lines)
│   ├── Audio/       AudioManager (procedurally synthesised clips)
│   ├── Util/        NavalMath, SpriteFactory, InputHub
│   └── RL/          RLEnvironment (trainer link, lockstep), RLObservation, RLActions,
│                    RLRewardTracker, RLMetrics, RLWire, RLLayout, RLPolicy, ...
└── Shaders/         NavalOcean, NavalFog, NavalParticle
```

### Physics

Ships are dynamic `Rigidbody2D` bodies (`gravityScale 0`, mass ≈ length × beam, continuous
collision, `CapsuleCollider2D` hull). `ShipMovement` drives them with three forces:

| Force | What it does |
|---|---|
| **Thrust** | A velocity servo along the bow, clamped to the class's acceleration/deceleration, so ships build speed gradually and coast when the engines are cut. |
| **Keel grip** | A sideways force that cancels lateral slip, so a hull follows its bow and drifts through hard turns instead of sliding like an air-hockey puck. |
| **Rudder torque** | Scaled by `dt`, so turning feels identical at 1x and 8x. Turn rate scales with speed — a stopped ship cannot steer. |

Other details:

- **Near-zero hull damping.** Real drag would cap a battleship below its rated speed, because its
  acceleration budget is so small.
- **Ship-vs-ship collisions** are handled by Physics2D and cause ramming damage and flooding.
- **Land is not a collider.** It is the height field, and grounding is calculated analytically from
  the depth gradient.
- **Time compression** scales `Time.timeScale` and widens `Time.fixedDeltaTime` (capped at the 4x
  value), so 8x does not mean 400 physics ticks per second.

### How the systems feed each other

- **Detection → AI → gunnery.** `DetectionSystem` runs at 8 Hz, accounting for signature, weather,
  smoke, land line-of-sight and submarine depth. It keeps each team's contacts as *confirmed*,
  *sonar/unknown* or *last known position*. The AI can only shoot at what its team can currently see.
- **Damage → capability → behaviour.** Hits are resolved against armour and impact angle, and damage
  lands on one of seven systems depending on where the shell struck. Engine damage cuts speed,
  steering damage cuts turn rate, sensor damage widens dispersion. Fires and flooding deal damage over
  time and make the ship easier to spot.
- **Consumables → AI.** The player's action bar and the AI use the same `ShipAbilities` code, so enemy
  destroyers really do smoke up under fire, cruisers radar a nearby destroyer, battleships repair once
  their fires are out, and submarines ping before firing.
- **Fleet commander → squadrons.** See [Enemy AI](#enemy-ai).

### Design decisions

- **Gun shells are simulated analytically, not with colliders.** Shells arc *over* the water, so a
  collider-based shell would wrongly hit ships it should fly past. Instead, each shell has a real time
  of flight to an aim point scattered by dispersion, and resolves when it lands — which is also why
  leading a target matters. Torpedoes and capture zones *do* use `Collider2D` triggers, since they are
  in the water.
- **Health bars use batched line drawing, not a world-space Canvas per ship.** The result on screen is
  the same (a constant-size bar above every spotted hull), but up to 60 ships plus contacts would
  otherwise mean dozens of canvases rebuilding every frame.
- **Pathfinding uses an A\* grid, not NavMesh.** The water is a height field with per-draft
  passability, which a baked NavMesh cannot express — a destroyer and a battleship need different
  navigable areas over the same water.

---

## Tuning

| What | Where |
|---|---|
| Ship balance (one block per class) | `Core/ShipDatabase.cs` |
| Consumable cooldowns, durations and charges | `Ships/ShipAbilities.cs` |
| Objective pacing | `GameManager.ZonePointsPerSecond`, `KillPoints`, `ScoreToWin`, `TimeLimit` |
| World scale | `GameConfig.WorldSize` — 4000 units, 1 unit ≈ 10 m |
| Draft vs. water depth | `WorldMap.DraftToDepth` — a destroyer can get about 17 units from a beach, a battleship needs roughly 50 |

---

## Reinforcement learning

Every ship has a `Controller`:

| Controller | Driven by |
|---|---|
| `Human` | The player's orders |
| `RuleAI` | `ShipAI` under a `FleetCommander` |
| `Learned` | A trained policy |

A learned ship decides where to go, how fast, what to shoot, whether to hold fire, when to launch
torpedoes and which consumable to use. The existing autopilot, gun lead and turret training then
carry out those decisions.

`Training/` contains the MAPPO trainer: a recurrent actor, a critic that sees the true game state, a
curriculum and a self-play league.

- **[Training/README.md](Training/README.md)** — how to build the headless player, train, evaluate and
  watch a policy, plus results so far.
- **[RL_README.md](RL_README.md)** — how the whole system works, file by file, with diagrams.

---

## Type-checking without the editor

`dotnet` can type-check the whole game against Unity's assemblies without opening Unity — useful in
CI or when the editor is closed:

```bash
dotnet build naval-check.csproj -v q --nologo
```

The project file compiles `Assets/Scripts/**/*.cs` against `Editor/Data/Managed/UnityEngine/*.dll`
and the package assemblies in `Library/ScriptAssemblies`.

---

## Further documentation

| Document | Covers |
|---|---|
| [SHIPS.md](SHIPS.md) | How each real ship's statistics were converted into game values |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Terrain, draft and grounding, deployment, weather, fog of war, objective rules |
| [RL_README.md](RL_README.md) | The reinforcement-learning system in depth |
| [Training/README.md](Training/README.md) | Building the training player, training, evaluation and results |
