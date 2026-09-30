# Naval Warfare — 2D Fleet Combat (Unity 6)

A top-down 2D naval combat game in the spirit of World of Warships: fleets of **1 to 30 ships a
side**, a **hybrid control scheme** that lets you either con a single warship yourself or command
the whole fleet as an RTS, **three battlefields** with configurable objectives, four combat ship
classes, and **time compression up to 8x**.

Everything is generated at runtime — no art, audio or prefab assets. Ship sprites, turrets, the
water/terrain shader, particle atlases and every sound effect are synthesised in code.

## Running it

Open `Assets/Scenes/SampleScene.unity` and press Play. The scene contains a single `GameBootstrap`
object that builds the entire game.

| Bootstrap field | Meaning |
|---|---|
| `startMode` | Domination (default), Skirmish, FleetBattle, CaptureAndControl, Escort |
| `seed` | 0 = new map every run, any other value = reproducible map |
| `startWeather` | Clear, Fog, Rain, Storm |
| `skipMenu` | Skip fleet selection and drop straight into deployment |
| `heightmapResolution` | Terrain sampling resolution (512 is a good default) |

**Flow:** fleet selection → deployment → battle → result.

1. **Battle setup.** Slide each side's fleet size (1–30, default 6), choose how the enemy is built
   (**Balanced** / **Custom slots** / **Mirror yours**), pick the battlefield, and choose whether you
   start as **fleet commander** (default) or as the **captain** of one ship.
2. **Deployment.** The sim is paused. The fleet deploys as **three squadrons — LEFT, CENTRE and
   RIGHT** — on your baseline, facing the enemy across the map, with caps A/B/C strung along the
   centre line between you. Drag ships to reposition them inside their squadron's area (drag one
   across to hand it to a neighbouring group), pick a formation, then **START BATTLE**.
3. **Battle.** You start in full RTS fleet command. `Tab` takes the helm of the selected ship and
   `Tab` again hands it back.

The three squadrons come **pre-bound to control groups 1, 2 and 3**, so `1`/`2`/`3` instantly
select your left, centre and right groups.

## Controls

**`Tab` — switch between DIRECT CONTROL and FLEET COMMAND.**

### Direct control (you are the captain)

| Input | Action |
|---|---|
| `W` / `S` | Engine order telegraph — throttle ahead / astern (momentum, not instant) |
| `A` / `D` | Rudder to port / starboard; releasing eases it amidships |
| `Space` | All stop |
| Mouse | Trains the guns; the reticle auto-leads a target it is resting on |
| Left click / hold | Fire the main battery |
| Right click | Torpedo spread along the reticle bearing (arcs are drawn on screen) |
| `1` – `6` | Class consumables (see below) |
| `X` | Submarine dive / surface |
| `E` | Damage control party |

### Fleet command (RTS)

Left click selects, drag box selects many, `Shift`+click adds, double click grabs the whole class.
**`1` / `2` / `3` select the left, centre and right squadrons**; `Ctrl`+`1..9` rebinds a group and
**`Ctrl`+`A` selects the whole fleet**. Right click moves or attacks; `Shift`+right click queues
waypoints. `C` attack-move, `V` patrol, `B` escort, `T` focus fire, `Space` stop, `H` hold,
`R` astern, `G` retreat, `Y` return to port, `Q` smoke, `F5`–`F9` formations, `F10` break.
Clicking a ship in the task force roster selects it (or takes its helm if you are in direct mode).

### Camera, time and system

`WASD`/edge scroll/middle-drag pan (fleet mode only — in direct control WASD is the helm), wheel
zooms out far enough to see the whole map, `F` follows, `` ` `` frames the fleet. **`+` / `-` cycle time compression 1x → 2x → 4x → 8x**,
`P` pauses, or use the buttons on the status bar. `F1` debug draw, `F2` reveal map, `F3` nav grid,
`F4` command reference, **`F6` dev view**.

**Dev view (`F6`)** enlarges hulls so they stay readable at strategic zoom, draws every turret's
firing arc and blind sector, and rings the selected ship with its concealment, spotting, gun and
torpedo ranges. Turrets that can bear on the current target are drawn in the team colour; those that
cannot are red.

## Ship classes and consumables

Every hull is a real Tier 10 ship, and every number below comes from its actual statistics —
see [SHIPS.md](SHIPS.md) for the full conversion.

| Class | Ship | HP | Speed | Concealment | Gun range | Consumables |
|---|---|---|---|---|---|---|
| **DD** | Shimakaze | 17 900 | 39 kn | **5.6 km** | 11.4 km | HE `1`, AP `2`, torpedoes `3`, smoke `4`, engine boost `5`, damage control `6` |
| **CA** | Des Moines | 50 600 | 33 kn | 10.9 km | 15.8 km | HE `1`, AP `2`, radar `3`, hydro `4`, repair `5`, damage control `6` |
| **BB** | Yamato | 97 200 | 27 kn | 14.1 km | **26.6 km** | HE `1`, AP `2`, damage control `3`, repair `4`, spotter plane `5` |
| **SS** | Balao | 20 200 | 30 kn | 5.9 km surfaced, 2.3 km at periscope | 4.0 km deck gun | HE `1`, homing torps `2`, ping `3`, hydrophone `4`, surveillance `5`, damage control `6`, dive `X` |

Guns out-range eyes by a wide margin — a Yamato shoots 26.6 km but is only *seen* at 14.1 km — so
**spotting decides the battle**. A destroyer that stays dark is what lets the battle line shoot at
all, and radar is what takes that away from it.

Every ship carries an **overhead class symbol** — DD two diamonds, CA diamond with a slash, BB
diamond with two slashes, SS chevron — held at a constant screen size and coloured
cyan for friendly, crimson for hostile, amber for neutral. It stays readable through smoke and
weather and at any zoom, which is how you read a 30-ship battle at a glance. The hull also carries an
elongated team aura tracing its waterline, so you can see which way a contact is pointing.

**HE vs AP** matters: HE trades penetration and raw damage for a much higher fire chance and can
never citadel; AP does full damage, can over-penetrate light hulls, and can land citadel hits on a
broadside target.

**Angling and overmatch.** A shell striking more than ~45° off the plate normal starts to bounce and
always bounces past 60°, so turning your bow toward the enemy is how you survive. Two things defeat
it. *Improved angles*: Des Moines super-heavy AP does not begin ricocheting until 60° and only
always bounces at 75°. *Overmatch*: a shell defeats plating thinner than calibre/14.3 **regardless of
angle** — Yamato's 460 mm rifles overmatch 32 mm, which is exactly a cruiser's bow, so angling does
not save a cruiser from a Yamato even though it saves it from everything else.

Measured over 400 shells per case:

| Shot | Result |
|---|---|
| Yamato AP into an angled Des Moines bow | 100% penetration — overmatched |
| Yamato AP into a broadside Des Moines | 34% citadel, 66% penetration |
| Yamato AP into a Shimakaze | 100% overpenetration — nothing to arm the fuse |
| Des Moines AP into an angled Yamato bow | 100% ricochet |
| Des Moines AP into a broadside Yamato | 100% penetration, **never a citadel** (450 mm cannot beat a 410 mm belt) |
| Des Moines AP into a broadside Des Moines | 37% citadel |

**Turret arcs.** Each mount sits at its own point along the hull and trains within its own arc, so
firepower depends on heading. A Yamato bow-on to its target brings only its two forward turrets to
bear — **6 of 9 barrels** — and gets all nine only once it opens to about 40°. Stern-on it has three.
Turrets track to their arc limit and wait there rather than centring, so they are already pressed
against the stop when the hull comes round.

| Yamato heading relative to target | Barrels bearing |
|---|---|
| 0–30° (bow-on) | 6 of 9 |
| 40–120° (broadside) | **9 of 9** |
| 150° | 6 of 9 |
| 180° (stern-on) | 3 of 9 |

This is the cost side of angling: the heading that bounces shells is also the heading that silences a
third of your guns.

**Submarine ping → homing torpedoes:** a sonar ping marks a target for ~25 seconds. Torpedoes fired
while the mark holds steer onto it; the marked ship also lights up on the plot until the lock decays.

**Radar and hydro defeat concealment outright.** Surveillance radar (10 km) spots everything inside
its circle through smoke *and* through islands; hydroacoustic search (5 km) and the submarine's
hydrophone (7 km) do the same but only through smoke, and both also hear submerged boats. This is the
counter to a destroyer sitting invisible on a cap.

## Battlefields and objectives

| Preset | Terrain | Default objective |
|---|---|---|
| **Ocean Archipelago** | Scattered islands giving cover and torpedo chokepoints | Three-point domination (A/B/C) |
| **Open Sea** | No cover at all — pure gunnery and angling | King of the Hill (one large central point) |
| **Strait Clash** | Two landmasses squeezing a narrow central channel | Two-flag assault (a flag in front of each base) |

Island density, weather and capture radius (500–2000 m) are all adjustable on the setup screen, and
each preset sets its own spawn-to-cap distance so first contact happens early rather than after a
five-minute sail.

Zones are `CircleCollider2D` triggers: ships inside fill the meter (about 40 seconds solo, faster
with more hulls, diminishing returns). **Both fleets inside freezes it — contested.**

**Capture is permanent.** Once the meter completes, the zone is yours and keeps scoring whether or
not anyone stays behind; partial progress the enemy made before breaking off decays away. The only
way to lose a point is for the other side to sail in and complete a capture of their own. That means
taking a cap frees your ships to move on instead of parking on it, and makes the objective a clean
discrete state — `Neutral → Capturing → Captured`, plus `Contested` — rather than a value that
quietly bleeds away.

Zone income is normalised by how many zones the map has — `3.6 / zoneCount` points per second each,
so holding the whole map wins in the same ~278 seconds whether that map has one flag or five. Each
kill is worth 12. **Win by** reaching **1000 points**, sinking the enemy fleet, or leading on points
when the **20:00** clock expires. Domination runs 20 minutes because the ships move at real speeds —
a Yamato makes 27 knots, and the nearest cap is 7.5 km from the start line.

The battlefield itself — terrain generation, draft and grounding, deployment geometry, weather, fog
of war and the full objective ruleset — is documented in **[ENVIRONMENT.md](ENVIRONMENT.md)**.

## Scenario editor

**SCENARIO EDITOR** on the battle-setup screen opens a separate authoring screen. The procedural
match is fine for play, but a reinforcement-learning agent needs the *same* situation over and over,
and needs it back next week, so scenarios are hand-built and saved to disk.

| Tool | What it does |
|---|---|
| **Select / Move** | Click to pick a ship, zone or island. Drag the middle to move it, drag the rim to resize. Right-drag turns a ship. `Delete` removes it. |
| **Place ship** | Pick a class and side, then click to drop a hull. |
| **Place zone** | Click to drop a capture circle and drag out its radius (500–2000 m). |
| **Place island** | Click to drop an island and drag out its radius; the height field rebuilds when you play. |

The right-hand panel sets weather, battlefield preset, enemy skill, time limit, map seed and fog of
war. **SAVE** writes the scenario as JSON to `<persistentDataPath>/Scenarios/`, **LOAD** cycles
through what is saved, and **PLAY SCENARIO** launches exactly what is on screen.

A zone can be given a **starting owner**, so a scenario can open with a flag already held — useful
for training a decap or a defence in isolation rather than always from a neutral board.

```json
{
  "scenarioName": "RL Training Alpha",
  "seed": 4242, "preset": 1, "weather": 1,
  "timeLimit": 900.0, "aiDifficulty": 1, "fogOfWar": true,
  "ships":  [ { "cls": 2, "team": 0, "x": -200.0, "y": -500.0, "heading": 15.0 } ],
  "zones":  [ { "name": "A", "x": -400.0, "y": 120.0, "radius": 175.0, "owner": 1 } ],
  "islands":[ { "x": 60.0, "y": 40.0, "radius": 145.0, "isRock": false } ]
}
```

## Architecture

`Assets/Scripts/` — one namespace (`Naval`). Ships are `MonoBehaviour` hosts carrying a
`Rigidbody2D` and hull collider; every gameplay subsystem is a plain C# class ticked in a fixed
order, so the simulation never depends on Unity's component execution order. Frame work (AI,
navigation, gunnery, visuals) runs in `Update`; forces run in `FixedUpdate`.

```
Core/      NavalTypes, ShipStats, ShipDatabase, GameEvents, ShipRegistry,
           GameManager (fleet setup, domination, time compression), GameBootstrap
World/     WorldMap (height field, islands, ports, zones), MapConfig (presets/layouts),
           NavGrid (draft-aware A*), OceanRenderer, FogOfWarRenderer, WeatherSystem,
           CaptureZone, NavalPort
Ships/     Ship, ShipMovement (Rigidbody2D), ShipNavigation, ShipDamage, ShipDetection,
           ShipWeapons, ShipAbilities, SubmarineSystem, ShipResources, ShipVisual
Detection/ DetectionSystem (contact memory), SmokeScreen
Combat/    ProjectileSystem (shells, torpedoes, homing, depth charges)
AI/        BattleAssessment (shared situational picture), FleetCommander (strategy),
           ShipAI (per-ship FSM, tactics, consumables)
Player/    ControlModeManager, DirectShipController, RTSCamera, SelectionManager,
           CommandSystem, FormationManager
UI/        UIManager (fleet menu, HUD, action bar), Minimap, WorldOverlay, DebugOverlay
FX/        ParticleFX (batched CPU particles), LineDrawer (batched world lines)
Audio/     AudioManager (procedurally synthesised clips)
Util/      NavalMath, SpriteFactory, InputHub
RL/        RLEnvironment (trainer link, lockstep), RLObservation (entity tokens, fog-of-war legal
           actor view + privileged critic view), RLActions (intent actions and masks),
           RLRewardTracker, RLMetrics, RLWire, RLLayout
Shaders/   NavalOcean, NavalFog, NavalParticle
```

### Physics

Ships are dynamic `Rigidbody2D` bodies (`gravityScale 0`, mass ≈ length × beam, continuous
collision, a `CapsuleCollider2D` hull). `ShipMovement` drives them with three forces:

* **Thrust** — a velocity servo along the bow clamped to the class's acceleration/deceleration, so
  ships build way and coast when the engines are cut.
* **Keel grip** — a lateral force that kills sideways slip, which is what makes a hull track its bow
  and drift through hard turns instead of sliding like an air-hockey puck.
* **Rudder torque** — derived from `dt`, so the turn response is identical at 1x and 8x. Turn rate
  scales with speed: a stopped ship cannot steer at all.

Hull damping is deliberately near zero — real drag there would cap a battleship below its rated
speed, because its acceleration budget is tiny. Ship-vs-ship collisions are resolved by Physics2D
and produce ramming damage and flooding; **land is not a collider**, it is the height field, and
grounding is resolved analytically against the depth gradient.

Time compression scales `Time.timeScale` and widens `Time.fixedDeltaTime` (clamped at 4x worth) so
8x does not turn into 400 solver ticks a second.

### How the systems feed each other

* **Detection → AI → gunnery.** `DetectionSystem` runs at 8 Hz over signature, weather, smoke, land
  line-of-sight and submarine depth, keeping per-team contacts as *confirmed*, *sonar/unknown* or
  *last known position*. The AI can only shoot at what its team can currently see.
* **Damage → capability → behaviour.** Hits resolve against armour and angle; damage lands on one of
  seven systems chosen by where the shell struck. Engine damage cuts speed, steering damage cuts turn
  rate, sensor damage widens dispersion. Fires and flooding tick damage and raise your signature.
* **Consumables → AI.** The same `ShipAbilities` code path serves the player's action bar and the AI,
  so enemy destroyers really do smoke up under fire, cruisers radar a knife-fighting destroyer,
  battleships heal once their fires are out, and submarines ping before shooting.
* **Fleet commander → squadrons.** See the AI section below.

## The enemy AI

The AI plays by **the same fog of war you do** — it only knows what its team has actually detected —
but it reasons hard about what it does know.

**It knows what game it is playing.** `BattleAssessment` (rebuilt about twice a second per team) reads
the mode, the score, the clock, and the points/second each side is earning, and projects who wins if
nothing changes. That projection drives a **posture**:

| Posture | When | Behaviour |
|---|---|---|
| `LandGrab` | early, points still neutral | take the cheap caps fast |
| `Press` | losing on projection | force fights and flip points |
| `Hold` | winning on projection | garrison what wins the match, trade only when safe |
| `CloseOut` | winning, clock running out | disengage and stall it out |
| `Desperate` | losing, clock running out | everything onto one point |

In Skirmish/Fleet Battle there are no points, so it drops the caps entirely, **masses into one force**
and goes after the weakest isolated enemy group.

**Allocation is a draft, not a fixed split.** Ships are dealt one at a time to whichever zone needs
help most, and a zone's need falls as it is fed — so a flank that is losing keeps pulling
reinforcements automatically, with no special-case rotation code.

**It actually captures points.** The first ships sent to a zone are *cap sitters*: they are required
to be inside the ring and orbit within it rather than drifting off to shoot at something.

**Fire concentration without overkill.** Shooters are committed to a target only until the assigned
firepower covers its remaining hit points; the rest move to the next target. Kill-securing overrides
everything — a target that dies to the next salvo gets shot first.

**Tactical habits** (both fleets, so your uncommanded ships fight well too):

- **Armour angling** — bow-on while reloading, broadside when the salvo is ready
- **Support discipline** — no pushing more than ~300 units past the nearest friendly heavy
- **Terrain cover** — when disengaging, prefers a spot that actually breaks line of sight, tested with
  the same function the detection system uses
- **Regroup, not retreat** — damaged ships fall back behind friends and heal instead of sailing home
- **Local strength gating** — pushes when winning its corner of the fight, opens the range when not
- **Inference** — a destroyer that goes dark inside torpedo range makes ships weave instead of
  steaming predictably, and cruisers will radar a point they believe a hidden destroyer is sitting on

**Difficulty** is set on `GameBootstrap.enemyDifficulty`: `Recruit` (slow reactions, no inference or
cover), `Veteran`, `Elite` (default — fastest reactions, full inference and cover).

Press **F1** in game to see it think: assignment-coloured lines to each ship's station, cap
commitments, local strength bars, and a readout of both commanders' postures and projections.

### Tuning

Balance lives in `Core/ShipDatabase.cs`, one block per class. Consumable cooldowns, durations and
charges are in `Ships/ShipAbilities.cs`. Objective pacing is `GameManager.ZonePointsPerSecond`,
`KillPoints`, `ScoreToWin` and `TimeLimit`. World scale is `GameConfig.WorldSize` (4000 units,
1 unit ≈ 10 m); `WorldMap.DraftToDepth` maps draft onto required depth — a destroyer clears a beach
about 17 units out, a battleship needs roughly 50.

## Deviations from the brief, and why

* **Gun shells are analytic, not colliders.** Shells arc *over* the water, so a collider-based shell
  would wrongly hit ships it should fly past. They fly with a real time of flight to a dispersion
  scattered aim point and resolve on landing — which is also what makes leading a target matter.
  Torpedoes and capture zones do use `Collider2D` triggers, since they run in the water.
* **Health bars are batched line-drawing, not a world-space Canvas per ship.** Same result on screen
  — a bar above every spotted hull, at constant screen size — but 36 ships plus contacts would
  otherwise mean dozens of extra canvases rebuilding every frame.
* **A\* grid pathfinding** rather than NavMesh: the water is a height field with per-draft
  passability, which a baked NavMesh cannot express (a destroyer and a battleship need different
  navigable areas over the same water).

## Reinforcement learning

Ships have a `Controller`: `Human` (the player's orders), `RuleAI` (ShipAI under a FleetCommander) or
`Learned` (a trained policy). A learned ship picks where to go, how fast, what to shoot, whether to
hold fire, when to launch torpedoes and which consumable to use, and the existing autopilot, gun lead
and turret training carry that out. `Training/` holds the MAPPO trainer (recurrent actor, critic
that sees the true state, curriculum, self-play league). [Training/README.md](Training/README.md)
<<<<<<< HEAD
covers building the headless player and running it, and [RL_README.md](RL_README.md) explains how
the whole system works, file by file, with diagrams.
=======
covers building the headless player and running it.
>>>>>>> 54a8e3c5ec5ccc60ed3cb77b484c2f2e01797910

## Verifying without the editor

`dotnet` can type-check the whole game against Unity's assemblies without opening Unity, which is
useful in CI or when the editor is closed:

```bash
dotnet build naval-check.csproj -v q --nologo
```

The project file references `Editor/Data/Managed/UnityEngine/*.dll` plus the package assemblies in
`Library/ScriptAssemblies` and compiles `Assets/Scripts/**/*.cs`.
