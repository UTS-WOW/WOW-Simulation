# Ship Data

Every hull in the game is a real Tier 10 warship. This document records the source statistics, the
conversion into simulation units, and the handful of places where the source data did not cover
something the simulation needs.

The data lives in [`ShipDatabase.cs`](Assets/Scripts/Core/ShipDatabase.cs).

## Contents

1. [Conversion](#1-conversion)
2. [The four hulls](#2-the-four-hulls) — hull, shells, torpedoes, submarine specifics, consumables
3. [What the source data did not cover](#3-what-the-source-data-did-not-cover)
4. [Consequences for the simulation](#4-consequences-for-the-simulation)

---

## 1. Conversion

The battlefield runs at **1 world unit = 10 m** (see [ENVIRONMENT.md](ENVIRONMENT.md)).

| Source quantity | Conversion | Helper |
|---|---|---|
| Kilometres (ranges, concealment) | `× 100` | `Km(x)` |
| Metres (hull length, dispersion, smoke radius) | `× 0.1` | `Metres(x)` |
| Knots (speed) | `÷ 19.4` | `Kn(x)` |
| Metres/second (muzzle velocity) | `× 0.1` | `Mps(x)` |
| Millimetres (armour, penetration) | direct | — |
| Seconds (reload, cooldown, action time) | direct | — |

Two derived quantities are not stored directly in the source data:

**Turn rate** comes from the turning circle radius at full speed, `ω = v / r`:

```text
TurnRateFor(knots, circleRadiusMetres) = Kn(knots) / Metres(radius) × Rad2Deg
```

**Rudder shift** is stored as a time in seconds and becomes a rate of rudder travel, `1 / seconds`.

These two produce genuinely sluggish handling — a Yamato turns at 0.89°/s and takes 22 seconds just to
get the rudder over — because that is what the real figures say. Committing to a turn is a decision
you live with for a minute.

---

## 2. The four hulls

| | Shimakaze (DD) | Des Moines (CA) | Yamato (BB) | Balao (SS) |
|---|---|---|---|---|
| Nation | Japan | USA | Japan | USA |
| Hull | 129.5 × 11.2 m | 218 × 23 m | 263 × 38.9 m | 95 × 8.3 m |
| **HP** | 17 900 | 50 600 | 97 200 | 20 200 |
| Bow / hull plating | 19 mm | 27 mm | 32 mm | 19 mm |
| Citadel belt | none | 152 mm | 410 mm | none |
| Torpedo protection | 0% | 7% | **55%** | 0% |
| **Speed** | 39 kn (2.01 u/s) | 33 kn (1.70) | 27 kn (1.39) | 30 kn (1.55) |
| Turning circle | 690 m | 770 m | 900 m | 590 m |
| Turn rate | 1.67°/s | 1.27°/s | **0.89°/s** | 1.50°/s |
| Rudder shift | 3.0 s | 8.6 s | **22.1 s** | 5.8 s |
| **Concealment** | **5.6 km** | 10.9 km | 14.1 km | 5.9 / 2.3 km |
| Gun range | 11.4 km | 15.8 km | **26.6 km** | 4.0 km (deck gun) |
| Battery | 3 × 2 × 127 mm | 3 × 3 × 203 mm | 3 × 3 × 460 mm | 1 × 127 mm |
| Reload | 5.7 s | 5.5 s | 30 s | 4.5 s |
| Turret traverse | 7.9°/s | 30°/s | **3.0°/s** | 30°/s |
| Dispersion at max range | 104 m | 143 m | 275 m | 90 m |
| Sigma | 2.0 | 2.05 | 2.1 | — |
| Fleet point cost | 2 | 3 | 5 | 3 |

### Shells

| | HE damage | HE pen | Fire chance | AP damage | AP pen | Overmatch | Ricochet (starts / always) |
|---|---|---|---|---|---|---|---|
| Shimakaze | 2 150 | 21 mm | 9% | — (HE only) | — | 8.9 mm | 45° / 60° |
| Des Moines | 2 800 | 34 mm | 14% | 5 000 | **450 mm** | 14.2 mm | **60° / 75°** |
| Yamato | 7 300 | 76 mm | 36% | **14 800** | **850 mm** | **32 mm** | 45° / 60° |

Overmatch is `calibre / 14.3`: plating at or below that thickness is defeated regardless of impact
angle. Yamato's 32 mm is the number that matters — it is exactly a cruiser's bow, which is why
angling does not protect a cruiser from a Yamato.

Des Moines carries super-heavy AP with **improved ricochet angles**: it does not start bouncing until
60° off the plate normal and only always bounces at 75°, against 45°/60° for everything else.

### Torpedoes

| | Shimakaze (Type 93 mod 3) | Balao (acoustic homing) |
|---|---|---|
| Tubes | 3 × 5 | 2 × 5 |
| Damage | **23 767** | 7 833 |
| Reload | 153 s | 48 s |
| Range | 12.0 km | 14.0 km |
| Speed | 67 kn | 89 kn |
| Detection range | 1.7 km | 2.1 km |
| Homing | no | yes, on a sonar-ping mark |

### Submarine specifics

Dive capacity is the real resource, not the hull: **240 seconds** submerged, draining at 1/s and
recharging at 1/s. Speed is unchanged at periscope depth and drops to 18 knots at maximum depth.
Concealment falls from 5.9 km surfaced to 2.3 km at periscope depth, and to nothing at depth — a deep
boat is found only by hydrophone or submarine surveillance.

### Consumables

| Ship | Consumable | Charges | Cooldown | Action | Effect |
|---|---|---|---|---|---|
| DD | Smoke generator | 3 | 160 s | 20 s emit, 97 s life | 450 m radius |
| DD | Engine boost | 3 | 120 s | 120 s | ×1.08 speed |
| DD / CA / BB / SS | Damage control | ∞ / ∞ / ∞ / 3 | 40 / 60 / 80 / 40 s | — | Clears fire and flooding |
| CA | Surveillance radar | 3 | 120 s | 40 s | **10 km, through islands and smoke** |
| CA / SS | Hydroacoustic search / hydrophone | 3 / 4 | 120 / 60 s | 100 / 30 s | 5 km / 7 km, through smoke, hears submerged |
| CA / BB | Repair party | 3 / 4 | 80 s | 28 s | 0.5% max HP per second |
| BB | Spotter plane | 4 | 240 s | 100 s | ×1.2 main battery range |
| SS | Submarine surveillance | 3 | 120 s | 60 s | 9 km, finds boats at depth |

Radar, hydro, hydrophone and submarine surveillance grant **assured detection**: a contact inside the
radius is spotted regardless of its concealment. Radar alone reaches through land.

---

## 3. What the source data did not cover

These are the places where a figure had to be supplied rather than converted. All are flagged in the
code comments too.

| Gap | What was done |
|---|---|
| **Yamato HE** | The source lists only AP. Filled in with the historical Yamato HE round: 7 300 damage, 76 mm penetration, 36% fire chance. |
| **Balao artillery** | The source replaces artillery with the sonar pinger. A surfaced boat with no weapon at all cannot defend itself while the tubes reload, so the historical 127 mm deck gun is carried, surface-only. |
| **Balao's second torpedo** | The alternative heavy torpedo (14 900 damage, no homing) is not carried — the simulation supports one torpedo profile per hull, and the acoustic homing fish is the boat's primary weapon. |
| **Bow / stern tubes** | Modelled as two groups of five rather than a bow six and a stern four; the simulation has no separate fore and aft launcher arcs. |
| **Destroyer sonar** | No ASW sensor is listed, which would make submarines unkillable. A modest 2 km depth-charge search set is fitted. |
| **Assured acquisition** | Only the submarine lists it. A 2 km guaranteed acquisition range is applied to every surface ship. |
| **Anti-aircraft** | AA ratings and flak bursts are recorded but inert — there are no aircraft in the game. |
| **Air detection** | Not modelled, for the same reason. |
| **Fire and flood rates** | Not in the source. Set as a percentage of max HP per second, preserving the previous balance: roughly 0.16–0.32%/s for fire and 0.22–1.1%/s for flooding. |
| **Acceleration, fuel, reverse speed** | Not in the source. Carried over proportionally from the previous values. |

---

## 4. Consequences for the simulation

Adopting real statistics changed three things beyond the ships themselves.

**Detection range cutoff.** The observer/target loop used to stop at 1200 units. Yamato shoots 2660,
so a battleship would have been unable to see anything it could shell. The cutoff is now
`DetectionSystem.MaxDetectionRange = 2800` (28 km).

**AI standoff.** The tactical layer holds a fraction of gun range, which for a Yamato is 25 km — far
beyond the 14.1 km at which it can see anything. `ShipAI.KeepRange` now clamps the desired range to
what the ship can spot for itself, unless the contact is already being held by a friendly ship, in
which case it takes the free reach.

**Match pacing.** Real speeds are roughly a third of the previous arcade figures. Spawn distances
were cut so the nearest cap is 7.5 km from the start line, and Domination now runs **20 minutes**
rather than 15. The two battle lines start 15.0 km apart — deliberately just outside mutual
battleship spotting range, so the approach is still a phase of the battle rather than an immediate
gun duel.

The start lines, fleet separation and destroyer / battleship arrival times for each battlefield are
tabulated in [ENVIRONMENT.md § 4](ENVIRONMENT.md#4-deployment-geometry).
