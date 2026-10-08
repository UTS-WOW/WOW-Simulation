"""The reward: what each ship is paid for, every decision.

This is the one file to edit when you want the fleet to behave differently. Every term is listed in
REWARD_WEIGHTS with its weight; set a weight to 0 to switch a term off.

Two kinds of terms:

* Ship terms are earned by one ship for what it did itself (sailing into the circle, the damage its
  own guns did). They tell the policy which ship's action was good.
* Team terms are shared by every ship of the fleet (a capture, an enemy sunk, the win). In MAPPO the
  whole team optimises one shared objective, so everybody gets them, including ships that have
  already sunk: their critic still learns what the team went on to achieve.

The numbers Unity reports are described in Assets/Scripts/RL/RLRewardTracker.cs.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

# The circle terms are deliberately the largest steady signal. With in_circle 0.02, approach 0.2 and
# kills worth +-2, the fleet fought but never found the circle in 250 battles; with these weights it
# was sailing into it within about 150 (see simple_mappo/README.md, Results).
REWARD_WEIGHTS = OrderedDict([
    # ---- the objective: get to the centre of the capture circle and stay there
    ("in_circle", 0.05),          # ship: per second spent inside a capture circle
    ("approach_circle", 1.0),     # ship: per km sailed towards the nearest circle's centre (minus when sailing away)
    ("zone_captured", 1.0),       # team: our fleet captured a circle
    ("zone_lost", -1.0),          # team: the enemy took a circle from us
    # ---- fighting
    ("enemy_sunk", 1.0),          # team: per enemy ship sunk
    ("ship_lost", -1.0),          # team: per ship of ours sunk
    ("damage_dealt", 0.5),        # ship: per whole enemy hull of damage this ship's weapons did
    ("damage_taken", -0.25),      # ship: per whole hull of damage this ship took
    ("friendly_fire", -1.0),      # ship: per whole hull of damage done to our own side
    # ---- the result of the battle (last step only; there are no draws)
    ("win", 5.0),                 # team
    ("loss", -5.0),               # team
])

# Distances in the zone tokens are divided by 2800 game units, and one game unit is 10 m.
ZONE_DIST_TO_KM = 2800 * 10 / 1000


def ship_facts(zones: np.ndarray, zone_mask: np.ndarray, alive: np.ndarray, zone_features: list[str]) -> dict:
    """Where each ship is relative to the capture circles, read from its own observation.

    zones [N, Z, F] and zone_mask [N, Z] are one fleet's "zones" tokens; alive [N].
    Returns in_circle [N] (1 = inside a circle) and dist_km [N] (to the nearest circle's centre).
    """
    i_dist = zone_features.index("dist")
    i_inside = zone_features.index("i_am_inside")
    valid = zone_mask > 0.5
    dist = np.where(valid, zones[..., i_dist] * ZONE_DIST_TO_KM, np.inf).min(axis=1)
    dist = np.where(np.isfinite(dist) & (alive > 0.5), dist, 0.0)
    inside = (zones[..., i_inside] * valid).max(axis=1) * (alive > 0.5)
    return {"in_circle": inside.astype(np.float32), "dist_km": dist.astype(np.float32), "alive": alive.astype(np.float32)}


def compute_rewards(before: dict, after: dict, team: dict, ship: dict, n_own: int, n_enemy: int,
                    weights: dict = REWARD_WEIGHTS, seconds: float = 1.0) -> tuple[np.ndarray, dict]:
    """One decision's reward for every ship slot of the learning fleet.

    before / after: ship_facts() for the observation the ships acted on and the one that followed.
    team: Unity's team reward components for this step, by name (kills and losses are fractions
          of the fleet, so they are multiplied back into ship counts here).
    ship: Unity's per-ship reward components, by name, each [N].
    seconds: game time one decision covers (in_circle is paid per second).
    Returns (reward [N], {term: reward [N]}) - the breakdown is logged so you can see which terms
    the fleet is actually earning.
    """
    N = before["alive"].shape[0]
    was_alive = before["alive"] * after["alive"]
    # no ship sails 1 km in one decision: a jump that big is a new battle, not progress
    closer = before["dist_km"] - after["dist_km"]
    approach = np.where(np.abs(closer) <= 1.0, closer, 0.0) * was_alive

    team_value = {
        "zone_captured": team["zones_captured"],
        "zone_lost": team["zones_lost"],
        "enemy_sunk": team["kills"] * n_enemy,
        "ship_lost": team["losses"] * n_own,
        "win": team["win"],
        "loss": team["loss"],
    }
    ship_value = {
        "in_circle": after["in_circle"] * seconds,
        "approach_circle": approach,
        "damage_dealt": ship["damage_dealt"],
        "damage_taken": ship["damage_taken"],
        "friendly_fire": ship["friendly_fire_dealt"],
    }

    terms = {}
    for name, w in weights.items():
        if name in team_value:
            terms[name] = np.full(N, w * float(team_value[name]), dtype=np.float32)
        elif name in ship_value:
            terms[name] = (w * ship_value[name]).astype(np.float32)
        else:
            raise KeyError(f"unknown reward term '{name}'")
    total = np.sum(list(terms.values()), axis=0).astype(np.float32)
    return total, terms
