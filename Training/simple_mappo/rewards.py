"""The reward: what each ship is paid for, every decision.

This is the one file to edit when you want the fleet to behave differently. Every term is listed in
REWARD_WEIGHTS with its weight; set a weight to 0 to switch a term off.

Two kinds of terms:

* Ship terms are earned by one ship for what it did itself (sailing into the circle, the damage its
  own guns did). They tell the policy which ship's action was good.
* Team terms are shared by every ship of the fleet (a capture, an enemy sunk, the win). The whole
  team optimises one shared objective, so everybody gets them, including ships that have already
  sunk: their critic still learns what the team went on to achieve.

Three ideas from OpenAI Five (Dota 2) and Honor of Kings:

* Team spirit (tau, set per curriculum stage). Each ship's own terms are blended with its fleet's
  average:  own' = (1 - tau) * own + tau * fleet average.  tau = 0 early (learn your own job), tau
  -> 1 later (care about the fleet as much as about yourself). OpenAI Five anneals it 0.3 -> 1.
* Zero-sum ("zero_sum" weight). Whatever the enemy ships earn in combat is subtracted, so damaging
  them and not being damaged count the same. OpenAI Five subtracts the enemy team's reward too.
* Reward groups. The terms fall into three groups - objective, combat, outcome - and the critic
  predicts each group's return separately (Honor of Kings' multi-head value): three easier
  predictions instead of one hard one, and the logs show which group the fleet earns.

The circle terms follow the commander's order: a ship ordered to hold circle k is paid for sailing
to circle k, a ship ordered to engage gets no circle pay. A "free" ship is paid for the nearest circle
that still NEEDS a ship - one the fleet does not own yet, and not already taken by enough closer
allies - so a free fleet spreads over the circles instead of crowding the centre one. With every
circle owned, a free ship guards its nearest one. And "circles_held" pays the whole fleet, every
second, for holding more circles than the enemy (that is how Domination is scored).

The numbers Unity reports are described in Assets/Scripts/RL/RLRewardTracker.cs.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

# The circle terms are deliberately the largest steady signal. With in_circle 0.02, approach 0.2 and
# kills worth +-2, the fleet fought but never found the circle in 250 battles; with these weights it
# was sailing into it within about 150 (see simple_mappo/README.md, Results).
REWARD_WEIGHTS = OrderedDict([
    # ---- objective: get to the centre of the capture circle, stay there, stay afloat
    ("in_circle", 0.05),          # ship: per second spent inside its circle (see the commander's orders above)
    ("approach_circle", 1.0),     # ship: per km sailed towards its circle's centre (minus when sailing away)
    ("zone_captured", 1.0),       # team: our fleet captured a circle
    ("zone_lost", -1.0),          # team: the enemy took a circle from us
    ("survive", 0.0),             # ship: per second afloat (the "Defend" stage switches it on)
    ("circles_held", 0.05),       # team: per second, (our circles - the enemy's) / all circles
    # ---- combat
    ("enemy_sunk", 1.0),          # team: per enemy ship sunk
    ("ship_lost", -1.0),          # team: per ship of ours sunk
    ("damage_dealt", 0.5),        # ship: per whole enemy hull of damage this ship's weapons did
    ("damage_taken", -0.25),      # ship: per whole hull of damage this ship took
    ("friendly_fire", -1.0),      # ship: per whole hull of damage done to our own side
    ("aground", -0.05),           # ship: per second against land - aground, or stopped at the bank
    ("collision", -0.25),         # ship: per collision with another ship, ours or theirs (ramming damage
                                  #   still counts as damage too)
    ("zero_sum", 1.0),            # team: minus the enemy ships' average damage terms (0 switches it off)
    # ---- outcome: the result of the battle (last step only; there are no draws)
    ("win", 5.0),                 # team
    ("loss", -5.0),               # team
])

REWARD_GROUPS = ["objective", "combat", "outcome"]
TERM_GROUP = {"in_circle": 0, "approach_circle": 0, "zone_captured": 0, "zone_lost": 0, "survive": 0, "circles_held": 0,
              "enemy_sunk": 1, "ship_lost": 1, "damage_dealt": 1, "damage_taken": 1, "friendly_fire": 1,
              "aground": 1, "collision": 1,
              "zero_sum": 1, "win": 2, "loss": 2}
SHIP_TERMS = ("in_circle", "approach_circle", "survive", "damage_dealt", "damage_taken", "friendly_fire", "aground",
              "collision")

# Distances in the zone tokens are divided by 2800 game units, and one game unit is 10 m.
ZONE_DIST_TO_KM = 2800 * 10 / 1000

ORDER_FREE, ORDER_ENGAGE, ORDER_CIRCLE = 0, 1, 2       # as in networks.py


def ship_facts(zones: np.ndarray, zone_mask: np.ndarray, alive: np.ndarray, zone_features: list[str]) -> dict:
    """Where each ship is relative to every capture circle, read from its own observation.

    zones [N, Z, F] and zone_mask [N, Z] are one fleet's "zones" tokens; alive [N].
    Returns dist_km [N, Z] (inf where the circle does not exist), inside [N, Z], alive [N] and
    mine [Z] (1 = our fleet owns the circle; read from any ship still afloat).
    """
    i_dist = zone_features.index("dist")
    i_inside = zone_features.index("i_am_inside")
    valid = zone_mask > 0.5
    dist = np.where(valid, zones[..., i_dist] * ZONE_DIST_TO_KM, np.inf)
    inside = zones[..., i_inside] * valid
    afloat = (alive > 0.5)[:, None] & valid
    mine = (zones[..., zone_features.index("owner_mine")] * afloat).max(axis=0) if "owner_mine" in zone_features \
        else np.zeros(zones.shape[1])
    return {"dist_km": dist.astype(np.float32), "inside": inside.astype(np.float32), "alive": alive.astype(np.float32),
            "mine": (mine > 0.5).astype(np.float32)}


def no_facts(alive: np.ndarray, n_zones: int = 1) -> dict:
    """ship_facts for an environment without circle features (the mock)."""
    N = alive.shape[0]
    return {"dist_km": np.full((N, n_zones), np.inf, np.float32), "inside": np.zeros((N, n_zones), np.float32),
            "alive": alive.astype(np.float32), "mine": np.zeros(n_zones, np.float32)}


def circle_duty(facts: dict, orders: np.ndarray | None) -> np.ndarray:
    """The circle each ship is paid for [N] (-1: none), given the commander's orders:

        hold circle k  -> circle k
        engage         -> no circle
        free           -> the nearest circle that still needs a ship: one our fleet does not own yet,
                          with room left. The free ships are handed out nearest pair first, and each
                          needy circle takes at most its share (free ships / needy circles, rounded up;
                          ships ordered to it count too) - so 9 free ships and 5 neutral circles become
                          2-2-2-2-1 instead of 9 in the middle. With every circle ours: the nearest one.
    """
    dist = facts["dist_km"]
    N, Z = dist.shape
    orders = np.zeros(N, dtype=np.int64) if orders is None else np.asarray(orders, dtype=np.int64)
    afloat = facts["alive"] > 0.5
    exists = np.isfinite(dist).any(axis=0)                                      # circles of this battle
    circle = np.where(orders >= ORDER_CIRCLE, np.minimum(orders - ORDER_CIRCLE, Z - 1), -1)
    free = np.flatnonzero((orders == ORDER_FREE) & afloat & np.isfinite(dist).any(axis=1))
    if len(free) == 0 or not exists.any():
        return circle
    needy = exists & (facts.get("mine", np.zeros(Z)) < 0.5)
    if not needy.any():                                                          # all ours: guard the nearest
        circle[free] = dist[free].argmin(axis=1)
        return circle
    share = int(np.ceil(len(free) / needy.sum()))
    load = np.bincount(circle[(circle >= 0) & afloat], minlength=Z)[:Z]           # ships ordered to each circle
    pairs = sorted((dist[i, c], i, c) for i in free for c in np.flatnonzero(needy))
    for _, i, c in pairs:                                                        # nearest pair first
        if circle[i] < 0 and load[c] < share:
            circle[i] = c
            load[c] += 1
    left = free[circle[free] < 0]                                                # every needy circle is full
    if len(left):
        circle[left] = np.where(needy, dist[left], np.inf).argmin(axis=1)
    return circle


def compute_rewards(before: dict, after: dict, team: dict, ship: dict, enemy_ship: dict | None, n_own: int,
                    n_enemy: int, weights: dict = REWARD_WEIGHTS, seconds: float = 1.0,
                    orders: np.ndarray | None = None, team_spirit: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """One decision's reward for every ship slot of the learning fleet.

    before / after: ship_facts() for the observation the ships acted on and the one that followed.
    team:  Unity's team reward components for this step, by name (kills and losses are fractions
           of the fleet, so they are multiplied back into ship counts here).
    ship / enemy_ship: Unity's per-ship reward components of our fleet / the enemy fleet, each [N].
    orders: the commander's order for every ship [N] (None: everybody free).
    team_spirit: tau, how much of a ship's own terms is replaced by the fleet average.
    Returns (reward [N], reward per group [N, 3], {term: reward [N]}) - the breakdown is logged so you
    can see which terms the fleet is actually earning.
    """
    N = before["alive"].shape[0]
    exists = (np.arange(N) < n_own).astype(np.float32)
    was_alive = before["alive"] * after["alive"]
    # the circle each ship was sent to when it acted; progress is measured towards that same circle
    circle = circle_duty(before, orders)
    rows, k = np.arange(N), np.maximum(circle, 0)
    duty = (circle >= 0) & np.isfinite(before["dist_km"][rows, k]) & np.isfinite(after["dist_km"][rows, k])
    dist_before = np.where(duty, before["dist_km"][rows, k], 0.0)
    dist_after = np.where(duty, after["dist_km"][rows, k], 0.0)
    inside = np.where(duty, after["inside"][rows, k], 0.0) * after["alive"]
    # no ship sails 1 km in one decision: a jump that big is a new battle or a new order, not progress
    closer = dist_before - dist_after
    approach = np.where(np.abs(closer) <= 1.0, closer, 0.0) * was_alive

    enemy_combat = 0.0
    if enemy_ship is not None and n_enemy > 0:
        their = (weights.get("damage_dealt", 0.0) * enemy_ship["damage_dealt"]
                 + weights.get("damage_taken", 0.0) * enemy_ship["damage_taken"]
                 + weights.get("friendly_fire", 0.0) * enemy_ship["friendly_fire_dealt"])
        enemy_combat = float(np.sum(their[:n_enemy]) / n_enemy)

    team_value = {
        "zone_captured": team["zones_captured"],
        "zone_lost": team["zones_lost"],
        "circles_held": team.get("circles_held", 0.0) * seconds,   # (ours - theirs) / circles, from env.py
        "enemy_sunk": team["kills"] * n_enemy,
        "ship_lost": team["losses"] * n_own,
        "zero_sum": -enemy_combat,
        "win": team["win"],
        "loss": team["loss"],
    }
    ship_value = {
        "in_circle": inside * seconds,
        "approach_circle": approach,
        "survive": after["alive"] * seconds,
        "damage_dealt": ship["damage_dealt"],
        "damage_taken": ship["damage_taken"],
        "friendly_fire": ship["friendly_fire_dealt"],
        # reported by players built after these terms were added; an older player has neither
        "aground": ship.get("aground", np.zeros(N, np.float32)) * seconds,
        "collision": ship.get("collisions", np.zeros(N, np.float32)),
    }

    terms = {}
    for name, w in weights.items():
        if name in team_value:
            terms[name] = np.full(N, w * float(team_value[name]), dtype=np.float32)
        elif name in ship_value:
            own = (w * ship_value[name]).astype(np.float32) * exists
            if team_spirit > 0 and n_own > 1:
                own = (1 - team_spirit) * own + team_spirit * own.sum() / n_own
            terms[name] = own.astype(np.float32)
        else:
            raise KeyError(f"unknown reward term '{name}'")
    groups = np.zeros((N, len(REWARD_GROUPS)), dtype=np.float32)
    for name, r in terms.items():
        groups[:, TERM_GROUP[name]] += r
    groups *= exists[:, None]
    return groups.sum(axis=1), groups, terms
