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
to circle k, a ship ordered to engage gets no circle pay, a "free" ship is paid for the nearest one.

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
    # ---- combat
    ("enemy_sunk", 1.0),          # team: per enemy ship sunk
    ("ship_lost", -1.0),          # team: per ship of ours sunk
    ("damage_dealt", 0.5),        # ship: per whole enemy hull of damage this ship's weapons did
    ("damage_taken", -0.25),      # ship: per whole hull of damage this ship took
    ("friendly_fire", -1.0),      # ship: per whole hull of damage done to our own side
    ("zero_sum", 1.0),            # team: minus the enemy ships' average damage terms (0 switches it off)
    # ---- outcome: the result of the battle (last step only; there are no draws)
    ("win", 5.0),                 # team
    ("loss", -5.0),               # team
])

REWARD_GROUPS = ["objective", "combat", "outcome"]
TERM_GROUP = {"in_circle": 0, "approach_circle": 0, "zone_captured": 0, "zone_lost": 0, "survive": 0,
              "enemy_sunk": 1, "ship_lost": 1, "damage_dealt": 1, "damage_taken": 1, "friendly_fire": 1,
              "zero_sum": 1, "win": 2, "loss": 2}
SHIP_TERMS = ("in_circle", "approach_circle", "survive", "damage_dealt", "damage_taken", "friendly_fire")

# Distances in the zone tokens are divided by 2800 game units, and one game unit is 10 m.
ZONE_DIST_TO_KM = 2800 * 10 / 1000

ORDER_FREE, ORDER_ENGAGE, ORDER_CIRCLE = 0, 1, 2       # as in networks.py


def ship_facts(zones: np.ndarray, zone_mask: np.ndarray, alive: np.ndarray, zone_features: list[str]) -> dict:
    """Where each ship is relative to every capture circle, read from its own observation.

    zones [N, Z, F] and zone_mask [N, Z] are one fleet's "zones" tokens; alive [N].
    Returns dist_km [N, Z] (inf where the circle does not exist), inside [N, Z] and alive [N].
    """
    i_dist = zone_features.index("dist")
    i_inside = zone_features.index("i_am_inside")
    valid = zone_mask > 0.5
    dist = np.where(valid, zones[..., i_dist] * ZONE_DIST_TO_KM, np.inf)
    inside = zones[..., i_inside] * valid
    return {"dist_km": dist.astype(np.float32), "inside": inside.astype(np.float32), "alive": alive.astype(np.float32)}


def no_facts(alive: np.ndarray, n_zones: int = 1) -> dict:
    """ship_facts for an environment without circle features (the mock)."""
    N = alive.shape[0]
    return {"dist_km": np.full((N, n_zones), np.inf, np.float32), "inside": np.zeros((N, n_zones), np.float32),
            "alive": alive.astype(np.float32)}


def circle_duty(facts: dict, orders: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(distance to the ship's circle in km [N], inside it [N]) given its orders.
    free -> the nearest circle; hold circle k -> circle k; engage -> no circle (distance 0, not inside)."""
    N, Z = facts["dist_km"].shape
    nearest = facts["dist_km"].argmin(axis=1)
    if orders is None:
        orders = np.zeros(N, dtype=np.int64)
    circle = np.where(orders >= ORDER_CIRCLE, np.minimum(orders - ORDER_CIRCLE, Z - 1), nearest)
    rows = np.arange(N)
    dist = facts["dist_km"][rows, circle]
    inside = facts["inside"][rows, circle]
    duty = (orders != ORDER_ENGAGE) & np.isfinite(dist) & (facts["alive"] > 0.5)
    return np.where(duty, dist, 0.0), np.where(duty, inside, 0.0)


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
    dist_before, _ = circle_duty(before, orders)
    dist_after, inside = circle_duty(after, orders)
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
