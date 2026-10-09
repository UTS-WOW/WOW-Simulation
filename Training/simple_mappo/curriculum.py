"""The curriculum: from shooting at a target that does not move up to random fleet battles.

Learning the full game from scratch is too hard, so training starts with single skills and moves
to the next stage once the fleet wins often enough - like a student moving up a year. Every stage
asks for a 70 % win rate before moving on.

    Stage              Battle                         Fleet        Opponent                     Commander  tau
    0 Gunnery          gunnery (open sea)             1 BB vs CA   passive target               -          -
                         passed: the target sunk within 150 s
    1 Capture          capture (one circle)           1 CA         passive, far away            -          -
                         passed: the circle captured
    2 Defend           defend (a neutral circle)      1 CA vs CA   Recruit                      -          -
    3 Duel             bb_duel (open sea)             1v1 BB       Recruit 70 % / itself 30 %   -          -
    4 King of the Hill koth_3v3 (one circle)          3v3 mixed    Veteran 70 % / itself 30 %   on         0.3
    5 Archipelago      archipelago_3v3 (3 circles)    3v3 mixed    Veteran 70 % / itself 30 %   on         0.5
    6 Domination       random map, random weather     6v6          Elite 40 / itself 40 / past  on         0.8
    7 Open             random mode, map, weather      4-8 a side   Elite 40 / itself 40 / past  on         1.0

  "itself" is a mirror battle: the current fleet flies BOTH sides and both sides' experience trains
  it - twice the data per battle, and an opponent that is always exactly as good as the fleet.
  Promotion only ever counts the battles against the rule AI.
                                                                   stop: 70 % of 250 held-out battles vs Elite

A stage is a dict:

    name               shown in the logs (and in the game's TRAINING STAGES screen, with
    battle, description  a one-line summary of the battle and of what the stage teaches)
    scenario           a battle file from Training/scenarios (or a list: one is picked per battle)
      or
    procedural         a newly generated battle every time, any value "random" or a list to pick from:
                         mode           0 Domination, 1 Skirmish, 2 Fleet Battle, 3 Capture and Control
                         preset         0 archipelago, 1 open sea, 2 strait
                         density        islands: 0 low, 1 medium, 2 high
                         weather        0 clear, 1 fog, 2 rain, 3 storm
                         ships          [fewest, most] ships per side
                         time_limit     seconds (0: the mode's own)
                         capture_radius [smallest, largest] in game units (10 m each)
                         world_reuse    battles fought on one generated map before a new one
    difficulty         the rule-based AI: 0 Recruit, 1 Veteran, 2 Elite
    opponent           who flies the enemy fleet:
                         "rule"     the game's rule-based AI at `difficulty`
                         "passive"  a target: Python orders it to stop and hold fire
                         "league"   a mix, given by `opponents` (OpenAI Five / AlphaStar self-play):
                                      rule    the rule AI at `difficulty` - the anchor that keeps the
                                              fleet honest, and the only battles that count for promotion
                                      mirror  our own current fleet on BOTH sides, and both sides learn
                                              (OpenAI Five's self-play): twice the data per battle
                                      latest  a frozen copy of our own fleet, refreshed every few updates
                                      past    an older copy from the league's pool, picked by
                                              prioritised fictitious self-play (the ones we lose to most)
    opponents          for "league": shares, e.g. {"rule": 0.4, "mirror": 0.4, "past": 0.2}
    commander          True: the fleet commander gives orders in this stage (False: every ship is free)
    team_spirit        tau in rewards.py: how much each ship's own reward is shared with the fleet
    rehearsal          share of battles replayed from earlier stages, so old skills are not forgotten
                       (they are logged under their own stage and never count for promotion)
    pass_if            what counts as passing a battle: "won" (default), "sunk" (every enemy ship sunk) or
                       "captured" (a circle captured); within: ... in at most this many seconds of battle
    promote_win_rate   move on once the share of passed battles over the last `window` qualifying
    window               battles (against the stage's own opponent: rule AI or passive target) reaches this
    stop_win_rate      final stage only: stop training once the win rate over the last
    stop_window          `stop_window` held-out evaluation battles reaches this. Held-out battles
                         are played on maps whose seeds training never uses.
    eval_battles       held-out battles played each time the stopping rule is checked
    eval_every         timesteps between those checks
    reward_weights     optional overrides of rewards.py's weights for this stage only, e.g.
                         {"approach_circle": 0.0} - in the duel the circle lies between the two
                         battleships, so racing to it means sailing bow-on into the enemy's guns

Add as many stages as you like. The last stage has no promotion, so training stays on it until its
stopping rule is met or total_timesteps runs out - drop stop_win_rate to keep it going for good.

One network - one method, one checkpoint - plays every stage: the observation always has room for 8
ships a side and 5 circles (env.py, MAX_SHIPS / MAX_ZONES), and smaller battles fill the empty slots
with zeros and a mask. Promotion just changes the battles; training carries on with the same model.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from .callbacks import BaseCallback

NO_CIRCLE = {"approach_circle": 0.0, "in_circle": 0.0}
LEAGUE = {"rule": 0.4, "mirror": 0.4, "past": 0.2}       # stages 6-7: the rule AI, itself (both sides learn), past selves
SELF_PLAY = {"rule": 0.7, "mirror": 0.3}                  # stages 3-5: mostly the rule AI, some battles against itself

DEFAULT_CURRICULUM = [
    # ---- single skills, one ship
    # Measured with random play / a simple script (32 battles each): gunnery - target sunk within 150 s
    # 52 % / 76 % (close in, fire at will); capture - circle captured 0 % / 100 % (sail to it); defend -
    # won 12 % / 0 % (sail to the circle). The passive target never scores, so a plain "won" would be
    # passed by random play (the tiebreak goes our way): these stages are judged on their own skill.
    {"name": "0 Gunnery", "battle": "1 battleship vs a passive cruiser, 7 km, open sea",
     "description": "Destroy a ship: pick the target, keep the guns firing, close in, use torpedoes.",
     "scenario": "scenarios/stage0_gunnery.json", "opponent": "passive",
     "pass_if": "sunk", "within": 150, "promote_win_rate": 0.70, "window": 50, "reward_weights": NO_CIRCLE},
    {"name": "1 Capture", "battle": "1 cruiser, one neutral circle, the enemy far away",
     "description": "Sail into the capture circle and stay inside until it is captured.",
     "scenario": "scenarios/stage1_capture.json", "opponent": "passive",
     "pass_if": "captured", "promote_win_rate": 0.70, "window": 50},
    {"name": "2 Defend", "battle": "1 cruiser vs 1 cruiser over a neutral circle",
     "description": "Take the circle and hold it while staying afloat: angle, dodge, smoke, repair, fight back.",
     "scenario": "scenarios/stage2_defend.json", "opponent": "rule", "difficulty": 0,
     "pass_if": "won", "promote_win_rate": 0.70, "window": 100, "reward_weights": {"survive": 0.02}},
    # The duel is about gunnery and angling. Its circle lies between the two battleships, so the circle
    # reward pulls the ship bow-on into the enemy's guns: measured, sailing to the circle wins 33% of
    # duels, broadside and fire at will 70%, and MAPPO learned the duel only with the circle terms off.
    {"name": "3 Duel", "battle": "1v1 battleships, 13 km, open sea",
     "description": "Win a battleship duel: choose and time fire, angle the hull, do not throw the ship away.",
     "scenario": "scenarios/stage0_bb_duel.json", "opponent": "league", "opponents": SELF_PLAY, "difficulty": 0,
     "promote_win_rate": 0.70, "window": 100, "rehearsal": 0.1, "reward_weights": NO_CIRCLE},
    # ---- fleets: the commander joins in, and the ships learn to share the reward
    {"name": "4 King of the Hill", "battle": "3v3 mixed fleets, one circle, open sea",
     "description": "The first fleet battle: the commander orders ships to hold the circle or engage.",
     "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "league", "opponents": SELF_PLAY, "difficulty": 1,
     "commander": True, "team_spirit": 0.3, "rehearsal": 0.15, "promote_win_rate": 0.70, "window": 100},
    {"name": "5 Archipelago", "battle": "3v3 mixed fleets, three circles among islands",
     "description": "Which ship holds which circle, and who hunts - with islands, spotting and smoke.",
     "scenario": "scenarios/stage2_archipelago_3v3.json", "opponent": "league", "opponents": SELF_PLAY, "difficulty": 1,
     "commander": True, "team_spirit": 0.5, "rehearsal": 0.15, "promote_win_rate": 0.70, "window": 150},
    # ---- the full game against the Elite AI and a league of our own past selves
    {"name": "6 Domination", "battle": "6v6 on a new random map and weather",
     "description": "The full Domination game against the Elite AI and a league of the fleet's own past selves.",
     "procedural": {"mode": 0, "preset": "random", "density": "random",
                                            "weather": "random", "ships": [6, 6], "time_limit": 900,
                                            "world_reuse": 8},
     "opponent": "league", "opponents": LEAGUE, "difficulty": 2, "commander": True, "team_spirit": 0.8,
     "rehearsal": 0.1, "promote_win_rate": 0.70, "window": 200},
    {"name": "7 Open", "battle": "4-8 ships a side, random mode, map, weather and circle size",
     "description": "Anything the game can produce - judged on maps training never saw.",
     "procedural": {"mode": "random", "preset": "random", "density": "random",
                                      "weather": "random", "ships": [4, 8], "time_limit": 1200,
                                      "capture_radius": [130, 200], "world_reuse": 4},
     "opponent": "league", "opponents": LEAGUE, "difficulty": 2, "commander": True, "team_spirit": 1.0,
     "rehearsal": 0.1, "stop_win_rate": 0.70, "stop_window": 250, "eval_battles": 50, "eval_every": 100_000},
]

RANDOM_CHOICES = {"mode": [0, 1, 2, 3], "preset": [0, 1, 2], "density": [0, 1, 2], "weather": [0, 1, 2, 3]}
MAX_PROCEDURAL_ZONES = 5            # Capture and Control has five circles
HELD_OUT_EVERY = 10                 # map seeds divisible by 10 are kept for held-out evaluation
QUALIFYING = ("rule", "passive")    # battles that count for promotion: never self-play, never evaluations


def load_curriculum(spec) -> list[dict]:
    """"default", a JSON file of stages, or a list of stage dicts."""
    if spec is None or spec == "default":
        return [dict(s) for s in DEFAULT_CURRICULUM]
    if isinstance(spec, str):
        from .env import TRAINING_DIR
        path = spec if os.path.isabs(spec) else os.path.join(TRAINING_DIR, spec)
        with open(path) as f:
            spec = json.load(f)
    return [dict(s) for s in spec]


def stage_sizes(stage: dict, load_scenario) -> tuple[int, int, int]:
    """(most own ships, most enemy ships, most capture circles) a stage can produce."""
    if "procedural" in stage:
        p = stage["procedural"]
        ships = p.get("ships", [6, 6])
        n = int(max(ships)) if isinstance(ships, (list, tuple)) else int(ships)
        modes = RANDOM_CHOICES["mode"] if p.get("mode", 0) == "random" else np.atleast_1d(p.get("mode", 0))
        zones = MAX_PROCEDURAL_ZONES if 3 in list(modes) else 3
        return n, n, zones
    paths = stage["scenario"] if isinstance(stage["scenario"], list) else [stage["scenario"]]
    own = enemy = zones = 0
    for path in paths:
        sc = load_scenario(path)
        own = max(own, sum(1 for s in sc["ships"] if s["team"] == 0))
        enemy = max(enemy, sum(1 for s in sc["ships"] if s["team"] == 1))
        zones = max(zones, len(sc.get("zones", [])))
    return own, enemy, max(1, zones)


def map_seed(rng: np.random.Generator, held_out: bool) -> int:
    """Training maps and held-out maps never share a seed."""
    if held_out:
        return HELD_OUT_EVERY * int(rng.integers(1, 99_999))
    while True:
        seed = int(rng.integers(1, 999_999))
        if seed % HELD_OUT_EVERY:
            return seed


def procedural_reset(p: dict, rng: np.random.Generator, world: tuple[int, int],
                     held_out: bool = False) -> tuple[dict, tuple[int, int]]:
    """The reset message for a generated battle, and the (map seed, battles fought on it) to keep.
    Held-out battles always get a fresh held-out map."""
    def pick(key, default):
        v = p.get(key, default)
        if v == "random":
            return int(rng.choice(RANDOM_CHOICES[key]))
        if isinstance(v, (list, tuple)):
            return int(rng.choice(v))
        return int(v)

    seed, used = world
    if held_out or used >= int(p.get("world_reuse", 1)) or seed % HELD_OUT_EVERY == 0:
        seed, used = map_seed(rng, held_out), 0
    ships = p.get("ships", [6, 6])
    lo, hi = (ships[0], ships[-1]) if isinstance(ships, (list, tuple)) else (ships, ships)
    n = int(rng.integers(int(lo), int(hi) + 1))
    msg = {"use_scenario": False, "mode": pick("mode", 0), "preset": pick("preset", 0), "density": pick("density", 1),
           "weather": pick("weather", 0), "player_ships": n, "enemy_ships": n, "seed": seed,
           "time_limit": float(p.get("time_limit", 0))}
    if "capture_radius" in p:
        r = p["capture_radius"]
        msg["capture_radius"] = float(rng.uniform(r[0], r[1])) if isinstance(r, (list, tuple)) else float(r)
    return msg, (seed, used + 1)


class CurriculumCallback(BaseCallback):
    """Promotes the environment to the next stage, and applies the final stage's stopping rule.

    Promotion counts only battles that started on the current stage against the stage's own
    opponent (the rule-based AI, or the passive target of the first stages; never self-play), and
    the window starts empty on every new stage, so a promotion is always earned on the stage itself.

    The windows carry over between sessions: at the start of training they are rebuilt from the
    run's monitor.csv, so a session that stops after 60 of the 100 battles a stage needs is continued
    by the next session (yours or a teammate's) instead of starting the count again.

    stop_on_promotion=True ends training right after a promotion (the per-stage notebooks use it: the
    saved model then carries the new stage, and the next stage's notebook continues from it).
    """

    def __init__(self, verbose: int = 1, stop_on_promotion: bool = False):
        super().__init__(verbose)
        self.stop_on_promotion = stop_on_promotion
        self.results: list[float] = []          # current stage, rule-AI battles
        self.held_out: list[float] = []         # final stage, held-out evaluation battles
        self.last_eval = 0
        self.history: list[dict] = []           # promotions and evaluations, for the notebook
        self.promoted = False
        self.finished = False                   # the final stage's stopping rule is met

    def _on_training_start(self) -> None:
        self.last_eval = self.num_timesteps
        self.promoted = self.finished = False
        self.results, self.held_out = self._from_log()
        env = self.model.env
        stage = env.stages[env.stage]
        if self.results and self.verbose:
            window = int(stage.get("window", 100))
            print(f"continuing the promotion window of '{stage['name']}': {len(self.results)} of {window} battles so far, "
                  f"won {np.mean(self.results):.0%} (needs {stage.get('promote_win_rate', 0):.0%})")
        if self.held_out and self.verbose:
            need = int(stage.get("stop_window", 250))
            print(f"continuing the held-out window: {len(self.held_out)} of {need} battles, won {np.mean(self.held_out):.0%}"
                  + (" - the stopping rule is already met; training on tries to do better"
                     if len(self.held_out) >= need and np.mean(self.held_out) >= stage.get("stop_win_rate", 1.0) else ""))

    def _from_log(self) -> tuple[list[float], list[float]]:
        """The current stage's rule-AI battles and held-out battles, from the run's monitor.csv."""
        env = self.model.env
        path = getattr(env, "path", None)                  # the Monitor wrapper's file
        if not path or not os.path.exists(path):
            return [], []
        stage = env.stages[env.stage]
        with open(path) as f:
            next(f)                                        # the '#{...}' header line
            rows = [r for r in csv.DictReader(f) if int(float(r.get("stage") or 0)) == env.stage]
        passed = "passed" if rows and rows[0].get("passed") not in (None, "") else "won"
        results = [float(r[passed]) for r in rows if r.get("opponent", "rule") in QUALIFYING]
        held_out = [float(r[passed]) for r in rows if r.get("opponent") == "eval-heldout"]
        return results[-int(stage.get("window", 100)):], held_out[-int(stage.get("stop_window", 250)):]

    def _on_step(self) -> bool:
        env = self.model.env
        stage = env.stages[env.stage]
        for info in self.locals.get("infos", []):
            if info and info.get("stage") == env.stage and info.get("opponent") in QUALIFYING and not info.get("evaluation"):
                self.results.append(float(info.get("passed", info["won"])))
        window = int(stage.get("window", 100))
        self.results = self.results[-window:]
        threshold = stage.get("promote_win_rate")
        last = env.stage == len(env.stages) - 1
        if not last and threshold is not None and len(self.results) >= window and np.mean(self.results) >= threshold:
            msg = (f"stage '{stage['name']}' passed: passed {np.mean(self.results):.0%} of the last {window} "
                   f"qualifying battles - moving on to '{env.stages[env.stage + 1]['name']}'")
            self.history.append({"timesteps": self.num_timesteps, "event": msg})
            if self.verbose:
                print(f"*** {msg} (at {self.num_timesteps} timesteps) ***")
            env.set_stage(env.stage + 1)
            self.results = []
            self.promoted = True
            if self.stop_on_promotion:
                return False
        return True

    def _on_rollout_end(self) -> bool:
        """Between updates: on the final stage, play held-out battles and check the stopping rule."""
        env = self.model.env
        stage = env.stages[env.stage]
        if env.stage != len(env.stages) - 1 or stage.get("stop_win_rate") is None:
            return True
        if self.num_timesteps - self.last_eval < int(stage.get("eval_every", 100_000)):
            return True
        from .utils import evaluate
        self.last_eval = self.num_timesteps
        r = evaluate(self.model, env, n_battles=int(stage.get("eval_battles", 50)), held_out=True, keep_results=True)
        self.held_out += [float(x.get("passed", x["won"])) for x in r["results"]]
        need = int(stage.get("stop_window", 250))
        self.held_out = self.held_out[-need:]
        rate = float(np.mean(self.held_out))
        msg = (f"held-out evaluation: won {r['win_rate']:.0%} of {r['battles']} battles; "
               f"{rate:.0%} over the last {len(self.held_out)}/{need}")
        self.history.append({"timesteps": self.num_timesteps, "event": msg})
        if self.verbose:
            print(f"*** {msg} ***")
        self.model.reset_battles()                  # the evaluation interrupted the training battles
        if len(self.held_out) >= need and rate >= stage["stop_win_rate"]:
            if self.verbose:
                print(f"*** stopping rule met: {rate:.0%} >= {stage['stop_win_rate']:.0%} over {need} held-out battles ***")
            self.finished = True
            return False
        return True
