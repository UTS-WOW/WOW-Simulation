"""The naval battle as a multi-agent environment.

How it fits together:

    Python (this file)  -- actions for every ship -->  Unity headless player (one battle per process)
                        <-- observations, raw reward      simulates decision_period seconds
                            numbers, "battle over?" --

NavalEnv runs n_envs battles side by side - the same idea as make_atari_env(n_envs=4) in the Atari
notebooks. Every ship of the Player fleet is an agent. The enemy fleet is flown by the game's own
rule-based AI, by Python as a passive target (stop, hold fire), or - in league battles - by a frozen
copy of our own fleet (see curriculum.py, "opponent").

What reset() and step() return (E = n_envs, N = ships per side, everything padded to MAX_SHIPS etc.):

  what each ship knows - the captain network reads only these:
    obs["self"]         [E, N, ...]         the ship itself (+ the score and the kind of battle)
    obs["allies"]       [E, N, A, ...]      its squadron mates            (+ obs["ally_mask"])
    obs["contacts"]     [E, N, C, ...]      enemies its TEAM has spotted, last known positions
                                                                          (+ obs["contact_mask"])
    obs["zones"]        [E, N, Z, ...]      the capture circles           (+ obs["zone_mask"])
    obs["obstacles"]    [E, N, O, ...]      the nearest islands and smoke (+ obs["obstacle_mask"])
    obs["action_mask"]  [E, N, sum(heads)]  1 = the option is legal right now

  the true battle - the critic's view, training only:
    obs["critic_own"]   [E, N, ...]         our ships  (+ obs["own_mask"])
    obs["critic_enemy"] [E, N, ...]         the enemy ships, true positions included (+ obs["critic_enemy_mask"])
    obs["critic_zones"] [E, Z, ...]         the circles (+ obs["critic_zone_mask"])
    obs["critic_match"] [E, ...]            score, time, the kind of battle

  the fleet commander's view - the same, with everything the team cannot know zeroed:
    obs["fleet_enemy"]  [E, N, ...]         critic_enemy without the true-state columns (belief only)
    obs["fleet_match"]  [E, ...]            critic_match without the enemy's true strength

  bookkeeping:
    obs["alive"]        [E, N]              the ship is afloat
    obs["exists"]       [E, N]              the slot holds a ship in this battle
    obs["starts"]       [E]                 1 on the first observation of a battle (clear the memory)
    obs["t"]            [E]                 decisions since the battle started
    obs["commander_on"] [E]                 this battle's stage uses the fleet commander

    rewards             [E, N]              see rewards.py (env.last_reward_groups: the same per group)
    dones               [E]                 the battle ended on this step. Like Stable-Baselines3's
                                            vector envs, the next battle has already been started:
                                            obs is its first observation and infos[e] holds the result.

Every ship picks one option from each action head:

    move     keep course, 8 compass legs, close in, broadside, open range, regroup, take cover,
             return to port, or "sail to circle k"
    speed    full, 2/3, 1/3, stop, astern
    target   auto, or focus enemy contact k
    fire     fire at will, hold fire
    torpedo  none, launch at the target
    ability  none, one of 12 consumables (shell type, smoke, radar, repair...), dive, surface

step(actions, orders) also takes the commander's orders [E, N] (rewards.py pays the circle terms for
the ordered circle). record_expert=True turns the environment into a recorder for imitation learning:
the rule AI flies both fleets and env.last_expert holds what it did with every ship of ours.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time

import numpy as np

from naval_rl.env import init_message, make_workers, port_free

from .curriculum import load_curriculum, procedural_reset, stage_sizes
from .rewards import REWARD_GROUPS, REWARD_WEIGHTS, compute_rewards, no_facts, ship_facts

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)

# The observation always has room for this many ships a side, capture circles and nearby obstacles,
# whatever the battle. Smaller battles leave slots empty (zeros, with a mask), so ONE network - and
# one checkpoint - trains through every curriculum stage, including stages added later.
MAX_SHIPS = 8
MAX_ZONES = 5          # Capture and Control has five circles, the most the game makes
MAX_OBSTACLES = 4

# what the captain network reads (each with its mask), and the critic's true picture of the battle
CAPTAIN_ARRAYS = ["self", "allies", "ally_mask", "contacts", "contact_mask", "zones", "zone_mask",
                  "obstacles", "obstacle_mask"]
CRITIC_ARRAYS = ["critic_own", "critic_enemy", "critic_enemy_mask", "critic_zones", "critic_zone_mask", "critic_match"]
OBS_KEYS = CAPTAIN_ARRAYS + CRITIC_ARRAYS + ["action_mask", "alive", "exists", "own_mask", "fleet_enemy",
                                            "fleet_match", "starts", "t", "commander_on"]
# columns of critic_match that are not public knowledge: the commander does not get them
PRIVILEGED_MATCH = ("their_fleet_strength", "their_alive_frac_true")
# a passive target: stop (speed option 3) and hold fire (fire option 1); every other head option 0
PASSIVE = {"speed": 3, "fire": 1}


def find_unity_binary(path: str | None = None) -> str:
    """The headless training player: an explicit path, $NAVAL_UNITY_BINARY, a SageMaker "unity"
    input channel, the repo's Builds/ folder, or a NavalTrainer/ folder next to this package."""
    candidates = [
        path,
        os.environ.get("NAVAL_UNITY_BINARY"),
        os.path.join(os.environ["SM_CHANNEL_UNITY"], "NavalTrainer.x86_64") if "SM_CHANNEL_UNITY" in os.environ else None,
        os.path.join(TRAINING_DIR, "..", "Builds", "NavalTrainer", "NavalTrainer.x86_64"),
        os.path.join(TRAINING_DIR, "NavalTrainer", "NavalTrainer.x86_64"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            # copies through zip files and S3 lose the execute permission
            mode = os.stat(c).st_mode
            if not mode & stat.S_IXUSR:
                os.chmod(c, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            check_glibc()
            return _runnable(os.path.abspath(c))
    raise FileNotFoundError("no headless training player found - build it with Training/build_player.sh, "
                            "or pass unity_binary=... / set NAVAL_UNITY_BINARY")


def _runnable(binary: str) -> str:
    """Some machines mount the home folder 'noexec': programs stored there cannot run (the player
    would die loading UnityPlayer.so). Then the player folder is copied to the temp folder, once."""
    folder = os.path.dirname(binary)
    if not os.statvfs(folder).f_flag & getattr(os, "ST_NOEXEC", 8):
        return binary
    key = hashlib.md5(json.dumps(sorted((n, os.path.getsize(os.path.join(folder, n))) for n in os.listdir(folder)
                                        if os.path.isfile(os.path.join(folder, n)))).encode()).hexdigest()[:10]
    target = os.path.join(tempfile.gettempdir(), f"naval_trainer_{key}")
    if not os.path.isdir(target):
        print(f"{folder} does not allow running programs - copying the player to {target}")
        shutil.copytree(folder, target + ".tmp", ignore=shutil.ignore_patterns("*DoNotShip*", "runs"))
        os.rename(target + ".tmp", target)
    copied = os.path.join(target, os.path.basename(binary))
    os.chmod(copied, os.stat(copied).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return copied


def _free_ports(start: int, n: int, tries: int = 100) -> int:
    """The first block of n free ports from `start` on. Players left over from an earlier session
    (a restarted notebook kernel does not stop them) would otherwise block the default ports."""
    port = start
    for _ in range(tries):
        if all(port_free(port + i) for i in range(n)):
            return port
        port += n
    raise RuntimeError(f"no {n} free ports between {start} and {port} - stop old players with: pkill -f NavalTrainer.x86_64")


def _log_tail(path: str | None, lines: int = 25) -> str:
    if not path or not os.path.exists(path):
        return f"    ({path or 'no log'} was not written)"
    with open(path, errors="replace") as f:
        tail = f.read().splitlines()[-lines:]
    return "\n".join("    " + line for line in tail) if tail else f"    ({path} is empty)"


# The Unity 6 player links against glibc 2.35 (Ubuntu 22.04 and newer)
MIN_GLIBC = (2, 35)


def check_glibc() -> None:
    """Fails early, with the fix, where the player could not start (e.g. an Ubuntu 20.04 container)."""
    try:
        name, version = os.confstr("CS_GNU_LIBC_VERSION").split()
        found = tuple(int(x) for x in version.split(".")[:2])
    except (ValueError, OSError, AttributeError):
        return                                        # not glibc (or unknown): let the player try
    if name == "glibc" and found < MIN_GLIBC:
        raise RuntimeError(f"this system has glibc {version}, the Unity player needs {MIN_GLIBC[0]}.{MIN_GLIBC[1]}+ "
                           f"(Ubuntu 22.04 or newer). On SageMaker use a PyTorch training image 2.4 or newer "
                           f"(framework_version='2.5', py_version='py311').")


def load_scenario(path: str) -> dict:
    full = path if os.path.isabs(path) else os.path.join(TRAINING_DIR, path)
    with open(full) as f:
        return json.load(f)


class NavalEnv:
    """n_envs naval battles run in parallel; the learning fleet is the Player side of each."""

    def __init__(self, scenario: str | list[str] = "scenarios/stage1_koth_3v3.json", n_envs: int = 4,
                 opponent_difficulty: int = 0, stages=None, start_stage: int = 0,
                 decision_period: float = 1.0, sim_dt: float = 0.08, time_limit: float | None = None,
                 jitter: float = 40.0, reward_weights: dict | None = None,
                 unity_binary: str | None = None, ports: list[int] | None = None, base_port: int = 5005,
                 seed: int = 0, mock: bool = False, unity_log_dir: str | None = None, graphics: bool = False,
                 record_expert: bool = False, expert_difficulty: int = 2):
        """
        scenario             a battle from Training/scenarios (or a list to pick one from per battle)
        opponent_difficulty  the rule AI's skill: 0 Recruit, 1 Veteran, 2 Elite
        stages               a curriculum instead of one scenario: "default" (stages 0-7), a JSON
                             file, or a list of stage dicts - see curriculum.py. The observation has
                             the same size in every stage (MAX_SHIPS, MAX_ZONES), so one network and
                             one checkpoint train through all of them.
        start_stage          the stage to begin on
        decision_period      game seconds between decisions; the ships keep carrying out their
                             orders in between (like frame skipping in the Atari notebooks)
        sim_dt               game seconds per simulated frame. 0.02 is the game's normal 50 fps step;
                             0.08 is the step the game itself uses at x4-x8 time compression and
                             simulates about 3x faster.
        time_limit           override every battle's length (seconds)
        jitter               random offset of every ship's start position (game units, 10 m each)
        ports                connect to environments that are already running (e.g. the Unity editor
                             in Play mode with GameBootstrap.rlTrainingServer ticked) instead of
                             launching headless players
        mock                 a fake environment with the same shapes, to test the code without Unity
        graphics             launch players that can render real game frames (record_unity_video).
                             Needs a computer with a screen; training itself never needs it
        record_expert        imitation-learning recorder: the rule AI at expert_difficulty flies both
                             fleets, and after every step env.last_expert holds its choices for our ships
        """
        self.rng = np.random.default_rng(seed)
        if stages is None:
            stages = [{"name": os.path.splitext(os.path.basename(scenario if isinstance(scenario, str) else scenario[0]))[0],
                       "scenario": scenario, "difficulty": opponent_difficulty}]
        self.stages = load_curriculum(stages)
        self._scenario_cache = {}
        self.time_limit = time_limit
        self.jitter = jitter
        self.decision_period = decision_period
        self.reward_weights = dict(REWARD_WEIGHTS if reward_weights is None else reward_weights)
        self.record_expert, self.expert_difficulty = record_expert, expert_difficulty

        # one fixed observation size for every stage (see MAX_SHIPS), so models carry over between stages
        for st in self.stages:
            own, enemy, zones = stage_sizes(st, self._scenario)
            if max(own, enemy) > MAX_SHIPS or zones > MAX_ZONES:
                raise ValueError(f"stage '{st.get('name')}' has {max(own, enemy)} ships a side and {zones} circles; "
                                 f"the network has room for {MAX_SHIPS} and {MAX_ZONES} (MAX_SHIPS / MAX_ZONES in env.py - "
                                 f"raising them changes the network, so existing checkpoints would not load)")
        init = init_message(max_team=MAX_SHIPS, max_allies=MAX_SHIPS - 1, max_contacts=MAX_SHIPS, max_zones=MAX_ZONES,
                            action_mode="intent", decision_period=decision_period, sim_dt=sim_dt, reflexes=True,
                            max_obstacles=MAX_OBSTACLES)
        binary = None if (mock or ports) else find_unity_binary(unity_binary)
        if binary:
            free = _free_ports(base_port, n_envs)
            if free != base_port:
                print(f"ports {base_port}-{base_port + n_envs - 1} are in use (players from an earlier run?) - "
                      f"using {free}-{free + n_envs - 1}. To stop old players: pkill -f NavalTrainer.x86_64")
                base_port = free
        try:
            self.workers = make_workers(len(ports) if ports else n_envs, init, binary, base_port, ports,
                                        log_dir=unity_log_dir, mock=mock,
                                        mock_kwargs={"episode_length": 60} if mock else None, graphics=graphics)
        except ConnectionError as e:
            if not unity_log_dir:
                raise
            # show why the player stopped, instead of only where its log is
            port = base_port
            raise RuntimeError(
                f"{e}\n\nThe player's own output (unity_{port}.out):\n"
                f"{_log_tail(os.path.join(unity_log_dir, f'unity_{port}.out'))}\n\n"
                f"The end of Unity's log (unity_{port}.log):\n{_log_tail(os.path.join(unity_log_dir, f'unity_{port}.log'))}"
            ) from e
        self._closed = False
        atexit.register(self.close)          # a kernel restart or the end of a script stops the players
        spec = self.workers[0].spec
        self.spec = spec
        self.n_envs = len(self.workers)
        self.n_agents = spec.max_team
        self.heads = [dict(h.__dict__) for h in spec.heads]       # name, fixed, pointer, size
        self.head_names = [h["name"] for h in self.heads]
        self.head_sizes = [h["size"] for h in self.heads]
        self.zone_features = spec.features.get("zone")          # the mock environment has no names
        self.team_components = spec.team_reward_components
        self.ship_components = spec.agent_reward_components
        # the sizes the networks are built from; the commander's views have the critic's sizes
        self.dims = dict(spec.dims, fleet_match=spec.dims["critic_match"])
        # what the commander may not see: the enemy's true state and true strength
        p0, p1 = spec.enemy_privileged
        self._fleet_enemy_keep = np.ones(spec.dims["critic_enemy"], np.float32)
        self._fleet_enemy_keep[p0:p1] = 0.0
        names = spec.features.get("critic_match") or [""] * spec.dims["critic_match"]
        self._fleet_match_keep = np.array([0.0 if n in PRIVILEGED_MATCH else 1.0 for n in names], np.float32)

        E, N = self.n_envs, self.n_agents
        self.stage = min(max(0, start_stage), len(self.stages) - 1)
        self._battle_stage = np.zeros(E, dtype=np.int64)   # the stage each running battle was started on
        self._weights = [self.reward_weights] * E          # that battle's reward weights (a stage may override some)
        self._opponent = ["rule"] * E                      # who flies its enemy: rule, passive, latest or past
        self._opponent_id = np.full(E, -1, dtype=np.int64) # which past snapshot ("past" only)
        self._commander = np.zeros(E, dtype=bool)          # the fleet commander gives orders in that battle
        self._spirit = np.zeros(E, dtype=np.float32)       # its team spirit (rewards.py)
        self._eval_battle = np.zeros(E, dtype=bool)        # that battle was started for an evaluation
        self._held_out_battle = np.zeros(E, dtype=bool)
        self._t = np.zeros(E, dtype=np.int64)              # decisions since the battle started
        self._starts = np.ones(E, dtype=np.float32)        # the observation is a battle's first
        self.opponent_policy = None            # set by MAPPO: (enemy obs, envs, snapshot ids) -> enemy actions
        self.opponent_sampler = None           # set by MAPPO: () -> a past snapshot to fight (PFSP), or None
        self.evaluating = False                # evaluation: the stage's own opponent only (no self-play)
        self.held_out = False                  # evaluation on maps training never sees
        self._world = [(0, 10**9)] * E         # generated map seed and battles fought on it, per slot
        self._raw = [None] * E                 # the last message from each battle
        self._final = [None] * E               # the last message of each battle's previous fight
        self._facts = [None] * E               # where the learning fleet's ships are relative to the circles
        self._ships = np.zeros((E, 2), dtype=np.int64)
        self._ep_return = np.zeros((E, N), dtype=np.float64)
        self._ep_terms = [dict() for _ in range(E)]
        self._ep_len = np.zeros(E, dtype=np.int64)
        self._t_start = time.time()
        self.last_reward_groups = np.zeros((E, N, len(REWARD_GROUPS)), dtype=np.float32)
        self.last_expert = None
        self.reset()

    # ------------------------------------------------------------------ the environment interface

    def reset(self) -> dict:
        for e, w in enumerate(self.workers):
            w.reset_async(self._reset_message(e))
        for e, w in enumerate(self.workers):
            self._begin(e, w.recv())
        return self._observe()

    def set_stage(self, stage: int) -> None:
        """Battles started from now on use this stage; battles already running finish on theirs."""
        self.stage = min(max(0, stage), len(self.stages) - 1)

    def set_opponent_policy(self, policy, sampler=None) -> None:
        """policy(enemy obs, envs, snapshot ids) -> enemy actions, for battles our own (frozen) fleet
        flies the enemy in; sampler() -> the id of a past snapshot to fight, or None."""
        self.opponent_policy, self.opponent_sampler = policy, sampler

    def set_evaluation(self, on: bool, held_out: bool = False) -> None:
        """Evaluation battles are against the stage's own opponent (rule AI or passive target), never
        self-play; held_out ones on maps training never uses."""
        self.evaluating, self.held_out = on, on and held_out

    def step(self, actions: np.ndarray, orders: np.ndarray | None = None):
        """actions [E, N, heads] (orders [E, N]: the commander's) -> (obs, rewards [E, N], dones [E], infos)."""
        E, N, H = self.n_envs, self.n_agents, len(self.head_sizes)
        actions = np.asarray(actions, dtype=np.int32).reshape(E, N, H)
        orders = np.zeros((E, N), dtype=np.int64) if orders is None else np.asarray(orders).reshape(E, N)
        enemy = np.zeros((E, N, H), dtype=np.int32)        # ignored where the rule AI flies the enemy
        passive = [e for e in range(E) if self._opponent[e] == "passive"]
        if passive:
            enemy[passive] = self._passive_actions(passive)
        ours = [e for e in range(E) if self._opponent[e] in ("latest", "past")]
        if ours:
            enemy[ours] = self.opponent_policy(self._observe(team=1, envs=ours), ours,
                                               [int(self._opponent_id[e]) for e in ours])
        for e, w in enumerate(self.workers):
            w.step_async(np.stack([actions[e], enemy[e]]))

        rewards = np.zeros((E, N), dtype=np.float32)
        groups = np.zeros((E, N, len(REWARD_GROUPS)), dtype=np.float32)
        dones = np.zeros(E, dtype=bool)
        infos = [{} for _ in range(E)]
        expert = np.zeros((E, N, H), dtype=np.int64)
        expert_valid = np.zeros((E, N), dtype=np.float32)
        self._starts[:] = 0.0
        for e, w in enumerate(self.workers):
            o = w.recv()
            if self.record_expert:
                # what the rule AI did with each of our ships during this step
                expert[e], expert_valid[e] = o.arrays["expert_actions"][0], o.arrays["expert_valid"][0]
            facts = self._ship_facts(o)
            arr = o.arrays
            team = dict(zip(self.team_components, arr["team_reward"][0].tolist()))
            ship = {k: arr["agent_reward"][0, :, i] for i, k in enumerate(self.ship_components)}
            their = {k: arr["agent_reward"][1, :, i] for i, k in enumerate(self.ship_components)}
            r, g, terms = compute_rewards(self._facts[e], facts, team, ship, their,
                                          int(self._ships[e, 0]), int(self._ships[e, 1]), self._weights[e],
                                          self.decision_period, orders[e], float(self._spirit[e]))
            exists = self._exists(e)
            rewards[e] = r * exists
            groups[e] = g
            self._ep_return[e] += rewards[e]
            self._ep_len[e] += 1
            self._t[e] += 1
            for k, v in terms.items():
                self._ep_terms[e][k] = self._ep_terms[e].get(k, 0.0) + float((v * exists).sum())
            self._facts[e] = facts

            if o.terminal:
                dones[e] = True
                infos[e] = self._result(e, o)
                self._final[e] = o
                w.reset_async(self._reset_message(e))
                self._begin(e, w.recv())
            else:
                self._raw[e] = o
        self.last_reward_groups = groups
        if self.record_expert:
            self.last_expert = (expert, expert_valid)
        return self._observe(), rewards, dones, infos

    def sample_random_actions(self, obs: dict) -> np.ndarray:
        """A uniformly random legal option on every head - the baseline any policy has to beat."""
        mask = obs["action_mask"]
        E, N = mask.shape[:2]
        out = np.zeros((E, N, len(self.head_sizes)), dtype=np.int32)
        off = 0
        for h, size in enumerate(self.head_sizes):
            p = mask[..., off:off + size] * self.rng.random((E, N, size))
            out[..., h] = p.argmax(-1)
            off += size
        return out

    def raw_arrays(self, e: int = 0) -> dict:
        """Everything Unity sent for battle e on the last step, by name, for both fleets."""
        return self._raw[e].arrays

    def final_arrays(self, e: int = 0) -> dict:
        """The last message of battle e's most recent finished fight (step() has already reset it)."""
        return self._final[e].arrays

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for w in self.workers:
            w.close()

    # ------------------------------------------------------------------ internals

    def _scenario(self, path: str) -> dict:
        if path not in self._scenario_cache:
            self._scenario_cache[path] = load_scenario(path)
        return self._scenario_cache[path]

    def _pick_opponent(self, stage: dict) -> tuple[str, int]:
        """Who flies the enemy in a new battle of this stage: (rule | passive | latest | past, snapshot id)."""
        kind = stage.get("opponent")
        if kind is None:                                   # older stage files: a "self_play" share
            share = float(stage.get("self_play", 0.0))
            kind, mix = ("league", {"rule": 1.0 - share, "latest": share}) if share > 0 else ("rule", None)
        else:
            mix = stage.get("opponents", {"rule": 1.0})
        if self.evaluating or self.record_expert:
            return ("passive" if kind == "passive" else "rule"), -1
        if kind != "league":
            return kind, -1
        names = list(mix)
        p = np.array([float(mix[n]) for n in names])
        choice = names[int(self.rng.choice(len(names), p=p / p.sum()))]
        if choice == "rule" or self.opponent_policy is None:
            return "rule", -1
        if choice == "past":
            ident = self.opponent_sampler() if self.opponent_sampler else None
            return ("past", int(ident)) if ident is not None else ("latest", -1)
        return "latest", -1

    def _reset_message(self, e: int) -> dict:
        stage_index = self.stage
        stage = self.stages[stage_index]
        if (not self.evaluating and not self.record_expert and stage_index > 0
                and self.rng.random() < float(stage.get("rehearsal", 0.0))):
            stage_index = int(self.rng.integers(stage_index))     # replay an earlier stage, so its skills stay
            stage = self.stages[stage_index]
        self._battle_stage[e] = stage_index
        self._weights[e] = {**self.reward_weights, **stage.get("reward_weights", {})}
        self._commander[e] = bool(stage.get("commander", False))
        self._spirit[e] = float(stage.get("team_spirit", 0.0))
        self._opponent[e], self._opponent_id[e] = self._pick_opponent(stage)
        self._eval_battle[e], self._held_out_battle[e] = self.evaluating, self.held_out
        difficulty = int(stage.get("difficulty", 0))
        # learned_teams is a bitmask: 1 = Python flies our fleet, 2 = Python flies the enemy fleet too
        msg = {"learned_teams": 1 if self._opponent[e] == "rule" else 3, "opponent_difficulty": difficulty,
               "episode_seed": int(self.rng.integers(1, 2**31 - 1))}
        if self.record_expert:
            difficulty = self.expert_difficulty
            msg.update({"learned_teams": 0, "record_teams": 1, "opponent_difficulty": difficulty})
        if "procedural" in stage:
            if self.held_out:
                battle, _ = procedural_reset(stage["procedural"], self.rng, self._world[e], held_out=True)
            else:
                battle, self._world[e] = procedural_reset(stage["procedural"], self.rng, self._world[e])
            msg.update(battle)
        else:
            paths = stage["scenario"] if isinstance(stage["scenario"], list) else [stage["scenario"]]
            sc = copy.deepcopy(self._scenario(paths[int(self.rng.integers(len(paths)))]))
            for s in sc["ships"]:
                s["x"] = float(s["x"]) + float(self.rng.uniform(-self.jitter, self.jitter))
                s["y"] = float(s["y"]) + float(self.rng.uniform(-self.jitter, self.jitter))
                s["heading"] = float(s.get("heading", 0.0)) + float(self.rng.uniform(-15, 15))
            sc["aiDifficulty"] = difficulty
            msg.update({"use_scenario": True, "scenario": sc})
            if stage.get("time_limit"):
                msg["time_limit"] = float(stage["time_limit"])
        if self.time_limit:
            msg["time_limit"] = float(self.time_limit)
        return msg

    def _begin(self, e: int, o) -> None:
        self._raw[e] = o
        self._ships[e] = o.ships
        self._facts[e] = self._ship_facts(o)
        self._ep_return[e] = 0.0
        self._ep_terms[e] = {}
        self._ep_len[e] = 0
        self._t[e] = 0
        self._starts[e] = 1.0

    def _exists(self, e: int, team: int = 0) -> np.ndarray:
        return (np.arange(self.n_agents) < self._ships[e, team]).astype(np.float32)

    def _ship_facts(self, o) -> dict:
        a = o.arrays
        if self.zone_features is None:
            return no_facts(a["alive"][0], MAX_ZONES)
        return ship_facts(a["zones"][0], a["zone_mask"][0], a["alive"][0], self.zone_features)

    def _passive_actions(self, envs: list[int]) -> np.ndarray:
        """A target that does nothing: stop and hold fire (every other head takes option 0)."""
        out = np.zeros((len(envs), self.n_agents, len(self.head_sizes)), dtype=np.int32)
        for i, e in enumerate(envs):
            mask = self._legal(self._raw[e].arrays["action_mask"][1])
            off = 0
            for h, (name, size) in enumerate(zip(self.head_names, self.head_sizes)):
                want = PASSIVE.get(name, 0)
                legal = mask[:, off:off + size] > 0.5
                out[i, :, h] = np.where(legal[:, want], want, legal.argmax(1))
                off += size
        return out

    def _result(self, e: int, o) -> dict:
        n = max(1, int(self._ships[e, 0]))
        info = {
            "episode": {"r": float(self._ep_return[e].sum() / n), "l": int(self._ep_len[e]),
                        "t": round(time.time() - self._t_start, 2)},
            "won": o.winner == 0,
            "stage": int(self._battle_stage[e]),
            "opponent": self._opponent[e],
            "opponent_id": int(self._opponent_id[e]),
            "commander": bool(self._commander[e]),
            "survived": float(o.arrays["alive"][0][:n].mean()),
            "evaluation": bool(self._eval_battle[e]),
            "held_out": bool(self._held_out_battle[e]),
            "reason": o.reason,
            "battle_time": o.battle_time,
            "reward_terms": {k: v / n for k, v in self._ep_terms[e].items()},
        }
        # Unity's end-of-battle statistics; pairs are [Player, Enemy]
        for k, v in (o.stats or {}).items():
            if isinstance(v, list) and len(v) == 2:
                info.setdefault("stats", {})[k] = v[0]
                info["stats"]["enemy_" + k] = v[1]
            elif isinstance(v, (int, float)):
                info.setdefault("stats", {})[k] = v
        return info

    def _observe(self, team: int = 0, envs=None) -> dict:
        """One fleet's observation for every battle (team 0, the learning fleet) - or, for league
        battles, the enemy fleet's (team 1) for the battles listed in envs. See the top of this file."""
        envs = range(self.n_envs) if envs is None else envs
        out = {k: [] for k in OBS_KEYS}
        for e in envs:
            a = self._raw[e].arrays
            for key in CAPTAIN_ARRAYS + CRITIC_ARRAYS:
                out[key].append(a[key][team])
            out["action_mask"].append(self._legal(a["action_mask"][team]))
            out["alive"].append(a["alive"][team])
            exists = self._exists(e, team)
            out["exists"].append(exists)
            out["own_mask"].append(exists)
            out["fleet_enemy"].append(a["critic_enemy"][team] * self._fleet_enemy_keep)
            out["fleet_match"].append(a["critic_match"][team] * self._fleet_match_keep)
            out["starts"].append(self._starts[e])
            out["t"].append(self._t[e])
            out["commander_on"].append(float(self._commander[e]))
        return {k: np.asarray(np.stack(v), dtype=np.float32) for k, v in out.items()}

    def _legal(self, mask: np.ndarray) -> np.ndarray:
        """Every head needs at least one legal option; sunk ships and empty slots get option 0."""
        mask = mask.copy()
        off = 0
        for size in self.head_sizes:
            empty = mask[:, off:off + size].sum(axis=1) < 0.5
            mask[empty, off] = 1.0
            off += size
        return mask
