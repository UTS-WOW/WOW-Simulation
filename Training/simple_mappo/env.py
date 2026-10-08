"""The naval battle as a multi-agent environment.

How it fits together:

    Python (this file)  -- actions for every ship -->  Unity headless player (one battle per process)
                        <-- observations, raw reward      simulates decision_period seconds
                            numbers, "battle over?" --

NavalEnv runs n_envs battles side by side - the same idea as make_atari_env(n_envs=4) in the Atari
notebooks. Every ship of the Player fleet is an agent; the enemy fleet is flown by the game's own
rule-based AI at the chosen difficulty, or - in self-play battles - by a frozen copy of the policy.

What reset() and step() return (E = n_envs, N = ships per side):

    obs["local"]        [E, N, local_dim]   what each ship knows: itself, its allies, the enemy
                                            contacts its team has spotted, the capture circles and
                                            the nearest islands. The actor (policy) sees only this.
    obs["state"]        [E, N, state_dim]   the critic's view: the true state of the whole battle
                                            (both fleets, every circle) + that ship's own view + which
                                            ship it is. Only used in training.
    obs["action_mask"]  [E, N, sum(heads)]  1 = the option is legal right now
    obs["alive"]        [E, N]              the ship is afloat
    obs["exists"]       [E, N]              the slot holds a ship in this battle
    rewards             [E, N]              see rewards.py
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
from .rewards import REWARD_WEIGHTS, compute_rewards, ship_facts

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)

# The observation always has room for this many ships a side, capture circles and nearby obstacles,
# whatever the battle. Smaller battles leave slots empty (zeros, with a mask), so ONE network - and
# one checkpoint - trains through every curriculum stage, including stages added later.
MAX_SHIPS = 8
MAX_ZONES = 5          # Capture and Control has five circles, the most the game makes
MAX_OBSTACLES = 4

# observation pieces, each with the mask saying which of its rows are real
LOCAL_PARTS = [("self", None), ("allies", "ally_mask"), ("contacts", "contact_mask"),
               ("zones", "zone_mask"), ("obstacles", "obstacle_mask")]
STATE_PARTS = [("critic_match", None), ("critic_own", None), ("critic_enemy", "critic_enemy_mask"),
               ("critic_zones", "critic_zone_mask")]


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
                 seed: int = 0, mock: bool = False, unity_log_dir: str | None = None, graphics: bool = False):
        """
        scenario             a battle from Training/scenarios (or a list to pick one from per battle)
        opponent_difficulty  the rule AI's skill: 0 Recruit, 1 Veteran, 2 Elite
        stages               a curriculum instead of one scenario: "default" (stages 0-4), a JSON
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
        self.head_names = [h.name for h in spec.heads]
        self.head_sizes = spec.head_sizes
        self.zone_features = spec.features.get("zone")          # the mock environment has no names
        self.team_components = spec.team_reward_components
        self.ship_components = spec.agent_reward_components

        E, N = self.n_envs, self.n_agents
        self.stage = min(max(0, start_stage), len(self.stages) - 1)
        self._battle_stage = np.zeros(E, dtype=np.int64)   # the stage each running battle was started on
        self._weights = [self.reward_weights] * E          # that battle's reward weights (a stage may override some)
        self._self_play = np.zeros(E, dtype=bool)          # the enemy of that battle is our frozen copy
        self._eval_battle = np.zeros(E, dtype=bool)        # that battle was started for an evaluation
        self._held_out_battle = np.zeros(E, dtype=bool)
        self.opponent_policy = None            # set by MAPPO: enemy observation -> enemy actions
        self.evaluating = False                # evaluation: rule AI only (no self-play)
        self.held_out = False                  # evaluation on maps training never sees
        self._world = [(0, 10**9)] * E         # generated map seed and battles fought on it, per slot
        self._raw = [None] * E                 # the last message from each battle
        self._final = [None] * E               # the last message of each battle's previous fight
        self._facts = [None] * E               # in_circle / dist_km / alive of the learning fleet
        self._ships = np.zeros((E, 2), dtype=np.int64)
        self._ep_return = np.zeros((E, N), dtype=np.float64)
        self._ep_terms = [dict() for _ in range(E)]
        self._ep_len = np.zeros(E, dtype=np.int64)
        self._t_start = time.time()

        o = self.reset()
        self.local_dim = o["local"].shape[-1]
        self.state_dim = o["state"].shape[-1]

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

    def set_opponent_policy(self, policy) -> None:
        """policy(obs) -> actions, called with the enemy fleet's observation in self-play battles."""
        self.opponent_policy = policy

    def set_evaluation(self, on: bool, held_out: bool = False) -> None:
        """Evaluation battles are against the rule AI only; held_out ones on maps training never uses."""
        self.evaluating, self.held_out = on, on and held_out

    def step(self, actions: np.ndarray):
        """actions [E, N, heads] -> (obs, rewards [E, N], dones [E], infos)."""
        E, N, H = self.n_envs, self.n_agents, len(self.head_sizes)
        actions = np.asarray(actions, dtype=np.int32).reshape(E, N, H)
        enemy = np.zeros((E, N, H), dtype=np.int32)       # ignored where the rule AI flies the enemy
        sp = np.flatnonzero(self._self_play)
        if len(sp):
            enemy[sp] = self.opponent_policy(self._observe(team=1, envs=sp))
        for e, w in enumerate(self.workers):
            w.step_async(np.stack([actions[e], enemy[e]]))

        rewards = np.zeros((E, N), dtype=np.float32)
        dones = np.zeros(E, dtype=bool)
        infos = [{} for _ in range(E)]
        for e, w in enumerate(self.workers):
            o = w.recv()
            facts = self._ship_facts(o)
            team = dict(zip(self.team_components, o.arrays["team_reward"][0].tolist()))
            ship = {k: o.arrays["agent_reward"][0, :, i] for i, k in enumerate(self.ship_components)}
            r, terms = compute_rewards(self._facts[e], facts, team, ship,
                                       int(self._ships[e, 0]), int(self._ships[e, 1]), self._weights[e],
                                       self.decision_period)
            exists = self._exists(e)
            rewards[e] = r * exists
            self._ep_return[e] += rewards[e]
            self._ep_len[e] += 1
            for k, v in terms.items():
                self._ep_terms[e][k] = self._ep_terms[e].get(k, 0.0) + float((v * exists).sum())
            self._facts[e] = facts

            if o.terminal:
                dones[e] = True
                infos[e] = self._result(e, o)
                self._final[e] = o
                w.reset_async(self._reset_message(e))
                o = w.recv()
                self._begin(e, o)
            else:
                self._raw[e] = o
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

    def _reset_message(self, e: int) -> dict:
        stage = self.stages[self.stage]
        self._battle_stage[e] = self.stage
        self._weights[e] = {**self.reward_weights, **stage.get("reward_weights", {})}
        difficulty = int(stage.get("difficulty", 0))
        msg = {"learned_teams": 1, "opponent_difficulty": difficulty,
               "episode_seed": int(self.rng.integers(1, 2**31 - 1))}
        self._eval_battle[e], self._held_out_battle[e] = self.evaluating, self.held_out
        self._self_play[e] = (not self.evaluating and self.opponent_policy is not None
                              and self.rng.random() < float(stage.get("self_play", 0.0)))
        if self._self_play[e]:
            msg["learned_teams"] = 3                       # both fleets take their orders from Python
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

    def _exists(self, e: int) -> np.ndarray:
        return (np.arange(self.n_agents) < self._ships[e, 0]).astype(np.float32)

    def _ship_facts(self, o) -> dict:
        a = o.arrays
        if self.zone_features is None:
            z = np.zeros(self.n_agents, dtype=np.float32)
            return {"in_circle": z, "dist_km": z, "alive": a["alive"][0].astype(np.float32)}
        return ship_facts(a["zones"][0], a["zone_mask"][0], a["alive"][0], self.zone_features)

    def _result(self, e: int, o) -> dict:
        n = max(1, int(self._ships[e, 0]))
        info = {
            "episode": {"r": float(self._ep_return[e].sum() / n), "l": int(self._ep_len[e]),
                        "t": round(time.time() - self._t_start, 2)},
            "won": o.winner == 0,
            "stage": int(self._battle_stage[e]),
            "opponent": "self" if self._self_play[e] else "rule",
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
        """The learning fleet's observation (team 0) for every battle, or - for self-play - the enemy
        fleet's actor inputs (team 1) for the battles listed in envs."""
        envs = range(self.n_envs) if envs is None else envs
        N = self.n_agents
        local, state, mask, alive, exists = [], [], [], [], []
        for e in envs:
            a = self._raw[e].arrays
            loc = np.concatenate([np.concatenate([a[k][team].reshape(N, -1)] + ([a[m][team].reshape(N, -1)] if m else []), axis=1)
                                  for k, m in LOCAL_PARTS], axis=1)
            local.append(loc)
            mask.append(self._legal(a["action_mask"][team]))
            if team == 0:
                glob = np.concatenate([np.concatenate([a[k][0].ravel()] + ([a[m][0].ravel()] if m else []))
                                       for k, m in STATE_PARTS])
                # MAPPO's "agent-specific global state": the whole battle, plus this ship's own view and id
                state.append(np.concatenate([np.repeat(glob[None], N, axis=0), loc, np.eye(N, dtype=np.float32)], axis=1))
                alive.append(a["alive"][0])
                exists.append(self._exists(e))
        out = {"local": np.stack(local).astype(np.float32), "action_mask": np.stack(mask).astype(np.float32)}
        if team == 0:
            out.update({"state": np.stack(state).astype(np.float32), "alive": np.stack(alive).astype(np.float32),
                        "exists": np.stack(exists)})
        return out

    def _legal(self, mask: np.ndarray) -> np.ndarray:
        """Every head needs at least one legal option; sunk ships and empty slots get option 0."""
        mask = mask.copy()
        off = 0
        for size in self.head_sizes:
            empty = mask[:, off:off + size].sum(axis=1) < 0.5
            mask[empty, off] = 1.0
            off += size
        return mask
