"""PQN - the Parallelised Q-Network: the value-based method compared with MAPPO.

    Gallici, Fellows, Ellis, Pou, Masmitja, Foerster, Martin, "Simplifying Deep Temporal Difference
    Learning", ICLR 2025 (arXiv 2407.04811) - code: github.com/mttga/purejaxql

Deep Q-learning as DQN made it needs a large replay buffer and a target network to stay stable. PQN
shows that with normalisation in the Q-network (LayerNorm) neither is needed, and replaces them with
what an on-policy method like PPO does: many battles in parallel, a short rollout from each, and a few
epochs of minibatch updates on it, after which the rollout is thrown away. On SMAX (StarCraft
multi-agent micro-management) PQN with value decomposition beat both QMIX and MAPPO.

    rollout     num_steps (128) decisions in every battle, as chunks of chunk_len (32) with each ship's
                GRU memory at the chunk's start [R2D2: Kapturowski et al. 2019]
    targets     Q(lambda), backwards along each chunk, from the network as it was when the rollout
                ended (a frozen copy for the update):
                    R_t = r_t + gamma * [ lambda * R_t+1 + (1 - lambda) * max_a Q(s_t+1, a) ]
                0 after a battle's end; lambda 0.85, gamma 0.99 (the paper's SMAX setting). With
                gamma * lambda = 0.84, 0.4 % of a target is left beyond a 32-decision chunk.
    fleet       PQN-VDN [Sunehag et al. 2018]: the fleet's Q is the sum of its ships' Q-values, trained
                on the fleet's reward (mixer="none": every ship on its own reward)
    update      epochs (4) x minibatches (4) of squared TD error, gradients clipped at 1, Adam 2.5e-4
    heads       six action heads, dueling and branching [Wang et al. 2016, Tavakoli et al. 2018]:
                Q_d(s, a_d) = V(s) + A_d(s, a_d) - mean A_d; the captain's head scores are the A_d,
                so the Q-network IS the captain (the same network as MAPPO's, with LayerNorm after
                its encoder and in every attention block - PQN's stabiliser), and the game flies it
                with the same C# code, taking the best option
    explore     exploration="softmax": Boltzmann, options in proportion to exp(Q / temperature);
                "epsilon": epsilon-greedy (the paper's). From the Elite clone, whose scores x 0.1 are
                the first advantages, softmax with temperature 0.1 first plays the clone's own
                choices (greedy play of a cloned policy keeps only its likeliest option per head)
    commander   its scores for (ship, order) are Q-values; one-step intra-option Q-learning
                [h-DQN: Kulkarni et al. 2016], epsilon-greedy orders
    clone start the clone's scores x 0.1 become the first advantages, and a fading kickstarting
                penalty KL(clone || softmax(A / 0.1)) keeps it close at first [Schmitt et al. 2018]

Why it suits this game: no replay buffer, so no stale data under self-play, where the opponents keep
changing; lambda-returns carry a battle's result back over many one-second decisions; and it does
little network work per timestep (measured with 4 battles at once: 276 / 133 / 51 timesteps/s on
stages 0 / 4 / 8, against MAPPO's 292 / 223 / 72).
"""

from __future__ import annotations

import copy
import csv
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .callbacks import BaseCallback, CallbackList
from .league import LeagueMixin, OpponentPool
from .networks import CAPTAIN_KEYS, NEG_INF, Commander, FleetPolicy, Memory, kl_to, layer, same_heads, to_tensors

CRITIC_KEYS = ["critic_own", "own_mask", "critic_enemy", "critic_enemy_mask", "critic_zones", "critic_zone_mask",
               "critic_match"]
# what one chunk of a rollout keeps of every observation (the commander's view is rebuilt from
# the critic's arrays when training, so it is not stored twice)
STORED_OBS = CAPTAIN_KEYS + ["action_mask", "alive", "exists"] + [k for k in CRITIC_KEYS if k != "own_mask"] + \
             ["starts", "t", "commander_on"]
SHIP_AXIS = set(CAPTAIN_KEYS) | {"action_mask", "alive", "exists", "critic_own", "actions", "orders", "rewards"}
ENEMY_AXIS = {"critic_enemy", "critic_enemy_mask"}
MEMORY = {"h0", "teacher_h0"}                                                            # [ships, hidden]
SMALL = {"action_mask", "alive", "exists", "critic_enemy_mask", "critic_zone_mask", "ally_mask", "contact_mask",
         "zone_mask", "obstacle_mask", "starts", "commander_on"}                       # 0/1 values: one byte each
PROGRESS_FIELDS = ["total_timesteps", "updates", "gradient_steps", "fps", "time_elapsed", "stage", "episodes",
                   "ep_rew_mean", "ep_len_mean", "win_rate", "pass_rate", "win_rate_self_play", "q_loss", "q_value",
                   "target_value", "kl_bc", "commander_loss", "epsilon", "orders_free", "rollout_decisions", "elo"]


def _linear(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, progress))


# ---------------------------------------------------------------------------------------------- one rollout

class RolloutChunks:
    """One rollout: chunks of consecutive decisions of one fleet (one row of the environment), each with
    the captains' memory at its start. A chunk holds chunk_len + 1 observations: the last one is only
    there to bootstrap from (it is also the first of the next chunk). Observations are kept as float16
    and 0/1 masks as bytes, and only the ship slots the battle uses, so a 1v1 chunk is small."""

    def __init__(self, chunk_len: int):
        self.chunk_len = chunk_len
        self.chunks: list[dict] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def decisions(self) -> int:
        return len(self.chunks) * self.chunk_len

    def add(self, chunk: dict) -> None:
        exists = chunk["exists"]
        n = int(np.flatnonzero(exists.any(0)).max()) + 1 if exists.any() else 1
        enemy = chunk["critic_enemy_mask"]
        m = int(np.flatnonzero(enemy.any(0)).max()) + 1 if enemy.any() else 1
        kept = {}
        for k, v in chunk.items():
            if k in SHIP_AXIS:
                v = v[:, :n]
            elif k in ENEMY_AXIS:
                v = v[:, :m]
            elif k in MEMORY:
                v = v[:n]
            if k in SMALL:
                v = v.astype(np.uint8)
            elif v.dtype == np.int64:
                v = v.astype(np.int16)
            elif v.dtype == np.float32 and k != "rewards" and k not in MEMORY:
                v = v.astype(np.float16)
            kept[k] = np.ascontiguousarray(v)
        self.chunks.append(kept)

    def stack(self, indices) -> dict:
        """The chunks at `indices`, padded to the same number of ship slots (PQN's minibatches use it too)."""
        picks = [self.chunks[i] for i in indices]
        keys = set.intersection(*(set(c) for c in picks))             # (teacher_h0 only after an imitation start)
        n = max(c["exists"].shape[1] for c in picks)
        m = max(c["critic_enemy_mask"].shape[1] for c in picks)
        out = {}
        for k in keys:
            parts = []
            for c in picks:
                v = c[k]
                axis, size = (1, n) if k in SHIP_AXIS else (1, m) if k in ENEMY_AXIS else (0, n) if k in MEMORY else (None, 0)
                if axis is not None and v.shape[axis] < size:
                    pad = [(0, 0)] * v.ndim
                    pad[axis] = (0, size - v.shape[axis])
                    v = np.pad(v, pad)
                parts.append(v)
            out[k] = np.stack(parts, axis=0 if k in MEMORY else 1)
        return out


# ---------------------------------------------------------------------------------------------- PQN

class FleetPQN(LeagueMixin):
    GREEDY = True                       # its own copies in the league choose by their Q-values

    def __init__(self, env, learning_rate: float = 2.5e-4, commander_learning_rate: float = 1e-4,
                 num_steps: int = 128, epochs: int = 4, minibatches: int = 4, lam: float = 0.85,
                 chunk_len: int = 32, gamma: float = 0.99, max_grad_norm: float = 1.0, mixer: str = "vdn",
                 exploration: str = "softmax", temperature: float = 0.1,
                 exploration_initial_eps: float = 1.0, exploration_final_eps: float = 0.01,
                 exploration_timesteps: int = 300_000, commander_eps_scale: float = 0.5, eval_epsilon: float = 0.02,
                 opponent_epsilon: float = 0.02, d_model: int = 64, attn_heads: int = 4, attn_layers: int = 2,
                 hidden_size: int = 128, recurrent: bool = True, commander: bool = True, commander_period: int = 10,
                 kl_bc_coef: float = 0.5, kl_bc_final: float = 0.05, kl_bc_steps: int = 5_000,
                 imitation_temperature: float = 0.1, opponent_refresh: int = 20, snapshot_every: int = 50,
                 pool_size: int = 20, pfsp: str = "hard", device: str = "auto", torch_threads: int | None = 4,
                 seed: int = 0, verbose: int = 1, log_dir: str | None = None):
        """
        learning_rate          Adam step size of the Q-network (captain + value stream); the commander
                               has commander_learning_rate
        num_steps              decisions per battle in a rollout (a multiple of chunk_len); a rollout of
                               num_steps x battles decisions is learned from, then thrown away
        epochs, minibatches    passes over the rollout, and minibatches per pass [PQN: 4 epochs]
        lam, gamma             lambda of the Q(lambda) targets and the discount per decision
        chunk_len              decisions per chunk: the unit the GRU replays from its stored memory
        max_grad_norm          gradient clipping
        mixer                  "vdn" (PQN-VDN: the fleet's Q is its ships' sum, on the fleet's reward)
                               or "none" (every ship on its own reward)
        exploration            "softmax" (Boltzmann: options in proportion to exp(Q / temperature)) or
                               "epsilon" (the best option, or with probability epsilon a random one);
                               deterministic play takes the best option either way
        exploration_*          epsilon falls from initial to final over exploration_timesteps (counted
                               from when the model was made) - epsilon-greedy captains, and always the
                               commander's orders (commander_eps_scale x epsilon)
        eval_epsilon           epsilon of predict() for epsilon-greedy play (evaluations, replays);
                               opponent_epsilon for the frozen copies in league battles
        d_model ... commander_period   the captain and commander - the same as MAPPO's
        kl_bc_coef             after an imitation start: weight of the kickstarting penalty, fading to
                               kl_bc_final over kl_bc_steps gradient steps
        imitation_temperature  tau: the clone's scores x tau are the starting advantages
        opponent_refresh       updates (rollouts) between refreshing the "latest" opponent;
                               snapshot_every between adding a past copy to the league
        """
        if mixer not in ("vdn", "none"):
            raise ValueError(f"mixer must be 'vdn' or 'none', not {mixer!r}")
        if exploration not in ("epsilon", "softmax"):
            raise ValueError(f"exploration must be 'epsilon' or 'softmax', not {exploration!r}")
        if num_steps % chunk_len:
            raise ValueError(f"num_steps ({num_steps}) must be a multiple of chunk_len ({chunk_len})")
        self.env = env
        self.hp = dict(learning_rate=learning_rate, commander_learning_rate=commander_learning_rate,
                       num_steps=num_steps, epochs=epochs, minibatches=minibatches, lam=lam, chunk_len=chunk_len,
                       gamma=gamma, max_grad_norm=max_grad_norm, mixer=mixer, exploration=exploration,
                       temperature=temperature, exploration_initial_eps=exploration_initial_eps,
                       exploration_final_eps=exploration_final_eps, exploration_timesteps=exploration_timesteps,
                       commander_eps_scale=commander_eps_scale, eval_epsilon=eval_epsilon,
                       opponent_epsilon=opponent_epsilon, d_model=d_model, attn_heads=attn_heads,
                       attn_layers=attn_layers, hidden_size=hidden_size, recurrent=recurrent, commander=commander,
                       commander_period=commander_period, kl_bc_coef=kl_bc_coef, kl_bc_final=kl_bc_final,
                       kl_bc_steps=kl_bc_steps, imitation_temperature=imitation_temperature,
                       opponent_refresh=opponent_refresh, snapshot_every=snapshot_every, pool_size=pool_size,
                       pfsp=pfsp, seed=seed)
        self.verbose = verbose
        self.log_dir = log_dir
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if torch_threads:
            torch.set_num_threads(torch_threads)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self.dims, self.heads = dict(env.dims), [dict(h) for h in env.heads]
        self.head_sizes, self.head_names = [h["size"] for h in self.heads], [h["name"] for h in self.heads]
        self.arch = dict(d=d_model, attn_heads=attn_heads, layers=attn_layers, hidden=hidden_size, recurrent=recurrent)
        # the Q-network: the captain (its head scores are the advantages A_d), the value stream V, and
        # the commander (its order scores are Q-values)
        self.policy = FleetPolicy(self.dims, self.heads, commander=commander, commander_period=commander_period,
                                  **self.arch).to(self.device)
        self.actor = self.policy                                    # the name the Week 9/10 notebooks print
        self.value_head = layer(hidden_size + d_model, 1, gain=1.0).to(self.device)
        self.mixer = None                                           # VDN is a plain sum: nothing to learn
        # the network as it was when the rollout ended: the targets of an update come from it [PQN]
        self.frozen_policy = copy.deepcopy(self.policy).eval()
        self.frozen_value_head = copy.deepcopy(self.value_head).eval()
        q_params = list(self.policy.captain.parameters()) + list(self.value_head.parameters())
        self.q_optimizer = torch.optim.Adam(q_params, lr=learning_rate, eps=1e-5)
        self.commander_optimizer = (torch.optim.Adam(self.policy.commander.parameters(), lr=commander_learning_rate,
                                                     eps=1e-5) if commander else None)
        self.rollout = RolloutChunks(chunk_len)
        # the commander's view is the critic's arrays with what the team cannot know zeroed (env.py)
        n_enemy, n_match = self.dims["critic_enemy"], self.dims["critic_match"]
        self._enemy_keep = torch.as_tensor(getattr(env, "_fleet_enemy_keep", np.ones(n_enemy)), dtype=torch.float32,
                                           device=self.device)
        self._match_keep = torch.as_tensor(getattr(env, "_fleet_match_keep", np.ones(n_match)), dtype=torch.float32,
                                           device=self.device)

        # the imitation clone the kickstarting penalty pulls towards (set by init_from_imitation)
        self.teacher: FleetPolicy | None = None
        self.steps_since_teacher = 0

        # the league, exactly as MAPPO's
        self.opponent = copy.deepcopy(self.policy).eval()
        self.league = OpponentPool(pool_size, seed, pfsp=pfsp)
        self._past: dict[int, FleetPolicy] = {}
        if hasattr(env, "set_opponent_policy"):
            env.set_opponent_policy(self._opponent_act, self.league.sample)
        if verbose:
            n_q = sum(p.numel() for p in self.policy.captain.parameters()) + sum(p.numel() for p in self.value_head.parameters())
            n_cmd = sum(p.numel() for p in self.policy.commander.parameters()) if commander else 0
            print(f"PQN on {self.device}: Q-network (captain + value stream) {n_q:,} parameters"
                  + (f", commander {n_cmd:,}" if commander else "") + f", mixer: {mixer}, exploration: {exploration}"
                  + f"; rollouts of {num_steps} decisions per battle, heads {dict(zip(self.head_names, self.head_sizes))}")

        self.num_timesteps = 0          # decisions, summed over the parallel battles (as MAPPO)
        self.n_updates = 0              # rollouts learned from
        self.n_gradient_steps = 0
        self._eps_start_timesteps = 0   # exploration is scheduled from here (the model's birth)
        self._last_obs = None
        self._episodes = []
        self._memory = self._play_memory = self._opp_memory = self._teacher_memory = None
        self._t0 = None
        self._start_timesteps = 0
        self._total_timesteps = None
        self.last_orders = None

    # ------------------------------------------------------------------ acting

    @property
    def exploration_rate(self) -> float:
        hp = self.hp
        progress = (self.num_timesteps - self._eps_start_timesteps) / max(1, hp["exploration_timesteps"])
        return _linear(hp["exploration_initial_eps"], hp["exploration_final_eps"], progress)

    def _tensors(self, obs: dict) -> dict:
        return to_tensors(obs, self.device)

    @staticmethod
    def _fits(memory: Memory | None, obs: dict) -> bool:
        return memory is not None and memory.h.shape[:2] == obs["alive"].shape

    @property
    def act_temperature(self) -> float | None:
        """The captains' Boltzmann temperature when exploration="softmax", else None (epsilon-greedy)."""
        return self.hp["temperature"] if self.hp.get("exploration", "epsilon") == "softmax" else None

    def predict(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        """Actions [E, N, heads]: the best ones (epsilon = eval_epsilon, 0 if deterministic) - or, with
        exploration="softmax", sampled from softmax(Q / temperature) as in training."""
        if not self._fits(self._play_memory, obs):
            self._play_memory = self.policy.new_memory(*obs["alive"].shape)
        self.policy.eval()
        eps = 0.0 if deterministic else self.hp["eval_epsilon"]
        out = self.policy.act(self._tensors(obs), self._play_memory, epsilon=eps, commander_epsilon=eps,
                              deterministic=deterministic, temperature=self.act_temperature)
        self.last_orders = out["orders"].cpu().numpy()
        return out["actions"].cpu().numpy()

    def reset_battles(self) -> None:
        self._last_obs = self.env.reset()

    # ------------------------------------------------------------------ training loop [PQN]

    def learn(self, total_timesteps: int, callback: BaseCallback | list | None = None, log_interval: int = 2):
        """Alternate between a rollout of num_steps decisions in every battle and `epochs` passes of
        minibatch updates on it; then the rollout is thrown away and the next one played."""
        callback = CallbackList(callback if isinstance(callback, list) else [callback] if callback else [])
        callback.init_callback(self)
        if hasattr(self.env, "set_opponent_policy"):
            self.env.set_opponent_policy(self._opponent_act, self.league.sample)   # another model may have taken it over
        self._last_obs = self.env.reset()            # fresh battles (an evaluation may have run in between)
        self._t0 = time.time()
        start = self._start_timesteps = self.num_timesteps
        self._total_timesteps = total_timesteps      # for status reports (speed, time left)
        hp = self.hp
        stats = defaultdict(list)
        while self.num_timesteps < total_timesteps:
            self.rollout = RolloutChunks(hp["chunk_len"])
            stopped = False
            for _ in range(hp["num_steps"] // hp["chunk_len"]):
                if not self.collect_chunks(callback):
                    stopped = True
                    break
            if len(self.rollout):
                self._update(stats)
            self.n_updates += 1
            if self.n_updates % hp["opponent_refresh"] == 0:
                self.opponent.load_state_dict(self.policy.state_dict())
            self._league_step()                           # past copies, the champion (league.py)
            if log_interval and self.n_updates % log_interval == 0:
                self._log({k: float(np.mean(v)) for k, v in stats.items()}, start)
                stats = defaultdict(list)
            if stopped or not callback.on_rollout_end():
                break
        callback.on_training_end()
        return self

    def collect_chunks(self, callback) -> bool:
        """Plays chunk_len decisions in every battle (exploring as `exploration` says) and adds one chunk
        per fleet (row) to the rollout - with the captains' memory at its start, so training can replay it."""
        L = self.hp["chunk_len"]
        obs = self._last_obs
        R, N = obs["alive"].shape                   # rows: a fleet in a battle (two per battle in mirror curricula)
        n_battles = getattr(self.env, "n_envs", R)
        if not self._fits(self._memory, obs):
            self._memory = self.policy.new_memory(R, N)
        if self.teacher is not None and not self._fits(self._teacher_memory, obs):
            self._teacher_memory = self.teacher.new_memory(R, N)
        eps = self.exploration_rate
        buf = {k: np.zeros((L + 1, *obs[k].shape), np.float32) for k in STORED_OBS}
        buf.update({"actions": np.zeros((L + 1, R, N, len(self.head_sizes)), np.int64),
                    "orders": np.zeros((L + 1, R, N), np.int64),
                    "rewards": np.zeros((L + 1, R, N), np.float32),
                    "dones": np.zeros((L + 1, R), np.float32)})
        h0 = teacher_h0 = None
        self.policy.eval()
        for t in range(L):
            tens = self._tensors(obs)
            out = self.policy.act(tens, self._memory, epsilon=eps,
                                  commander_epsilon=eps * self.hp["commander_eps_scale"], temperature=self.act_temperature)
            if t == 0:
                h0 = out["h"].cpu().numpy()          # the memory each ship starts the chunk with [R2D2]
            if self.teacher is not None:             # the clone follows along (as a free ship, the way it was
                teacher_h = self.teacher.act(tens, self._teacher_memory)["h"]    # trained), so its memory is known
                if t == 0:
                    teacher_h0 = teacher_h.cpu().numpy()
            for k in STORED_OBS:
                buf[k][t] = obs[k]
            buf["actions"][t], buf["orders"][t] = out["actions"].cpu().numpy(), out["orders"].cpu().numpy()
            obs, rewards, dones, infos = self.env.step(buf["actions"][t], buf["orders"][t])
            buf["rewards"][t], buf["dones"][t] = rewards, dones
            for info in infos:
                if info:
                    self._episodes.append(info)
                    if not info.get("evaluation"):
                        stage = self.env.stages[min(info["stage"], len(self.env.stages) - 1)]
                        self.league.record(info, difficulty=int(stage.get("difficulty", 0)))
            self.num_timesteps += n_battles
            if not callback.on_step(locals()):
                self._last_obs = obs
                return False
        for k in STORED_OBS:                         # the observation to bootstrap from
            buf[k][L] = obs[k]
        buf["orders"][L] = buf["orders"][L - 1]      # orders stay in force until the commander's next decision
        self._last_obs = obs
        for r in range(R):
            if buf["exists"][:, r].any():            # an enemy row of a battle that was not a mirror one: nothing
                chunk = {k: v[:, r] for k, v in buf.items()}
                chunk["h0"] = h0[r]
                if teacher_h0 is not None:
                    chunk["teacher_h0"] = teacher_h0[r]
                self.rollout.add(chunk)
        return True

    def _update(self, stats: dict) -> None:
        """`epochs` passes over the rollout in `minibatches` minibatches. The targets come from the
        network as it was when the rollout ended (the frozen copy - unchanged during the update)."""
        hp = self.hp
        self.frozen_policy.load_state_dict(self.policy.state_dict())
        self.frozen_value_head.load_state_dict(self.value_head.state_dict())
        n = len(self.rollout)
        for _ in range(hp["epochs"]):
            for idx in np.array_split(self._rng.permutation(n), min(hp["minibatches"], n)):
                raw = self.rollout.stack(idx)
                b = {k: torch.as_tensor(v.astype(np.int64) if k in ("actions", "orders") else v.astype(np.float32),
                                        device=self.device) for k, v in raw.items()}
                self._train_minibatch(b, stats)

    def _lambda_targets(self, rewards: torch.Tensor, dones: torch.Tensor, next_q: torch.Tensor) -> torch.Tensor:
        """Q(lambda) targets over a chunk, backwards [PQN]: R_L = max Q(s_L) and
        R_t = r_t + gamma * (1 - done_t) * (lambda * R_t+1 + (1 - lambda) * max Q(s_t+1)).
        rewards / dones / next_q: [T1, B(, N)] (next_q[t] = max Q at s_t) -> [T1 - 1, B(, N)]."""
        g, lam = self.hp["gamma"], self.hp["lam"]
        L = rewards.shape[0] - 1
        ret = next_q[L]
        ys = []
        for t in reversed(range(L)):
            d = dones[t]
            if d.dim() < ret.dim():                          # per-ship targets: one end flag per battle
                d = d.unsqueeze(-1)
            ret = rewards[t] + g * (1.0 - d) * (lam * ret + (1.0 - lam) * next_q[t + 1])
            ys.append(ret)
        return torch.stack(ys[::-1])

    def _train_minibatch(self, b: dict, stats: dict) -> None:
        """One gradient step on a minibatch of chunks: every head's Q of the action taken towards the
        Q(lambda) target (the heads share it [BDQ]), plus the kickstarting penalty; then the commander."""
        hp = self.hp
        T1, B, N = b["alive"].shape
        live = b["alive"] * b["exists"]
        view = self._fleet_view(b, T1, B)
        H = len(self.head_sizes)
        self.policy.train()
        q_heads, adv = self._ship_q(self.policy.captain, self.value_head, b, T1, B, N)
        acts = b["actions"].reshape(T1, B * N, H)
        taken = torch.stack([q.gather(-1, acts[..., d:d + 1]).squeeze(-1) for d, q in enumerate(q_heads)], -1)
        q_fleet = self._mix(taken.view(T1, B, N, H), live)                          # [T1, B, (N,) heads]
        with torch.no_grad():
            fq, _ = self._ship_q(self.frozen_policy.captain, self.frozen_value_head, b, T1, B, N)
            best = torch.stack([q.max(-1).values for q in fq], -1).view(T1, B, N, H)   # max_a Q per head
            next_fleet = self._mix(best, live).mean(-1)                               # the heads' mean [BDQ]
            if hp["mixer"] == "none":
                rewards = b["rewards"]
            else:                                            # the fleet's reward: its ships' average
                exists = b["exists"]
                rewards = (b["rewards"] * exists).sum(-1) / exists.sum(-1).clamp_min(1.0)
            y = self._lambda_targets(rewards, b["dones"], next_fleet)
        valid = live[:-1] if hp["mixer"] == "none" else (b["exists"][:-1].sum(-1) > 0).float()
        td = 0.5 * (q_fleet[:-1] - y.unsqueeze(-1)).pow(2).mean(-1)                   # squared error [PQN]
        loss = (td * valid).sum() / valid.sum().clamp_min(1.0)
        stats["q_loss"].append(loss.item())
        kl_coef = _linear(hp["kl_bc_coef"], hp["kl_bc_final"], self.steps_since_teacher / max(1, hp["kl_bc_steps"]))
        kl = self._kickstart(b, adv, live, T1, B, N) if kl_coef > 0 else None
        if kl is not None:
            stats["kl_bc"].append(kl.item())
            loss = loss + kl_coef * kl
        self.q_optimizer.zero_grad()
        loss.backward()
        params = [p for g in self.q_optimizer.param_groups for p in g["params"]]
        nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
        self.q_optimizer.step()
        stats["q_value"].append(float((q_fleet[:-1].detach().mean(-1) * valid).sum() / valid.sum().clamp_min(1.0)))
        stats["target_value"].append(float((y * valid).sum() / valid.sum().clamp_min(1.0)))
        if self.policy.commander is not None and b["commander_on"].sum() > 0:
            self._commander_step(b, view, T1, B, N, stats)
        self.n_gradient_steps += 1
        self.steps_since_teacher += 1

    # ------------------------------------------------------------------ the Q-values

    def _head_q(self, adv: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        """Dueling Q-values of every head [Duel, BDQ]: V + A_d - mean of A_d over the legal options."""
        out = []
        for a, legal in zip(torch.split(adv, self.head_sizes, -1), torch.split(mask, self.head_sizes, -1)):
            legal = legal > 0.5
            a = torch.where(legal, a, torch.zeros_like(a))
            mean = a.sum(-1, keepdim=True) / legal.sum(-1, keepdim=True).clamp_min(1)
            out.append((value + a - mean).masked_fill(~legal, NEG_INF))
        return out

    @staticmethod
    def _ship_inputs(b: dict, T1: int, B: int, N: int):
        """The captain's inputs over the chunk, one sequence per ship slot: obs [T1, B*N, ...], starts [T1, B*N]."""
        obs = {k: b[k].reshape(T1, B * N, *b[k].shape[3:]) for k in CAPTAIN_KEYS}
        return obs, b["starts"].unsqueeze(-1).expand(T1, B, N).reshape(T1, B * N)

    def _ship_q(self, captain, value_head, b: dict, T1: int, B: int, N: int):
        """Every ship's Q per head over the chunk (list of [T1, B*N, options]) and the advantages."""
        obs, starts = self._ship_inputs(b, T1, B, N)
        adv, feats = captain.sequence(obs, b["orders"].reshape(T1, B * N), b["h0"].reshape(B * N, -1), starts,
                                      features=True)
        return self._head_q(adv, value_head(feats), b["action_mask"].reshape(T1, B * N, -1)), adv

    def _kickstart(self, b: dict, adv: torch.Tensor, live: torch.Tensor, T1: int, B: int, N: int) -> torch.Tensor | None:
        """[Kick] KL(clone's choice || softmax(advantages)) per head, summed, over the live ship-decisions."""
        if self.teacher is None or "teacher_h0" not in b:
            return None
        obs, starts = self._ship_inputs(b, T1, B, N)
        with torch.no_grad():
            free = torch.zeros(T1, B * N, dtype=torch.long, device=adv.device)
            ref = self.teacher.captain.sequence(obs, free, b["teacher_h0"].reshape(B * N, -1), starts)
        tau = self.hp["imitation_temperature"]
        kl = kl_to(ref, adv / tau, b["action_mask"].reshape(T1, B * N, -1), self.head_sizes).view(T1, B, N)
        return (kl * live).sum() / live.sum().clamp_min(1.0)

    def _fleet_view(self, b: dict, T1: int, B: int) -> dict:
        """The commander's inputs [T1 * B, ...]: the critic's arrays with what the team cannot know zeroed."""
        flat = {k: b[k].reshape(T1 * B, *b[k].shape[2:]) for k in CRITIC_KEYS if k != "own_mask"}
        flat["own_mask"] = b["exists"].reshape(T1 * B, -1)
        flat["alive"] = b["alive"].reshape(T1 * B, -1)
        flat["fleet_enemy"] = flat["critic_enemy"] * self._enemy_keep
        flat["fleet_match"] = flat["critic_match"] * self._match_keep
        return flat

    def _mix(self, q: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        """Ships' Q per head [T1, B, N, D] -> the fleet's: their sum (VDN) [T1, B, D], or per ship
        [T1, B, N, D] for mixer="none". Ships that are sunk or do not exist count 0."""
        q = q * live.unsqueeze(-1)
        return q.sum(2) if self.hp["mixer"] == "vdn" else q

    def _commander_step(self, b: dict, view: dict, T1: int, B: int, N: int, stats: dict) -> None:
        """Intra-option Q-learning for the commander [h-DQN, Options], at every decision of the chunk:

            Q_fleet(s, orders) = sum over live ships of q(s, ship, its order)
            target = r + gamma * ( Q_frozen(s', the same orders)          if the orders stay in force
                                   Q_frozen(s', online's best orders)     at the commander's next decision )
        """
        g = self.hp["gamma"]
        own_live = (view["own_mask"] * view["alive"]).view(T1, B, -1)
        mask = Commander.order_mask(view)
        q = self.policy.commander(view).masked_fill(mask < 0.5, NEG_INF).view(T1, B, N, -1)
        orders = b["orders"]
        q_now = (q.gather(-1, orders.unsqueeze(-1)).squeeze(-1).masked_fill(own_live < 0.5, 0.0) * own_live).sum(-1)
        with torch.no_grad():
            tq = self.frozen_policy.commander(view).masked_fill(mask < 0.5, NEG_INF).view(T1, B, N, -1)
            best = q.argmax(-1, keepdim=True)
            keep = (tq.gather(-1, orders.unsqueeze(-1)).squeeze(-1).masked_fill(own_live < 0.5, 0.0) * own_live).sum(-1)
            switch = (tq.gather(-1, best).squeeze(-1).masked_fill(own_live < 0.5, 0.0) * own_live).sum(-1)
            decides = (b["commander_on"] > 0.5) & (b["t"].long() % self.hp["commander_period"] == 0)
            u = torch.where(decides, switch, keep)                                          # [T1, B]
            exists = b["exists"]
            reward = (b["rewards"] * exists).sum(-1) / exists.sum(-1).clamp_min(1.0)
            y = reward[:-1] + g * (1.0 - b["dones"][:-1]) * u[1:]
        on = (b["commander_on"][:-1] > 0.5).float() * (own_live[:-1].sum(-1) > 0).float()
        loss = (F.smooth_l1_loss(q_now[:-1], y, reduction="none") * on).sum() / on.sum().clamp_min(1.0)
        self.commander_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.commander.parameters(), self.hp["max_grad_norm"])
        self.commander_optimizer.step()
        stats["commander_loss"].append(loss.item())
        decided = decides[:-1] & (on > 0.5)
        if decided.any():
            chosen = orders[:-1][decided]                                                    # [k, N]
            alive = own_live[:-1][decided] > 0.5
            stats["orders_free"].append(float((chosen[alive] == 0).float().mean()) if alive.any() else 1.0)

    # ------------------------------------------------------------------ imitation warm start

    def init_from_imitation(self, path: str, start_epsilon: float | None = 0.3) -> None:
        """Starts the Q-network's captain from the imitation clone (imitation.py): its head scores become
        the advantages, so - with Boltzmann exploration - it starts by choosing as the Elite AI does.
        Epsilon then starts at start_epsilon instead of 1.0 (random play would only throw that away)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["dims"] != {k: v for k, v in self.dims.items() if k in ck["dims"]} or ck["arch"] != self.arch:
            raise ValueError(f"{path} was trained for another network (dims / architecture differ): "
                             f"{ck['arch']} vs {self.arch}")
        self.policy.captain.load_state_dict(ck["captain"])
        with torch.no_grad():                            # advantages = tau x the clone's scores (the same argmax)
            tau = self.hp["imitation_temperature"]
            for lin in list(self.policy.captain.fixed) + list(self.policy.captain.query.values()):
                lin.weight.mul_(tau)
                lin.bias.mul_(tau)
        self.frozen_policy.load_state_dict(self.policy.state_dict())
        self.opponent.load_state_dict(self.policy.state_dict())
        if self.hp["kl_bc_coef"] > 0:                    # the frozen clone the kickstarting penalty pulls towards
            self.teacher = FleetPolicy(self.dims, self.heads, commander=False, **self.arch).to(self.device).eval()
            self.teacher.captain.load_state_dict(ck["captain"])
            self.steps_since_teacher = 0
        if start_epsilon is not None:
            self.hp["exploration_initial_eps"] = min(self.hp["exploration_initial_eps"], start_epsilon)
        if self.verbose:
            print(f"Q-network starts from imitation ({os.path.basename(path)}); exploration "
                  f"{self.hp['exploration_initial_eps']} -> {self.hp['exploration_final_eps']} over "
                  f"{self.hp['exploration_timesteps']:,} timesteps"
                  + (f"; kickstarting {self.hp['kl_bc_coef']} -> {self.hp['kl_bc_final']}" if self.teacher else ""))

    # ------------------------------------------------------------------ logging

    def _log(self, train_stats: dict, start: int):
        eps, self._episodes = self._episodes, []
        elapsed = max(1e-6, time.time() - self._t0)
        row = {"total_timesteps": self.num_timesteps, "updates": self.n_updates, "gradient_steps": self.n_gradient_steps,
               "fps": int((self.num_timesteps - start) / elapsed), "time_elapsed": int(elapsed),
               "stage": getattr(self.env, "stage", 0), "episodes": len(eps)}
        if eps:
            row["ep_rew_mean"] = float(np.mean([e["episode"]["r"] for e in eps]))
            row["ep_len_mean"] = float(np.mean([e["episode"]["l"] for e in eps]))
            gate = [e for e in eps if e.get("opponent", "rule") in ("rule", "passive")]
            selfp = [e["won"] for e in eps if e.get("opponent") in ("latest", "past", "mirror", "champion", "rival")]
            if gate:
                row["win_rate"] = float(np.mean([e["won"] for e in gate]))
                row["pass_rate"] = float(np.mean([e.get("passed", e["won"]) for e in gate]))
            if selfp:
                row["win_rate_self_play"] = float(np.mean(selfp))
        row.update(train_stats)
        row["epsilon"] = round(self.exploration_rate, 4)
        row["rollout_decisions"] = self.rollout.decisions
        row["elo"] = round(self.league.ratings["learner"], 1)
        if self.verbose:
            print("-" * 44)
            for k, v in row.items():
                print(f"| {k:<22} | {v:>15.4g} |" if isinstance(v, float) else f"| {k:<22} | {v:>15} |")
            print("-" * 44)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, "progress.csv")
            new = not os.path.exists(path)
            if not new:
                with open(path) as f:
                    header = f.readline().strip().split(",")
                if header != PROGRESS_FIELDS:
                    os.replace(path, path.replace(".csv", "_old.csv"))
                    new = True
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)

    # ------------------------------------------------------------------ saving and loading

    def save(self, path: str) -> str:
        """Everything but the rollout (it is thrown away after every update anyway)."""
        path = path if path.endswith(".pt") else path + ".pt"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"method": "fleet-pqn", "policy": self.policy.state_dict(), "value_head": self.value_head.state_dict(),
                    "q_optimizer": self.q_optimizer.state_dict(),
                    "commander_optimizer": self.commander_optimizer.state_dict() if self.commander_optimizer else None,
                    "hyperparameters": self.hp, "dims": self.dims, "heads": self.heads, "head_names": self.head_names,
                    "num_timesteps": self.num_timesteps, "n_updates": self.n_updates,
                    "n_gradient_steps": self.n_gradient_steps, "eps_start_timesteps": self._eps_start_timesteps,
                    "stage": getattr(self.env, "stage", 0), "league": self.league.state_dict(),
                    "teacher": self.teacher.state_dict() if self.teacher is not None else None,
                    "steps_since_teacher": self.steps_since_teacher,
                    # for exporting to the game (export.py)
                    "decision_period": getattr(self.env, "decision_period", 1.0),
                    "enemy_privileged": list(getattr(getattr(self.env, "spec", None), "enemy_privileged", (1, 45))),
                    "match_hidden": [float(1.0 - k) for k in getattr(self.env, "_fleet_match_keep", [])] or None}, path)
        return path

    @classmethod
    def load(cls, path: str, env, device: str = "auto", **overrides) -> "FleetPQN":
        """A saved model, ready to predict or to continue learning in env (moved to the model's stage)."""
        path = path if path.endswith(".pt") else path + ".pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("method") != "fleet-pqn":
            raise ValueError(f"{path} is not a PQN checkpoint (method {ck.get('method')!r}) - "
                             f"use MAPPO.load for a MAPPO one")
        if ck["dims"] != dict(env.dims) or not same_heads(ck["heads"], env.heads):
            raise ValueError(f"{path} was trained on a different observation layout: {ck['dims']} vs {dict(env.dims)}")
        model = cls(env, device=device, **{"verbose": 0, **ck["hyperparameters"], **overrides})
        model.policy.load_state_dict(ck["policy"])
        model.value_head.load_state_dict(ck["value_head"])
        model.frozen_policy.load_state_dict(ck["policy"])
        model.frozen_value_head.load_state_dict(ck["value_head"])
        model.opponent.load_state_dict(ck["policy"])
        model.q_optimizer.load_state_dict(ck["q_optimizer"])
        if model.commander_optimizer is not None and ck.get("commander_optimizer"):
            model.commander_optimizer.load_state_dict(ck["commander_optimizer"])
        model.league.load_state_dict(ck["league"])
        if ck.get("teacher") is not None:
            model.teacher = FleetPolicy(model.dims, model.heads, commander=False, **model.arch).to(model.device).eval()
            model.teacher.load_state_dict(ck["teacher"])
            model.steps_since_teacher = ck.get("steps_since_teacher", 0)
        model.num_timesteps, model.n_updates = ck["num_timesteps"], ck["n_updates"]
        model.n_gradient_steps = ck.get("n_gradient_steps", 0)
        model._eps_start_timesteps = ck.get("eps_start_timesteps", 0)
        if hasattr(env, "set_stage"):
            env.set_stage(ck.get("stage", 0))
        return model
