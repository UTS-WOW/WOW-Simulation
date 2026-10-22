"""Double DQN for the fleet - the value-based line of game-playing RL, to compare with MAPPO.

References (each idea is marked where it is used below):

    [DQN]     Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
    [DDQN]    van Hasselt, Guez & Silver, "Deep Reinforcement Learning with Double Q-learning", AAAI 2016
    [Duel]    Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016
    [BDQ]     Tavakoli, Pardo & Kormushev, "Action Branching Architectures for Deep RL", AAAI 2018
    [R2D2]    Kapturowski et al., "Recurrent Experience Replay in Distributed RL", ICLR 2019
    [QMIX]    Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL", ICML 2018
    [pymarl2] Hu et al., "Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative
              Multi-Agent Reinforcement Learning" (QMIX tuned for StarCraft, SMAC), 2021/2023
    [h-DQN]   Kulkarni et al., "Hierarchical Deep RL: Integrating Temporal Abstraction and Intrinsic
              Motivation", NeurIPS 2016
    [Options] Sutton, Precup & Singh, "Between MDPs and semi-MDPs", Artificial Intelligence 1999
    [Kick]    Schmitt et al., "Kickstarting Deep Reinforcement Learning", 2018

Two families of RL won the game benchmarks: policy gradients (PPO - OpenAI Five, Honor of Kings;
our MAPPO) and value-based Q-learning (DQN - Atari; QMIX - the StarCraft multi-agent challenge,
where pymarl2 found a well-tuned QMIX with Double Q-learning still the one to beat). This file is
the second family, built from the SAME captain, commander, environment, rewards and curriculum as
MAPPO, so the two can be compared stage by stage.

It is the DQN of Week 9 - a Q-network, a frozen target network, a replay buffer, epsilon-greedy
exploration - with these additions:

* Double DQN [DDQN]: the ONLINE network picks the next action, the TARGET network scores it.
  Plain DQN uses the target network for both and overestimates (it takes the max of noisy values).

      y = r + gamma * Q_target(s', argmax_a Q_online(s', a))

* Dueling, branching heads [Duel, BDQ]: a ship chooses on six heads at once (move, speed, target,
  fire, torpedo, ability). One Q-value per complete action would be 20 x 5 x 10 x 2 x 2 x 15 =
  60,000 outputs; instead every head d has its own advantages, sharing one state value:

      Q_d(s, a_d) = V(s) + A_d(s, a_d) - mean of A_d over the legal options

  The advantages are exactly the captain network's outputs (networks.py: pointer heads for the
  target and the circle included), so the Q-network IS the captain, plus a value stream V. Every
  head is trained towards the same target, built from the average over the heads of the next
  state's values [BDQ] - so each head gets its own full-sized learning signal:

      y = r + gamma * mean over d of Q_target,d(s', argmax Q_online,d(s', .))
* Memory and sequence replay [R2D2]: the captain keeps its GRU. The replay buffer stores chunks of
  `chunk_len` consecutive decisions with the memory the ship had at the start of the chunk, and
  training replays the chunk through the GRU. Targets use n-step returns (R2D2 uses n = 5).
* Many ships, one network: by default every ship learns from ITS OWN reward (mixer="none":
  independent Double DQN with shared weights) - the same per-ship reward MAPPO learns from (rewards.py:
  its own terms, the fleet's terms and team spirit), so the two methods are compared on equal terms.
  mixer="qmix" is the StarCraft (SMAC) way [QMIX, pymarl2]: the fleet learns ONE value from the
  fleet's average reward, Q_fleet = mix(Q_ship_1 ... Q_ship_N, true battle), for every head. The
  mixing weights come from small "hypernetworks" that read the true battle (training only - like
  MAPPO's centralised critic) and are kept positive, so Q_fleet only rises when a ship's Q rises:
  every ship taking its own best action is then also the fleet's best joint action. mixer="vdn"
  just adds the ships' Q-values up. Measured on the mock task (rewards per ship): independent 0.28,
  VDN 0.08 after 6,000 timesteps with 3 ships - mixing must first work out which ship earned the
  fleet's reward, so it pays off only when the reward is truly shared (team spirit 1.0, stage 8).
* The commander as a meta-controller [h-DQN]: its scores for (ship, order) are Q-values, the fleet's
  value of a set of orders is their sum, and it is trained with intra-option Q-learning [Options]:
  at every decision, an order still in force is worth Q(s', same order); at the commander's next
  decision point it is worth the best order (Double-DQN style again).
* Exploration: epsilon-greedy, epsilon falling linearly (Week 9's exploration_fraction), applied per
  ship decision (all six heads random at once). Starting from the imitation clone, the greedy policy
  is already the Elite AI's, so epsilon can start low.
* Imitation start + anchor [Kick]: the Q-network's captain starts as the clone of the Elite AI - its
  head scores, times a temperature tau = 0.1, become the advantages (the same best options, at the
  size of Q-values: the clone's scores reach +-10, a step's reward is about 1) - and a fading
  penalty, the KL from the clone's choices to softmax(advantages / tau) on the ships' own replayed
  decisions ("kickstarting"), keeps it close while the Q-values are still meaningless. Without it the
  first TD updates erase the clone (measured on gunnery: 0.33 pass rate after 2,400 timesteps,
  against MAPPO's 0.78 with its KL anchor) - the same idea as MAPPO's KL anchor, so both methods start
  from the clone on equal terms.
* Self-play and the league: the same opponents as MAPPO (rule AI, mirror battles - both fleets are
  this network and both sides' experience goes into the replay buffer - and past copies).

The interface follows Stable-Baselines3's DQN, so it is used like MAPPO:

    model = FleetDDQN(env, learning_rate=2.5e-4, buffer_size=40_000, batch_size=32)
    model.learn(total_timesteps=200_000, callback=[...])
    model.save("models/model_final")
    model = FleetDDQN.load("models/model_final", env=env)
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
from .networks import (CAPTAIN_KEYS, NEG_INF, Commander, EntityEncoder, FleetPolicy, Memory, kl_to, layer, pool,
                       same_heads, to_tensors, trim)

CRITIC_KEYS = ["critic_own", "own_mask", "critic_enemy", "critic_enemy_mask", "critic_zones", "critic_zone_mask",
               "critic_match"]
# what one chunk of the replay buffer keeps of every observation (the commander's view is rebuilt from
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
                   "target_value", "kl_bc", "commander_loss", "epsilon", "orders_free", "replay_decisions", "elo"]


def _linear(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, progress))


# ---------------------------------------------------------------------------------------------- the mixer

class QMixer(nn.Module):
    """QMIX: the ships' Q-values -> the fleet's Q-value, using the TRUE battle (training only).

        hidden  = elu( sum_i |w1(s, ship i)| * Q_i  +  b1(s) )
        Q_fleet = |w2(s)| . hidden + b2(s)

    w1, b1, w2, b2 are hypernetworks: small networks whose outputs are the mixing weights. The
    absolute values keep every weight positive, so Q_fleet is monotonic in every Q_i. The battle is
    encoded like the critic's (one token per ship, enemy and circle), and ship i's weights come from
    its own token next to the pooled battle - so any number of ships can be mixed.
    """

    def __init__(self, dims: dict, d: int = 64, attn_heads: int = 4, layers: int = 1, embed: int = 32):
        super().__init__()
        self.encoder = EntityEncoder({"match": dims["critic_match"], "own": dims["critic_own"],
                                      "enemy": dims["critic_enemy"], "zone": dims["critic_zone"]}, d, attn_heads, layers)
        self.hyper_w1 = nn.Sequential(layer(3 * d, d), nn.ReLU(), layer(d, embed))
        self.hyper_b1 = layer(2 * d, embed)
        self.hyper_w2 = nn.Sequential(layer(2 * d, d), nn.ReLU(), layer(d, embed))
        self.hyper_b2 = nn.Sequential(layer(2 * d, d), nn.ReLU(), layer(d, 1))

    def forward(self, q: torch.Tensor, obs: dict) -> torch.Tensor:
        """q: every ship slot's Q-value [B, N, D] (0 for sunk and empty slots) - D separate sets of Q-values
        mixed with the same weights (one per action head); obs: the battle [B, ...] -> [B, D]."""
        B = q.shape[0]
        own, own_mask = trim(obs["critic_own"], obs["own_mask"])
        enemy, enemy_mask = trim(obs["critic_enemy"], obs["critic_enemy_mask"])
        zones, zone_mask = trim(obs["critic_zones"], obs["critic_zone_mask"])
        n = own.shape[1]
        groups = [("match", obs["critic_match"].unsqueeze(1), own.new_ones(B, 1)), ("own", own, own_mask),
                  ("enemy", enemy, enemy_mask), ("zone", zones, zone_mask)]
        tokens, mask = self.encoder(groups)
        battle = pool(tokens, mask)                                                        # [B, 2d]
        w1 = self.hyper_w1(torch.cat([tokens[:, 1:1 + n], battle.unsqueeze(1).expand(B, n, -1)], -1)).abs()  # [B, n, e]
        hidden = F.elu(torch.einsum("bnd,bne->bde", q[:, :n], w1) + self.hyper_b1(battle).unsqueeze(1))      # [B, D, e]
        return (hidden * self.hyper_w2(battle).abs().unsqueeze(1)).sum(-1) + self.hyper_b2(battle)            # [B, D]


# ---------------------------------------------------------------------------------------------- replay buffer

class SequenceReplay:
    """R2D2's replay buffer: chunks of consecutive decisions of one fleet (one row of the environment),
    each with the captains' memory at its start. A chunk holds chunk_len + 1 observations: the last
    one is only there to bootstrap from (it is also the first of the next chunk).

    Observations are kept as float16 and 0/1 masks as bytes, and only the ship slots the battle
    uses, so a 1v1 chunk is small. `capacity` counts decisions (chunks x chunk_len), like SB3's
    buffer_size; the oldest chunks go first.
    """

    def __init__(self, capacity: int, chunk_len: int, seed: int = 0):
        self.max_chunks = max(1, capacity // chunk_len)
        self.chunk_len = chunk_len
        self.chunks: list[dict] = []
        self.next = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def decisions(self) -> int:
        return len(self.chunks) * self.chunk_len

    def nbytes(self) -> int:
        return sum(v.nbytes for c in self.chunks for v in c.values())

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
        if len(self.chunks) < self.max_chunks:
            self.chunks.append(kept)
        else:
            self.chunks[self.next] = kept
            self.next = (self.next + 1) % self.max_chunks

    def sample(self, batch_size: int) -> dict:
        """batch_size chunks, padded to the same number of ship slots -> arrays [T + 1, B, ...] (h0: [B, N, H])."""
        return self.stack(self.rng.integers(len(self.chunks), size=batch_size))

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


# ---------------------------------------------------------------------------------------------- Double DQN

class FleetDDQN(LeagueMixin):
    GREEDY = True                       # its own copies in the league take their best action
    ALGORITHM, NAME = "ddqn", "Double DQN"   # pqn.py's FleetPQN shares everything but the training loop

    def __init__(self, env, learning_rate: float = 2.5e-4, commander_learning_rate: float = 1e-4,
                 buffer_size: int = 40_000, learning_starts: int = 4_000, batch_size: int = 32,
                 max_batch_ship_steps: int = 4096, chunk_len: int = 32, gradient_steps: int = 4,
                 gamma: float = 0.99, n_step: int = 5, target_update_interval: int = 250,
                 exploration_initial_eps: float = 1.0, exploration_final_eps: float = 0.05,
                 exploration_timesteps: int = 300_000, commander_eps_scale: float = 0.5, eval_epsilon: float = 0.02,
                 opponent_epsilon: float = 0.02, max_grad_norm: float = 10.0, mixer: str = "none",
                 mixing_embed: int = 32, d_model: int = 64, attn_heads: int = 4, attn_layers: int = 2,
                 hidden_size: int = 128, recurrent: bool = True, commander: bool = True, commander_period: int = 10,
                 kl_bc_coef: float = 0.5, kl_bc_final: float = 0.05, kl_bc_steps: int = 20_000,
                 imitation_temperature: float = 0.1, exploration: str = "epsilon", temperature: float = 0.1,
                 opponent_refresh: int = 80, snapshot_every: int = 200, pool_size: int = 20, pfsp: str = "hard",
                 device: str = "auto",
                 torch_threads: int | None = 4, seed: int = 0, verbose: int = 1, log_dir: str | None = None):
        """
        learning_rate          step size of the Q-network (captain + value stream + mixer); the
                               commander has commander_learning_rate
        buffer_size            decisions the replay buffer keeps (of one fleet each); the oldest go first
        learning_starts        decisions in the buffer before the first gradient step
        batch_size             chunks per gradient step - at most max_batch_ship_steps ship-decisions,
                               so a 9v9 batch is not 9 times the work of a 1v1 one
        chunk_len              decisions per chunk: the replay unit, and the decisions collected from
                               every battle between two rounds of gradient steps
        gradient_steps         gradient steps after every chunk_len decisions per battle
        gamma, n_step          discount per decision and the n of the n-step return
        target_update_interval gradient steps between copies of the online network into the target one
        exploration_*          epsilon falls from initial to final over exploration_timesteps (counted
                               from when the model was made); commander_eps_scale x epsilon for orders
        eval_epsilon           epsilon of predict() (evaluations, replays, the game: 0.02 - a fully
                               greedy policy can get stuck repeating one move); opponent_epsilon for
                               the frozen copies in league battles
        max_grad_norm          gradient clipping (10, as in DQN / pymarl2)
        mixer                  "none" (default: every ship on its own reward - independent DQN with
                               shared weights), "qmix" (one fleet value from the fleet's reward, mixed by
                               hypernetworks - QMIX) or "vdn" (the ships' Q-values added up)
        d_model ... commander_period   the captain and commander - the same as MAPPO's
        kl_bc_coef             after an imitation start: weight of the kickstarting penalty, fading to
                               kl_bc_final over kl_bc_steps gradient steps
        imitation_temperature  tau: the clone's scores x tau are the starting advantages
        exploration            how the captains choose while playing: "epsilon" (epsilon-greedy: the best
                               option, or a random one) or "softmax" (Boltzmann exploration: options in
                               proportion to exp(Q / temperature) - from the clone, with temperature =
                               imitation_temperature, it first plays the clone's own choices, sampled as
                               MAPPO does, instead of only its single likeliest one). The commander's
                               orders stay epsilon-greedy either way; deterministic play is greedy.
        opponent_refresh       updates (rounds of chunk_len decisions) between refreshing the "latest"
                               opponent; snapshot_every between adding a past copy to the league
        """
        if mixer not in ("qmix", "vdn", "none"):
            raise ValueError(f"mixer must be 'qmix', 'vdn' or 'none', not {mixer!r}")
        if exploration not in ("epsilon", "softmax"):
            raise ValueError(f"exploration must be 'epsilon' or 'softmax', not {exploration!r}")
        self.env = env
        self.hp = dict(learning_rate=learning_rate, commander_learning_rate=commander_learning_rate,
                       buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size,
                       max_batch_ship_steps=max_batch_ship_steps, chunk_len=chunk_len, gradient_steps=gradient_steps,
                       gamma=gamma, n_step=n_step, target_update_interval=target_update_interval,
                       exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps,
                       exploration_timesteps=exploration_timesteps, commander_eps_scale=commander_eps_scale,
                       eval_epsilon=eval_epsilon, opponent_epsilon=opponent_epsilon, max_grad_norm=max_grad_norm,
                       mixer=mixer, mixing_embed=mixing_embed, d_model=d_model, attn_heads=attn_heads,
                       attn_layers=attn_layers, hidden_size=hidden_size, recurrent=recurrent, commander=commander,
                       commander_period=commander_period, kl_bc_coef=kl_bc_coef, kl_bc_final=kl_bc_final,
                       kl_bc_steps=kl_bc_steps, imitation_temperature=imitation_temperature,
                       exploration=exploration, temperature=temperature, opponent_refresh=opponent_refresh,
                       snapshot_every=snapshot_every, pool_size=pool_size, pfsp=pfsp, seed=seed)
        self.verbose = verbose
        self.log_dir = log_dir
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if torch_threads:
            torch.set_num_threads(torch_threads)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.dims, self.heads = dict(env.dims), [dict(h) for h in env.heads]
        self.head_sizes, self.head_names = [h["size"] for h in self.heads], [h["name"] for h in self.heads]
        self.arch = dict(d=d_model, attn_heads=attn_heads, layers=attn_layers, hidden=hidden_size, recurrent=recurrent)
        # the online network: the captain (its head scores are the advantages A_d), the value stream V
        # and the commander (its order scores are Q-values) - and the mixer
        self.policy = FleetPolicy(self.dims, self.heads, commander=commander, commander_period=commander_period,
                                  **self.arch).to(self.device)
        self.actor = self.policy                                    # the name the Week 9/10 notebooks print
        self.value_head = layer(hidden_size + d_model, 1, gain=1.0).to(self.device)
        self.mixer = QMixer(self.dims, d_model, attn_heads, 1, mixing_embed).to(self.device) if mixer == "qmix" else None
        # the target network: a frozen copy, refreshed every target_update_interval gradient steps [DQN]
        self.target_policy = copy.deepcopy(self.policy).eval()
        self.target_value_head = copy.deepcopy(self.value_head).eval()
        self.target_mixer = copy.deepcopy(self.mixer).eval() if self.mixer is not None else None
        q_params = list(self.policy.captain.parameters()) + list(self.value_head.parameters())
        if self.mixer is not None:
            q_params += list(self.mixer.parameters())
        self.q_optimizer = torch.optim.Adam(q_params, lr=learning_rate, eps=1e-5)
        self.commander_optimizer = (torch.optim.Adam(self.policy.commander.parameters(), lr=commander_learning_rate,
                                                     eps=1e-5) if commander else None)
        self.replay = SequenceReplay(buffer_size, chunk_len, seed)
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
            n_mix = sum(p.numel() for p in self.mixer.parameters()) if self.mixer is not None else 0
            print(f"{self.NAME} on {self.device}: Q-network (captain + value stream) {n_q:,} parameters"
                  + (f", commander {n_cmd:,}" if commander else "") + (f", QMIX mixer {n_mix:,}" if n_mix else
                                                                      f", mixer: {mixer}")
                  + (f"; replay {buffer_size:,} decisions" if self.ALGORITHM == "ddqn" else "; no replay buffer: rollouts")
                  + f", heads {dict(zip(self.head_names, self.head_sizes))}")

        self.num_timesteps = 0          # decisions, summed over the parallel battles (as MAPPO)
        self.n_updates = 0              # rounds of collecting chunk_len decisions + gradient steps
        self.n_gradient_steps = 0
        self._eps_start_timesteps = 0   # exploration is scheduled from here (the model's birth)
        self._last_obs = None
        self._episodes = []
        self._t0 = None
        self._start_timesteps, self._total_timesteps = 0, None
        self._memory = self._play_memory = self._opp_memory = self._teacher_memory = None
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

    # ------------------------------------------------------------------ training loop

    def learn(self, total_timesteps: int, callback: BaseCallback | list | None = None, log_interval: int = 8):
        """Alternate between playing chunk_len decisions in every battle (into the replay buffer) and
        gradient_steps updates on chunks sampled from it."""
        callback = CallbackList(callback if isinstance(callback, list) else [callback] if callback else [])
        callback.init_callback(self)
        if hasattr(self.env, "set_opponent_policy"):
            self.env.set_opponent_policy(self._opponent_act, self.league.sample)
        self._last_obs = self.env.reset()
        self._t0 = time.time()
        start = self._start_timesteps = self.num_timesteps
        self._total_timesteps = total_timesteps
        stats = defaultdict(list)
        while self.num_timesteps < total_timesteps:
            if not self.collect_chunks(callback):
                break
            if self.replay.decisions >= self.hp["learning_starts"]:
                for _ in range(self.hp["gradient_steps"]):
                    self.train_step(stats)
            self.n_updates += 1
            if self.n_updates % self.hp["opponent_refresh"] == 0:
                self.opponent.load_state_dict(self.policy.state_dict())
            self._league_step()                           # past copies, the champion (league.py)
            if log_interval and self.n_updates % log_interval == 0:
                self._log({k: float(np.mean(v)) for k, v in stats.items()}, start)
                stats = defaultdict(list)
            if not callback.on_rollout_end():
                break
        callback.on_training_end()
        return self

    def collect_chunks(self, callback) -> bool:
        """Plays chunk_len decisions in every battle with epsilon-greedy actions and puts one chunk per
        fleet (row) into the replay buffer."""
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
                self.replay.add(chunk)
        return True

    # ------------------------------------------------------------------ the update

    def _batch(self) -> dict:
        n_ships = np.mean([c["exists"].shape[1] for c in self.replay.chunks[-64:]]) if self.replay.chunks else 1
        steps = (self.hp["chunk_len"] + 1) * max(1.0, n_ships)
        size = int(max(4, min(self.hp["batch_size"], self.hp["max_batch_ship_steps"] // steps)))
        raw = self.replay.sample(size)
        return {k: torch.as_tensor(v.astype(np.int64) if k in ("actions", "orders") else v.astype(np.float32),
                                   device=self.device) for k, v in raw.items()}

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
        """The commander's and the mixer's inputs [T1 * B, ...]: the critic's arrays (+ the commander's
        view with the hidden columns zeroed)."""
        flat = {k: b[k].reshape(T1 * B, *b[k].shape[2:]) for k in CRITIC_KEYS if k != "own_mask"}
        flat["own_mask"] = b["exists"].reshape(T1 * B, -1)
        flat["alive"] = b["alive"].reshape(T1 * B, -1)
        flat["fleet_enemy"] = flat["critic_enemy"] * self._enemy_keep
        flat["fleet_match"] = flat["critic_match"] * self._match_keep
        return flat

    def _mix(self, mixer, q: torch.Tensor, live: torch.Tensor, view: dict) -> torch.Tensor:
        """Ships' Q per head [T1, B, N, D] -> the fleet's [T1, B, D] (per ship [T1, B, N, D] for mixer="none")."""
        q = q * live.unsqueeze(-1)
        kind = self.hp["mixer"]
        if kind == "none":
            return q
        if kind == "vdn":
            return q.sum(2)
        T1, B, N, D = q.shape
        return mixer(q.reshape(T1 * B, N, D), view).view(T1, B, D)

    def _n_step_targets(self, rewards: torch.Tensor, dones: torch.Tensor, next_q: torch.Tensor) -> torch.Tensor:
        """y_t = r_t + g r_t+1 + ... + g^(k-1) r_t+k-1 + g^k Q_next(t+k), k = min(n, chunk end - t);
        the sum stops at the end of a battle (no bootstrap after it). rewards / dones / next_q: [T1, B(, N)]."""
        g, n = self.hp["gamma"], self.hp["n_step"]
        L = rewards.shape[0] - 1
        ys = []
        for t in range(L):
            ret, disc, cont = torch.zeros_like(next_q[0]), 1.0, torch.ones_like(next_q[0])
            k = min(n, L - t)
            for j in range(k):
                d = dones[t + j]
                if d.dim() < ret.dim():                      # per-ship targets (mixer="none"): one flag per battle
                    d = d.unsqueeze(-1)
                ret = ret + cont * disc * rewards[t + j]
                cont = cont * (1.0 - d)
                disc *= g
            ys.append(ret + cont * disc * next_q[t + k])
        return torch.stack(ys)

    def train_step(self, stats: dict) -> None:
        """One gradient step of Double DQN on a batch of chunks (and one for the commander)."""
        hp = self.hp
        b = self._batch()
        T1, B, N = b["alive"].shape
        live = b["alive"] * b["exists"]
        view = self._fleet_view(b, T1, B)
        H = len(self.head_sizes)
        self.policy.train()

        # ---- Q of the actions the ships took, per head (online network)
        q_heads, adv = self._ship_q(self.policy.captain, self.value_head, b, T1, B, N)
        acts = b["actions"].reshape(T1, B * N, H)
        taken = torch.stack([q.gather(-1, acts[..., d:d + 1]).squeeze(-1) for d, q in enumerate(q_heads)], -1)
        q_fleet = self._mix(self.mixer, taken.view(T1, B, N, H), live, view)          # [T1, B, (N,) heads]

        # ---- the target: the online network picks the next actions, the target network scores them [DDQN],
        # averaged over the heads [BDQ]
        with torch.no_grad():
            best = [q.argmax(-1, keepdim=True) for q in q_heads]
            tq, _ = self._ship_q(self.target_policy.captain, self.target_value_head, b, T1, B, N)
            next_heads = torch.stack([q.gather(-1, a).squeeze(-1) for q, a in zip(tq, best)], -1).view(T1, B, N, H)
            next_fleet = self._mix(self.target_mixer, next_heads, live, view).mean(-1)
            exists = b["exists"]
            if hp["mixer"] == "none":
                rewards = b["rewards"]
            else:                                                    # the fleet's reward: its ships' average
                rewards = (b["rewards"] * exists).sum(-1) / exists.sum(-1).clamp_min(1.0)
            y = self._n_step_targets(rewards, b["dones"], next_fleet)
        valid = (live[:-1] if hp["mixer"] == "none" else (exists[:-1].sum(-1) > 0).float())
        td = F.smooth_l1_loss(q_fleet[:-1], y.unsqueeze(-1).expand_as(q_fleet[:-1]), reduction="none").mean(-1)
        loss = (td * valid).sum() / valid.sum().clamp_min(1.0)
        kl_coef = _linear(hp["kl_bc_coef"], hp["kl_bc_final"], self.steps_since_teacher / max(1, hp["kl_bc_steps"]))
        kl = self._kickstart(b, adv, live, T1, B, N) if kl_coef > 0 else None
        if kl is not None:
            stats["q_loss"].append(loss.item())
            stats["kl_bc"].append(kl.item())
            loss = loss + kl_coef * kl
        else:
            stats["q_loss"].append(loss.item())
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
        if self.n_gradient_steps % hp["target_update_interval"] == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
            self.target_value_head.load_state_dict(self.value_head.state_dict())
            if self.mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _commander_step(self, b: dict, view: dict, T1: int, B: int, N: int, stats: dict) -> None:
        """Intra-option Q-learning for the commander [h-DQN, Options], at every decision of the chunk:

            Q_fleet(s, orders) = sum over live ships of q(s, ship, its order)
            target = r + gamma * ( Q_target(s', the same orders)          if the orders stay in force
                                   Q_target(s', online's best orders)     at the commander's next decision )
        """
        g = self.hp["gamma"]
        own_live = (view["own_mask"] * view["alive"]).view(T1, B, -1)
        mask = Commander.order_mask(view)
        q = self.policy.commander(view).masked_fill(mask < 0.5, NEG_INF).view(T1, B, N, -1)
        orders = b["orders"]
        q_now = (q.gather(-1, orders.unsqueeze(-1)).squeeze(-1).masked_fill(own_live < 0.5, 0.0) * own_live).sum(-1)
        with torch.no_grad():
            tq = self.target_policy.commander(view).masked_fill(mask < 0.5, NEG_INF).view(T1, B, N, -1)
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
        the advantages, so the greedy policy starts as the Elite AI's. Exploration then starts at
        start_epsilon instead of 1.0 (random play would only throw that away)."""
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
        self.target_policy.load_state_dict(self.policy.state_dict())
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
        row["replay_decisions"] = self.replay.decisions
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
        """Everything but the replay buffer (it would be gigabytes; a continued run refills it for
        learning_starts decisions before its first gradient step - SB3 does not save it either)."""
        path = path if path.endswith(".pt") else path + ".pt"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"method": "fleet-ddqn", "algorithm": self.ALGORITHM,
                    "policy": self.policy.state_dict(), "value_head": self.value_head.state_dict(),
                    "mixer": self.mixer.state_dict() if self.mixer is not None else None,
                    "target_policy": self.target_policy.state_dict(),
                    "target_value_head": self.target_value_head.state_dict(),
                    "target_mixer": self.target_mixer.state_dict() if self.target_mixer is not None else None,
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
    def load(cls, path: str, env, device: str = "auto", **overrides) -> "FleetDDQN":
        """A saved model, ready to predict or to continue learning in env (moved to the model's stage)."""
        path = path if path.endswith(".pt") else path + ".pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if cls is FleetDDQN and ck.get("algorithm") == "pqn":     # a PQN checkpoint: the same network
            from .pqn import FleetPQN
            return FleetPQN.load(path, env, device, **overrides)
        if ck.get("method") != "fleet-ddqn":
            raise ValueError(f"{path} is not a Double DQN checkpoint (method {ck.get('method')!r}) - "
                             f"use MAPPO.load for a MAPPO one")
        if ck["dims"] != dict(env.dims) or not same_heads(ck["heads"], env.heads):
            raise ValueError(f"{path} was trained on a different observation layout: {ck['dims']} vs {dict(env.dims)}")
        model = cls(env, device=device, **{"verbose": 0, **ck["hyperparameters"], **overrides})
        model.policy.load_state_dict(ck["policy"])
        model.value_head.load_state_dict(ck["value_head"])
        if model.mixer is not None and ck.get("mixer"):
            model.mixer.load_state_dict(ck["mixer"])
        model.target_policy.load_state_dict(ck["target_policy"])
        model.target_value_head.load_state_dict(ck["target_value_head"])
        if model.target_mixer is not None and ck.get("target_mixer"):
            model.target_mixer.load_state_dict(ck["target_mixer"])
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
