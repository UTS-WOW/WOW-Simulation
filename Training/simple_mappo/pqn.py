"""PQN - the Parallelised Q-Network: the value-based method compared with MAPPO.

    Gallici, Fellows, Ellis, Pou, Masmitja, Foerster, Martin, "Simplifying Deep Temporal Difference
    Learning", ICLR 2025 (arXiv 2407.04811) - code: github.com/mttga/purejaxql

Deep Q-learning as DQN made it needs a large replay buffer and a target network to stay stable. PQN
shows that with normalisation in the Q-network (LayerNorm) neither is needed, and replaces them with
what an on-policy method like PPO does: many battles in parallel, a short rollout from each, and a few
epochs of minibatch updates on it, after which the rollout is thrown away. On SMAX (StarCraft
multi-agent micro-management) PQN with value decomposition beat both QMIX and MAPPO.

Why it fits this game better than the Double DQN it replaces (ddqn.py, still here for its old runs):

* no replay buffer - under self-play the opponents keep changing, and a buffer of past battles goes
  stale; every PQN update learns from the battles just played;
* Q(lambda) targets instead of 5-step returns - a decision here is one second of a 6-15 minute battle,
  and lambda-returns carry the result back over many more decisions:

      R_t = r_t + gamma * [ lambda * R_t+1 + (1 - lambda) * max_a Q(s_t+1, a) ]      (0 at a battle's end)

  computed with the network as it was when the rollout was collected (frozen for the update's epochs);
* VDN [Sunehag et al. 2018] - PQN-VDN, the paper's multi-agent version: the fleet's Q is the sum of
  its ships' Q-values, trained on the fleet's reward;
* about 5x less network work per timestep than the Double DQN (4 epochs over each decision, against
  4 gradient steps on 32 replayed chunks every 32 decisions), so it trains faster.

Everything else is shared with the Double DQN (and so with MAPPO): the captain network - which already
has LayerNorm after its encoder and in every attention block, PQN's stabiliser - with its dueling,
branching Q-heads, the GRU memory (trained on 32-decision chunks from each ship's memory at their
start, R2D2-style), the fleet commander (one-step Q-learning on its orders), the start from the Elite
clone with a fading kickstarting penalty, the league and self-play, the checkpoints and the export to
the game. Exploration is the Double DQN's: Boltzmann (softmax over Q) or epsilon-greedy.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .callbacks import BaseCallback, CallbackList
from .ddqn import FleetDDQN, SequenceReplay, _linear


class FleetPQN(FleetDDQN):
    ALGORITHM = "pqn"
    NAME = "PQN"

    def __init__(self, env, learning_rate: float = 2.5e-4, num_steps: int = 128, epochs: int = 4,
                 minibatches: int = 4, lam: float = 0.85, gamma: float = 0.99, max_grad_norm: float = 1.0,
                 mixer: str = "vdn", exploration_initial_eps: float = 1.0, exploration_final_eps: float = 0.01,
                 exploration_timesteps: int = 300_000, kl_bc_steps: int = 5_000, opponent_refresh: int = 20,
                 snapshot_every: int = 50, chunk_len: int = 32, **kwargs):
        """
        num_steps     decisions per battle in a rollout (a multiple of chunk_len); a rollout of
                      num_steps x battles decisions is learned from, then thrown away
        epochs, minibatches   passes over the rollout and minibatches per pass [PQN: 4 epochs]
        lam           lambda of the Q(lambda) targets [PQN on SMAX: 0.85]
        gamma, learning_rate, max_grad_norm   as in the paper's SMAX setting (0.99, 2.5e-4, 1.0)
        mixer         "vdn" (PQN-VDN: the fleet's Q is its ships' sum), "none" (every ship on its own
                      reward) or "qmix"
        opponent_refresh, snapshot_every   counted in updates (one rollout each)
        Every other argument is the Double DQN's (ddqn.py) - exploration, the networks, the commander,
        the kickstarting penalty (kl_bc_steps counts gradient steps), the league.
        """
        if num_steps % chunk_len:
            raise ValueError(f"num_steps ({num_steps}) must be a multiple of chunk_len ({chunk_len})")
        kwargs.pop("verbose_name", None)
        for unused in ("buffer_size", "learning_starts", "batch_size", "gradient_steps", "n_step",
                       "target_update_interval"):        # Double DQN settings PQN does not use
            kwargs.pop(unused, None)
        super().__init__(env, learning_rate=learning_rate, gamma=gamma, max_grad_norm=max_grad_norm, mixer=mixer,
                         exploration_initial_eps=exploration_initial_eps,
                         exploration_final_eps=exploration_final_eps, exploration_timesteps=exploration_timesteps,
                         kl_bc_steps=kl_bc_steps, opponent_refresh=opponent_refresh, snapshot_every=snapshot_every,
                         chunk_len=chunk_len, buffer_size=num_steps, learning_starts=0, **kwargs)
        for unused in ("buffer_size", "learning_starts", "batch_size", "gradient_steps", "n_step",
                       "target_update_interval"):
            self.hp.pop(unused, None)
        self.hp.update(num_steps=num_steps, epochs=epochs, minibatches=minibatches, lam=lam)
        self._rng = np.random.default_rng(self.hp["seed"])

    # ------------------------------------------------------------------ training loop

    def learn(self, total_timesteps: int, callback: BaseCallback | list | None = None, log_interval: int = 2):
        """Alternate between a rollout of num_steps decisions in every battle and `epochs` passes of
        minibatch updates on it [PQN]."""
        callback = CallbackList(callback if isinstance(callback, list) else [callback] if callback else [])
        callback.init_callback(self)
        if hasattr(self.env, "set_opponent_policy"):
            self.env.set_opponent_policy(self._opponent_act, self.league.sample)
        self._last_obs = self.env.reset()
        self._t0 = time.time()
        start = self._start_timesteps = self.num_timesteps
        self._total_timesteps = total_timesteps
        hp = self.hp
        stats = defaultdict(list)
        while self.num_timesteps < total_timesteps:
            # the rollout: num_steps decisions in every battle, as chunks of chunk_len (this rollout only)
            self.replay = SequenceReplay(10**9, hp["chunk_len"], hp["seed"])
            stopped = False
            for _ in range(hp["num_steps"] // hp["chunk_len"]):
                if not self.collect_chunks(callback):
                    stopped = True
                    break
            if len(self.replay):
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

    def _update(self, stats: dict) -> None:
        """`epochs` passes over the rollout in `minibatches` minibatches. The targets come from the
        network as it was at the end of the rollout (copied into target_policy, frozen meanwhile)."""
        hp = self.hp
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value_head.load_state_dict(self.value_head.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        n = len(self.replay)
        for _ in range(hp["epochs"]):
            for idx in np.array_split(self._rng.permutation(n), min(hp["minibatches"], n)):
                raw = self.replay.stack(idx)
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
        hp = self.hp
        T1, B, N = b["alive"].shape
        live = b["alive"] * b["exists"]
        view = self._fleet_view(b, T1, B)
        H = len(self.head_sizes)
        self.policy.train()
        q_heads, adv = self._ship_q(self.policy.captain, self.value_head, b, T1, B, N)
        acts = b["actions"].reshape(T1, B * N, H)
        taken = torch.stack([q.gather(-1, acts[..., d:d + 1]).squeeze(-1) for d, q in enumerate(q_heads)], -1)
        q_fleet = self._mix(self.mixer, taken.view(T1, B, N, H), live, view)            # [T1, B, (N,) heads]
        with torch.no_grad():
            tq, _ = self._ship_q(self.target_policy.captain, self.target_value_head, b, T1, B, N)
            best = torch.stack([q.max(-1).values for q in tq], -1).view(T1, B, N, H)   # max_a Q per head
            next_fleet = self._mix(self.target_mixer, best, live, view).mean(-1)      # the heads' mean [BDQ]
            if hp["mixer"] == "none":
                rewards = b["rewards"]
            else:                                            # the fleet's reward: its ships' average
                exists = b["exists"]
                rewards = (b["rewards"] * exists).sum(-1) / exists.sum(-1).clamp_min(1.0)
            y = self._lambda_targets(rewards, b["dones"], next_fleet)
        valid = live[:-1] if hp["mixer"] == "none" else (b["exists"][:-1].sum(-1) > 0).float()
        td = 0.5 * (q_fleet[:-1] - y.unsqueeze(-1)).pow(2).mean(-1)                   # [PQN: squared error]
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
        torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
        self.q_optimizer.step()
        stats["q_value"].append(float((q_fleet[:-1].detach().mean(-1) * valid).sum() / valid.sum().clamp_min(1.0)))
        stats["target_value"].append(float((y * valid).sum() / valid.sum().clamp_min(1.0)))
        if self.policy.commander is not None and b["commander_on"].sum() > 0:
            self._commander_step(b, view, T1, B, N, stats)
        self.n_gradient_steps += 1
        self.steps_since_teacher += 1
