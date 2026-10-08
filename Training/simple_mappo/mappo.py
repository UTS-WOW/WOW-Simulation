"""MAPPO with a fleet commander: PPO for a team of ships, trained the way game-playing AIs are.

References (the method is a combination of these; each idea is marked where it is used below):

    [MAPPO]  Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022
    [Five]   OpenAI, "Dota 2 with Large Scale Deep Reinforcement Learning" (OpenAI Five), 2019
    [AS]     Vinyals et al., "Grandmaster level in StarCraft II using multi-agent RL" (AlphaStar), Nature 2019
    [HoK]    Ye et al., "Towards Playing Full MOBA Games with Deep RL" (Honor of Kings), NeurIPS 2020
    [HMS]    Wu et al., "Hierarchical Macro Strategy Model for MOBA Game AI", AAAI 2019
    [TiZero] Lin et al., "TiZero: Mastering Multi-Agent Football with Curriculum Learning and Self-Play", AAMAS 2023

It is still the PPO of Week 10 - collect experience, compute advantages, a few epochs of clipped
updates - with these additions:

* Captains and a commander (networks.py). Every ship runs the same captain network on its own view
  (decentralised execution, parameter sharing [MAPPO]). Every `commander_period` decisions the
  commander gives each ship an order (free / engage / hold circle k) [HMS]. Both are trained with
  PPO; the commander's "step" lasts commander_period decisions, and its ratio is the product over
  the fleet's ships (one joint decision, like TiZero's joint-ratio policy optimisation).
* Centralised critic [MAPPO]: during training it sees the true battle. It predicts each reward
  group separately (multi-head value [HoK]); the advantages of the groups are added up.
* Memory [Five]: the captain has a GRU. Training replays the rollout in chunks of `chunk_len`
  decisions, starting each chunk from the memory the ship had at that point (recurrent MAPPO).
* Dual-clip PPO [HoK]: besides PPO's clip, a negative advantage can never push the objective below
  dual_clip x advantage - one very stale sample cannot blow up an update.
* Value normalisation and value clipping [MAPPO], entropy bonus that decays over training.
* Imitation warm start + KL anchor [AS]: the captain can start from a copy of the rule AI
  (imitation.py) and is pulled back towards it by a KL penalty that fades out.
* League self-play [Five, AS]: the enemy is the rule AI, a frozen copy of the current fleet, or a
  past copy picked by PFSP (league.py) - per curriculum stage.
* Death masking [MAPPO]: a sunk ship stops training the actor; its critic keeps learning the team's
  value until the battle ends, because team rewards keep arriving.

The interface follows Stable-Baselines3, so the notebooks read like the Week 9 and 10 ones:

    model = MAPPO(env, learning_rate=3e-4, n_steps=256, n_epochs=5)
    model.learn(total_timesteps=200_000, callback=SaveOnIntervalCallback(...))
    model.save("models/model_final")
    model = MAPPO.load("models/model_final", env=env)
    actions = model.predict(obs)
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
from .league import OpponentPool
from .networks import (CAPTAIN_KEYS, NEG_INF, CentralCritic, Commander, FleetPolicy, LocalCritic, Memory,
                       kl_to, log_prob_entropy, to_tensors)
from .rewards import REWARD_GROUPS

CRITIC_KEYS = ["critic_own", "own_mask", "critic_enemy", "critic_enemy_mask", "critic_zones", "critic_zone_mask",
               "critic_match"]
COMMANDER_KEYS = CRITIC_KEYS[:-1] + ["fleet_enemy", "fleet_match", "alive"]
PROGRESS_FIELDS = ["total_timesteps", "updates", "fps", "time_elapsed", "stage", "episodes", "ep_rew_mean",
                   "ep_len_mean", "win_rate", "pass_rate", "win_rate_self_play", "policy_loss", "value_loss", "entropy",
                   "approx_kl", "clip_fraction", "explained_variance", "commander_loss", "commander_entropy",
                   "orders_free", "kl_bc", "win_prob_loss", "elo"]


class ValueNorm:
    """Running mean and standard deviation of the value targets, one per value head (MAPPO's value
    normalisation). The critic learns targets of about unit size; its output is turned back into
    game units for the advantages. beta = 0.99999 as in the MAPPO code (a fast 0.99 moved the
    targets every update and learning was noticeably slower)."""

    def __init__(self, shape=(), beta: float = 0.99999):
        self.beta = beta
        self.mean, self.mean_sq, self.debias = np.zeros(shape), np.zeros(shape), 0.0

    def update(self, x: np.ndarray) -> None:
        """x [samples, *shape]"""
        self.mean = self.beta * self.mean + (1 - self.beta) * x.mean(0)
        self.mean_sq = self.beta * self.mean_sq + (1 - self.beta) * (x ** 2).mean(0)
        self.debias = self.beta * self.debias + (1 - self.beta)

    def stats(self):
        if self.debias == 0.0:
            return np.zeros_like(self.mean), np.ones_like(self.mean)
        mean, mean_sq = self.mean / self.debias, self.mean_sq / self.debias
        return mean, np.sqrt(np.maximum(mean_sq - mean ** 2, 1e-2))

    def normalize(self, x):
        mean, std = self.stats()
        return (x - mean) / std

    def denormalize(self, x):
        mean, std = self.stats()
        return x * std + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "mean_sq": self.mean_sq, "debias": self.debias}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.mean_sq, self.debias = np.asarray(d["mean"]), np.asarray(d["mean_sq"]), d["debias"]


def _linear(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(1.0, max(0.0, progress))


def ppo_objective(ratio: torch.Tensor, advantage: torch.Tensor, clip_range: float, dual_clip: float | None):
    """PPO's clipped surrogate (Week 10), per sample - to be maximised:

        min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)

    plus Honor of Kings' dual clip: when A < 0 a huge ratio could make this arbitrarily negative,
    so it is bounded below by dual_clip * A."""
    surrogate = torch.min(ratio * advantage, ratio.clamp(1 - clip_range, 1 + clip_range) * advantage)
    if dual_clip:
        surrogate = torch.where(advantage < 0, torch.max(surrogate, dual_clip * advantage), surrogate)
    return surrogate


class MAPPO:
    def __init__(self, env, learning_rate: float = 3e-4, critic_learning_rate: float = 5e-4,
                 commander_learning_rate: float = 1e-4, n_steps: int = 256, n_minibatches: int = 4,
                 n_epochs: int = 5, gamma: float = 0.995, gae_lambda: float = 0.95, clip_range: float = 0.2,
                 dual_clip: float = 3.0, value_clip: float = 0.2, ent_coef: float | str = 0.01,
                 ent_coef_final: float = 0.002, ent_decay_updates: int = 1500, target_kl: float | None = None,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5, d_model: int = 64, attn_heads: int = 4,
                 attn_layers: int = 2, hidden_size: int = 128, recurrent: bool = True, chunk_len: int = 32,
                 centralised_critic: bool = True, normalize_values: bool = True, commander: bool = True,
                 commander_period: int = 10, kl_bc_coef: float = 0.1, kl_bc_final: float = 0.01,
                 kl_bc_updates: int = 1000, actor_warmup_updates: int = 0, opponent_refresh: int = 10,
                 snapshot_every: int = 25, pool_size: int = 20, device: str = "auto", torch_threads: int | None = 4,
                 seed: int = 0, verbose: int = 1, log_dir: str | None = None):
        """
        learning_rate        the captains' (actors') step size; critic_learning_rate and
                             commander_learning_rate for the critic and the commander
        n_steps              decisions collected from every battle before each update (a decision
                             covers env.decision_period seconds of game time)
        n_minibatches        the collected data is split into this many minibatches per epoch. A fixed
                             count (rather than a fixed size) gives every curriculum stage the same
                             number of gradient steps per update, whether 1 or 8 ships collected it
        n_epochs             passes over the collected data per update
        gamma                discount factor per decision (0.995 looks ~200 s ahead: battles are long)
        gae_lambda           bias/variance trade-off of the advantage estimate
        clip_range           how far one update may move the policy (PPO's clip)
        dual_clip            Honor of Kings' second clip for negative advantages (None: off)
        value_clip           how far one update may move the critic's prediction (None: off)
        ent_coef             bonus for keeping the policy random enough to explore; decays linearly to
                             ent_coef_final over ent_decay_updates. "auto": tuned automatically to keep
                             the entropy near a target, like SAC's ent_coef='auto' (Week 10)
        target_kl            stop the epochs early when the policy moved further than this (SB3's target_kl)
        d_model, attn_heads, attn_layers   the entity encoder (attn_layers=0: no attention, pooling only)
        hidden_size          the captain's memory (GRU) size
        recurrent            False: no memory (a plain layer instead of the GRU)
        chunk_len            decisions per training chunk of the recurrent policy
        centralised_critic   True: MAPPO (the critic sees the whole battle). False: IPPO (each ship's
                             critic sees only its own view) - for comparing the two
        normalize_values     MAPPO's value normalisation (ValueNorm)
        commander            build the fleet commander (it gives orders in stages with "commander": True)
        commander_period     decisions between the commander's orders (10 = every 10 s of game time)
        kl_bc_coef           after an imitation warm start: weight of the KL penalty that keeps the
                             captain near the imitation policy, fading to kl_bc_final over kl_bc_updates
        actor_warmup_updates updates that train only the critic (after a warm start, so the critic
                             catches up before the policy starts to move)
        opponent_refresh     updates between refreshing the frozen copy that plays "latest" in the league
        snapshot_every       updates between adding a past copy to the league's pool (league stages only)
        pool_size            past copies kept in the pool
        torch_threads        CPU threads for PyTorch: a few, so the network does not fight the Unity
                             players for the cores (None: PyTorch's default, all cores)
        log_dir              where progress.csv (one row per update) is written
        """
        if commander and not centralised_critic:
            raise ValueError("the commander is trained with the centralised critic - use commander=False for IPPO")
        self.env = env
        self.hp = dict(learning_rate=learning_rate, critic_learning_rate=critic_learning_rate,
                       commander_learning_rate=commander_learning_rate, n_steps=n_steps, n_minibatches=n_minibatches,
                       n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
                       dual_clip=dual_clip, value_clip=value_clip, ent_coef=ent_coef, ent_coef_final=ent_coef_final,
                       ent_decay_updates=ent_decay_updates, target_kl=target_kl, vf_coef=vf_coef,
                       max_grad_norm=max_grad_norm, d_model=d_model, attn_heads=attn_heads, attn_layers=attn_layers,
                       hidden_size=hidden_size, recurrent=recurrent, chunk_len=chunk_len,
                       centralised_critic=centralised_critic, normalize_values=normalize_values,
                       commander=commander, commander_period=commander_period, kl_bc_coef=kl_bc_coef,
                       kl_bc_final=kl_bc_final, kl_bc_updates=kl_bc_updates,
                       actor_warmup_updates=actor_warmup_updates, opponent_refresh=opponent_refresh,
                       snapshot_every=snapshot_every, pool_size=pool_size, seed=seed)
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
        self.n_groups = len(REWARD_GROUPS)
        self.arch = dict(d=d_model, attn_heads=attn_heads, layers=attn_layers, hidden=hidden_size, recurrent=recurrent)
        self.policy = FleetPolicy(self.dims, self.heads, commander=commander, commander_period=commander_period,
                                  **self.arch).to(self.device)
        self.actor = self.policy                                    # the name the Week 10 notebooks print
        critic_cls = CentralCritic if centralised_critic else LocalCritic
        self.critic = critic_cls(self.dims, self.n_groups, d_model, attn_heads, attn_layers).to(self.device)
        self.value_norm = ValueNorm((self.n_groups,))
        self.cmd_value_norm = ValueNorm(())
        self.actor_optimizer = torch.optim.Adam(self.policy.captain.parameters(), lr=learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-5)
        self.commander_optimizer = (torch.optim.Adam(self.policy.commander.parameters(), lr=commander_learning_rate,
                                                     eps=1e-5) if commander else None)
        # ent_coef="auto": a learned coefficient, as SAC does (target: 30 % of the largest possible entropy)
        self.log_alpha = torch.tensor(np.log(0.01), device=self.device, requires_grad=True) if ent_coef == "auto" else None
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3) if ent_coef == "auto" else None
        self.target_entropy = 0.3 * float(sum(np.log(s) for s in self.head_sizes))

        # the imitation policy the KL anchor pulls towards (set by init_from_imitation)
        self.teacher: FleetPolicy | None = None
        # the league: a frozen copy of the current fleet ("latest") and a pool of past copies
        self.opponent = copy.deepcopy(self.policy).eval()
        self.league = OpponentPool(pool_size, seed)
        self._past: dict[int, FleetPolicy] = {}
        if hasattr(env, "set_opponent_policy"):
            env.set_opponent_policy(self._opponent_act, self.league.sample)
        if verbose:
            n_actor = sum(p.numel() for p in self.policy.parameters())
            n_critic = sum(p.numel() for p in self.critic.parameters())
            print(f"{'MAPPO' if centralised_critic else 'IPPO'} on {self.device}: captain"
                  f"{' + commander' if commander else ''} {n_actor:,} parameters, critic {n_critic:,}; "
                  f"{'GRU memory' if recurrent else 'no memory'}, {attn_layers} attention layers, "
                  f"heads {dict(zip(self.head_names, self.head_sizes))}")

        self.num_timesteps = 0          # decisions, summed over the parallel battles
        self.n_updates = 0
        self._last_obs = None
        self._episodes = []             # results of battles finished since the last log line
        self._t0 = None
        self._start_timesteps, self._total_timesteps = 0, None
        self._memory = self._play_memory = self._opp_memory = self._teacher_memory = None
        self.last_orders = None         # the commander's orders at the last predict() (for replays)

    # ------------------------------------------------------------------ acting

    def _tensors(self, obs: dict) -> dict:
        return to_tensors(obs, self.device)

    @staticmethod
    def _fits(memory: Memory | None, obs: dict) -> bool:
        return memory is not None and memory.h.shape[:2] == obs["alive"].shape

    def predict(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        """Actions [E, N, heads] for an observation from env.reset() / env.step(). The fleet keeps its
        memory and orders between calls (it is cleared at every battle start)."""
        if not self._fits(self._play_memory, obs):
            self._play_memory = self.policy.new_memory(*obs["alive"].shape)
        self.policy.eval()
        out = self.policy.act(self._tensors(obs), self._play_memory, deterministic)
        self.last_orders = out["orders"].cpu().numpy()
        return out["actions"].cpu().numpy()

    def _opponent_act(self, obs: dict, envs: list[int], ids: list[int]) -> np.ndarray:
        """The enemy fleet's actions in league battles: the frozen latest copy (id -1) or a past one."""
        E = self.env.n_envs
        if self._opp_memory is None or self._opp_memory.h.shape[0] != E:
            self._opp_memory = self.policy.new_memory(E, self.env.n_agents)
        tens = self._tensors(obs)
        envs, ids = np.asarray(envs), np.asarray(ids)
        actions = np.zeros((len(envs), self.env.n_agents, len(self.head_sizes)), dtype=np.int32)
        for sid in np.unique(ids):
            rows = np.flatnonzero(ids == sid)
            policy = self.opponent if sid < 0 else self._past_policy(int(sid))
            memory = self._opp_memory.take(envs[rows])
            out = policy.act({k: v[rows] for k, v in tens.items()}, memory)
            self._opp_memory.put(envs[rows], memory)
            actions[rows] = out["actions"].cpu().numpy()
        return actions

    def _past_policy(self, sid: int) -> FleetPolicy:
        if sid not in self._past:
            weights = self.league.weights(sid)
            if weights is None:
                return self.opponent
            policy = copy.deepcopy(self.opponent)
            policy.load_state_dict(weights)
            self._past = {k: v for k, v in self._past.items() if self.league.weights(k) is not None}
            self._past[sid] = policy
        return self._past[sid]

    def reset_battles(self) -> None:
        """Starts fresh battles (e.g. after an evaluation took over the environment)."""
        self._last_obs = self.env.reset()

    # ------------------------------------------------------------------ training loop

    def learn(self, total_timesteps: int, callback: BaseCallback | list | None = None, log_interval: int = 1):
        """Alternate between playing n_steps decisions in every battle and updating the networks."""
        callback = CallbackList(callback if isinstance(callback, list) else [callback] if callback else [])
        callback.init_callback(self)
        if hasattr(self.env, "set_opponent_policy"):
            self.env.set_opponent_policy(self._opponent_act, self.league.sample)   # another model may have taken it over
        self._last_obs = self.env.reset()           # fresh battles (an evaluation may have run in between)
        self._t0 = time.time()
        start = self._start_timesteps = self.num_timesteps
        self._total_timesteps = total_timesteps     # for status reports (speed, time left)
        while self.num_timesteps < total_timesteps:
            if not self.collect_rollouts(callback):
                break
            stats = self.train()
            if self.n_updates % self.hp["opponent_refresh"] == 0:
                self.opponent.load_state_dict(self.policy.state_dict())
            stage = self.env.stages[self.env.stage]
            if stage.get("opponent") == "league" and self.n_updates % self.hp["snapshot_every"] == 0:
                self.league.add(self.policy, self.n_updates)
            if log_interval and self.n_updates % log_interval == 0:
                self._log(stats, start)
            if not callback.on_rollout_end():
                break
        callback.on_training_end()
        return self

    def collect_rollouts(self, callback) -> bool:
        """Plays n_steps decisions in every battle and stores what happened."""
        T = self.hp["n_steps"]
        obs = self._last_obs
        E, N = obs["alive"].shape
        G, L = self.n_groups, sum(self.head_sizes)
        if not self._fits(self._memory, obs):
            self._memory = self.policy.new_memory(E, N)
        if self.teacher is not None and not self._fits(self._teacher_memory, obs):
            self._teacher_memory = self.teacher.new_memory(E, N)
        buf = {k: np.zeros((T, *v.shape), np.float32) for k, v in obs.items()}
        buf.update({
            "actions": np.zeros((T, E, N, len(self.head_sizes)), np.int64),
            "orders": np.zeros((T, E, N), np.int64),
            "logp": np.zeros((T, E, N), np.float32),
            "h": np.zeros((T, E, N, self.policy.hidden), np.float32),
            "values": np.zeros((T + 1, E, N, G), np.float32),
            "rewards": np.zeros((T, E, N, G), np.float32),
            "dones": np.zeros((T, E), np.float32),
            "won": np.zeros((T, E), np.float32),
            "cmd_decide": np.zeros((T, E), np.float32),
            "cmd_logp": np.zeros((T, E), np.float32),
            "cmd_value": np.zeros((T + 1, E), np.float32),
            "teacher_logits": np.zeros((T, E, N, L), np.float32) if self.teacher is not None else None,
        })
        self.policy.eval()
        for t in range(T):
            tens = self._tensors(obs)
            out = self.policy.act(tens, self._memory)
            values, cmd_value = self._values(tens)
            for k, v in obs.items():
                buf[k][t] = v
            buf["actions"][t], buf["orders"][t] = out["actions"].cpu().numpy(), out["orders"].cpu().numpy()
            buf["logp"][t], buf["h"][t] = out["logp"].cpu().numpy(), out["h"].cpu().numpy()
            buf["cmd_decide"][t], buf["cmd_logp"][t] = out["cmd_decide"].cpu().numpy(), out["cmd_logp"].cpu().numpy()
            buf["values"][t], buf["cmd_value"][t] = values, cmd_value
            if self.teacher is not None:
                buf["teacher_logits"][t] = self.teacher.act(tens, self._teacher_memory)["logits"].cpu().numpy()

            obs, rewards, dones, infos = self.env.step(buf["actions"][t], buf["orders"][t])
            buf["rewards"][t], buf["dones"][t] = self.env.last_reward_groups, dones
            for e in np.flatnonzero(dones):
                buf["won"][t, e] = float(infos[e]["won"])
            for info in infos:
                if info:
                    self._episodes.append(info)
                    if not info.get("evaluation"):
                        stage = self.env.stages[min(info["stage"], len(self.env.stages) - 1)]
                        self.league.record(info, difficulty=int(stage.get("difficulty", 0)))
            self.num_timesteps += E
            if not callback.on_step(locals()):
                self._last_obs = obs
                return False
        buf["values"][T], buf["cmd_value"][T] = self._values(self._tensors(obs))   # bootstrap: still going
        self._last_obs = obs
        self.buffer = buf
        self._advantages()
        if self.policy.commander is not None:
            self._commander_advantages()
        return True

    @torch.no_grad()
    def _values(self, tens: dict):
        """The critic's values in reward units: per ship and group [E, N, G], and the commander's [E]."""
        self.critic.eval()
        values, cmd, _ = self.critic(tens)
        values = self.value_norm.denormalize(values.cpu().numpy())
        cmd = self.cmd_value_norm.denormalize(cmd.cpu().numpy()) if cmd is not None else np.zeros(values.shape[0])
        return values, cmd

    def _advantages(self):
        """Generalised Advantage Estimation (GAE), per ship slot and reward group, backwards in time:

            delta_t     = r_t + gamma * V(s_t+1) * (1 - done_t) - V(s_t)
            advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_t+1

        The groups' advantages are added up for the policy (multi-head value)."""
        b, g, lam = self.buffer, self.hp["gamma"], self.hp["gae_lambda"]
        T = b["rewards"].shape[0]
        adv = np.zeros_like(b["rewards"])
        last = np.zeros_like(b["rewards"][0])
        for t in reversed(range(T)):
            not_done = (1.0 - b["dones"][t])[:, None, None]
            delta = b["rewards"][t] + g * b["values"][t + 1] * not_done - b["values"][t]
            last = delta + g * lam * not_done * last
            adv[t] = last
        b["returns"] = adv + b["values"][:T]
        b["advantages"] = adv.sum(-1)
        # the battle's result, for the critic's win-probability head (unknown if it ends after the rollout)
        b["win_label"], b["win_known"] = np.zeros_like(b["dones"]), np.zeros_like(b["dones"])
        label, known = np.zeros(b["dones"].shape[1]), np.zeros(b["dones"].shape[1])
        for t in reversed(range(T)):
            done = b["dones"][t] > 0.5
            label, known = np.where(done, b["won"][t], label), np.where(done, 1.0, known)
            b["win_label"][t], b["win_known"][t] = label, known

    def _commander_advantages(self):
        """GAE for the commander, whose steps last commander_period decisions (a semi-MDP): its reward
        is the fleet's average reward summed (discounted) over the period."""
        b, g, lam = self.buffer, self.hp["gamma"], self.hp["gae_lambda"]
        T, E = b["dones"].shape
        fleet_reward = b["rewards"].sum(-1).sum(-1) / np.maximum(1.0, b["exists"].sum(-1))     # [T, E]
        b["cmd_adv"], b["cmd_ret"] = np.zeros((T, E), np.float32), np.zeros((T, E), np.float32)
        for e in range(E):
            times = np.flatnonzero(b["cmd_decide"][:, e] > 0.5)
            next_adv = 0.0
            for j in reversed(range(len(times))):
                t0, t1 = times[j], (times[j + 1] if j + 1 < len(times) else T)
                ret, disc, done = 0.0, 1.0, False
                for t in range(t0, t1):
                    ret += disc * fleet_reward[t, e]
                    disc *= g
                    if b["dones"][t, e] > 0.5:
                        done = True
                        break
                if done:
                    boot, next_a = 0.0, 0.0
                else:
                    boot = b["cmd_value"][t1, e]
                    next_a = next_adv if j + 1 < len(times) else 0.0
                adv = ret + disc * boot - b["cmd_value"][t0, e] + disc * lam * next_a
                b["cmd_adv"][t0, e], b["cmd_ret"][t0, e] = adv, adv + b["cmd_value"][t0, e]
                next_adv = adv

    # ------------------------------------------------------------------ the update

    def _coefficients(self) -> tuple[float, float]:
        """The entropy and KL-anchor coefficients for this update (both fade over training)."""
        hp = self.hp
        if self.log_alpha is not None:
            ent = float(self.log_alpha.exp())
        else:
            ent = _linear(hp["ent_coef"], hp["ent_coef_final"], self.n_updates / max(1, hp["ent_decay_updates"]))
        kl = 0.0
        if self.teacher is not None:
            kl = _linear(hp["kl_bc_coef"], hp["kl_bc_final"], self.n_updates_since_teacher / max(1, hp["kl_bc_updates"]))
        return ent, kl

    def train(self) -> dict:
        """One update: the critic, the captains (PPO over chunks of decisions) and the commander."""
        hp, b = self.hp, self.buffer
        stats = defaultdict(list)
        T, E, N = b["alive"].shape
        dev = self.device

        # ---- targets in normalised units
        exists = b["exists"] > 0.5
        if hp["normalize_values"]:
            self.value_norm.update(b["returns"][exists])
        returns_norm = self.value_norm.normalize(b["returns"])
        values_old_norm = self.value_norm.normalize(b["values"][:T])
        cmd_returns_norm = None
        if self.policy.commander is not None and b["cmd_decide"].sum() > 0:
            if hp["normalize_values"]:
                self.cmd_value_norm.update(b["cmd_ret"][b["cmd_decide"] > 0.5])
            cmd_returns_norm = self.cmd_value_norm.normalize(b["cmd_ret"])
        live = (b["alive"] * b["exists"]) > 0.5
        adv = b["advantages"]
        adv = (adv - adv[live].mean()) / (adv[live].std() + 1e-8) if live.any() else adv
        ent_coef, kl_coef = self._coefficients()
        train_actor = self.n_updates >= hp["actor_warmup_updates"]

        self.policy.train()
        self.critic.train()
        for epoch in range(hp["n_epochs"]):
            # the critic: on battle states (t, e) - it is not recurrent
            states = np.random.permutation(T * E)
            for mb in np.array_split(states, hp["n_minibatches"]):
                t_i, e_i = np.unravel_index(mb, (T, E))
                self._critic_step(b, t_i, e_i, returns_norm, values_old_norm, cmd_returns_norm, stats)

            # the captains: on chunks of chunk_len consecutive decisions of one ship slot
            if train_actor:
                chunks = self._chunks(b)
                for mb in np.array_split(np.random.permutation(len(chunks)), hp["n_minibatches"]):
                    self._captain_step(b, chunks[mb], adv, ent_coef, kl_coef, stats)

            # the commander: on its decisions
            if self.policy.commander is not None and train_actor and b["cmd_decide"].sum() > 0:
                self._commander_step(b, stats)

            if hp["target_kl"] is not None and stats["approx_kl"] and np.mean(stats["approx_kl"][-hp["n_minibatches"]:]) > 1.5 * hp["target_kl"]:
                break                                 # SB3's early stop: the policy already moved far enough

        self.n_updates += 1
        self.n_updates_since_teacher += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        y, v = b["returns"][exists].sum(-1), b["values"][:T][exists].sum(-1)
        out["explained_variance"] = float(1 - np.var(y - v) / (np.var(y) + 1e-8))
        if self.policy.commander is not None and b["cmd_decide"].sum() > 0:
            decided = b["orders"][b["cmd_decide"] > 0.5]
            ship_live = ((b["alive"] * b["exists"])[b["cmd_decide"] > 0.5]) > 0.5
            out["orders_free"] = float((decided[ship_live] == 0).mean()) if ship_live.any() else 1.0
        return out

    def _critic_step(self, b, t_i, e_i, returns_norm, values_old_norm, cmd_returns_norm, stats):
        hp = self.hp
        if hp["centralised_critic"]:
            obs = {k: torch.as_tensor(b[k][t_i, e_i], device=self.device) for k in CRITIC_KEYS}
        else:
            obs = {k: torch.as_tensor(b[k][t_i, e_i], device=self.device) for k in CAPTAIN_KEYS}
        values, cmd, win = self.critic(obs)
        target = torch.as_tensor(returns_norm[t_i, e_i], dtype=torch.float32, device=self.device)
        old = torch.as_tensor(values_old_norm[t_i, e_i], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(b["exists"][t_i, e_i], device=self.device).unsqueeze(-1)
        error = (values - target) ** 2
        if hp["value_clip"] is not None:
            clipped = old + (values - old).clamp(-hp["value_clip"], hp["value_clip"])
            error = torch.max(error, (clipped - target) ** 2)
        value_loss = (error * mask).sum() / (mask.sum() * self.n_groups).clamp_min(1.0)
        loss = value_loss
        if cmd is not None and cmd_returns_norm is not None:
            decided = torch.as_tensor(b["cmd_decide"][t_i, e_i], device=self.device)
            if decided.sum() > 0:
                cmd_target = torch.as_tensor(cmd_returns_norm[t_i, e_i], dtype=torch.float32, device=self.device)
                loss = loss + ((cmd - cmd_target) ** 2 * decided).sum() / decided.sum()
        if win is not None:
            known = torch.as_tensor(b["win_known"][t_i, e_i], device=self.device)
            if known.sum() > 0:
                label = torch.as_tensor(b["win_label"][t_i, e_i], device=self.device)
                win_loss = (F.binary_cross_entropy_with_logits(win, label, reduction="none") * known).sum() / known.sum()
                loss = loss + 0.25 * win_loss
                stats["win_prob_loss"].append(win_loss.item())
        self.critic_optimizer.zero_grad()
        (hp["vf_coef"] * loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), hp["max_grad_norm"])
        self.critic_optimizer.step()
        stats["value_loss"].append(value_loss.item())

    def _chunks(self, b) -> np.ndarray:
        """(start time, battle, ship slot) of every chunk that holds a ship at some point."""
        T, E, N = b["alive"].shape
        L = self.hp["chunk_len"]
        starts = np.arange(0, T, L)
        any_ship = np.stack([b["exists"][s:s + L].max(0) for s in starts])        # [chunks per slot, E, N]
        c, e, n = np.nonzero(any_ship > 0.5)
        return np.stack([starts[c], e, n], axis=1)

    def _captain_step(self, b, chunks, adv, ent_coef, kl_coef, stats):
        hp, dev = self.hp, self.device
        T = b["alive"].shape[0]
        L = hp["chunk_len"]
        t_i = chunks[:, 0][None, :] + np.arange(L)[:, None]                      # [L, B]
        pad = (t_i < T).astype(np.float32)
        t_i = np.minimum(t_i, T - 1)
        e_i, n_i = np.broadcast_to(chunks[:, 1], t_i.shape), np.broadcast_to(chunks[:, 2], t_i.shape)

        def take(key):
            return torch.as_tensor(b[key][t_i, e_i, n_i], device=dev)

        obs = {k: take(k) for k in CAPTAIN_KEYS}
        mask = take("action_mask")
        starts = torch.as_tensor(b["starts"][t_i, e_i], device=dev)
        h0 = torch.as_tensor(b["h"][chunks[:, 0], chunks[:, 1], chunks[:, 2]], device=dev)
        logits = self.policy.captain.sequence(obs, take("orders"), h0, starts)
        logp, entropy = log_prob_entropy(logits, mask, self.head_sizes, take("actions"))

        w = take("alive") * take("exists") * torch.as_tensor(pad, device=dev)     # death masking + padding
        denom = w.sum().clamp_min(1.0)
        a = torch.as_tensor(adv[t_i, e_i, n_i], dtype=torch.float32, device=dev)
        ratio = torch.exp(logp - take("logp"))
        surrogate = ppo_objective(ratio, a, hp["clip_range"], hp["dual_clip"])
        policy_loss = -(surrogate * w).sum() / denom
        entropy_mean = (entropy * w).sum() / denom
        loss = policy_loss - ent_coef * entropy_mean
        if kl_coef > 0:
            kl = (kl_to(take("teacher_logits"), logits, mask, self.head_sizes) * w).sum() / denom
            loss = loss + kl_coef * kl
            stats["kl_bc"].append(kl.item())

        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.captain.parameters(), hp["max_grad_norm"])
        self.actor_optimizer.step()
        if self.log_alpha is not None:          # SAC-style: raise the bonus when entropy is below target
            alpha_loss = -(self.log_alpha * (entropy_mean.detach() - self.target_entropy))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            log_ratio = logp - take("logp")
            stats["approx_kl"].append((((ratio - 1) - log_ratio) * w).sum().item() / denom.item())
            stats["clip_fraction"].append((((ratio - 1).abs() > hp["clip_range"]).float() * w).sum().item() / denom.item())
        stats["policy_loss"].append(policy_loss.item())
        stats["entropy"].append(entropy_mean.item())

    def _commander_step(self, b, stats):
        """PPO for the commander on all its decisions of this rollout (one minibatch: there are few)."""
        hp, dev = self.hp, self.device
        t_i, e_i = np.nonzero(b["cmd_decide"] > 0.5)
        obs = {k: torch.as_tensor(b[k][t_i, e_i], device=dev) for k in COMMANDER_KEYS}
        mask = Commander.order_mask(obs)
        logits = self.policy.commander(obs).masked_fill(mask < 0.5, NEG_INF)
        dist = torch.distributions.Categorical(logits=logits)
        live = obs["own_mask"] * obs["alive"]
        orders = torch.as_tensor(b["orders"][t_i, e_i], device=dev)
        logp = (dist.log_prob(orders) * live).sum(-1)                  # the fleet's orders: one joint decision
        entropy = (dist.entropy() * live).sum(-1) / live.sum(-1).clamp_min(1.0)
        adv = b["cmd_adv"][t_i, e_i]
        adv = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv,
                              dtype=torch.float32, device=dev)
        ratio = torch.exp(logp - torch.as_tensor(b["cmd_logp"][t_i, e_i], device=dev))
        surrogate = ppo_objective(ratio, adv, hp["clip_range"], hp["dual_clip"])
        loss = -surrogate.mean() - 0.01 * entropy.mean()
        self.commander_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.commander.parameters(), hp["max_grad_norm"])
        self.commander_optimizer.step()
        stats["commander_loss"].append(loss.item())
        stats["commander_entropy"].append(entropy.mean().item())

    # ------------------------------------------------------------------ imitation warm start

    n_updates_since_teacher = 0

    def init_from_imitation(self, path: str, kl_anchor: bool = True, warmup_updates: int = 10) -> None:
        """Starts the captains from an imitation-learning checkpoint (imitation.py) - AlphaStar's
        supervised start. kl_anchor keeps a frozen copy as the teacher of the KL penalty."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["dims"] != {k: v for k, v in self.dims.items() if k in ck["dims"]} or ck["arch"] != self.arch:
            raise ValueError(f"{path} was trained for another network (dims / architecture differ): "
                             f"{ck['arch']} vs {self.arch}")
        self.policy.captain.load_state_dict(ck["captain"])
        self.opponent.load_state_dict(self.policy.state_dict())
        if kl_anchor and self.hp["kl_bc_coef"] > 0:
            self.teacher = FleetPolicy(self.dims, self.heads, commander=False, **self.arch).to(self.device).eval()
            self.teacher.captain.load_state_dict(ck["captain"])
            self.n_updates_since_teacher = 0
        self.hp["actor_warmup_updates"] = self.n_updates + warmup_updates
        if self.verbose:
            acc = ck.get("accuracy", {})
            print(f"captains start from imitation ({os.path.basename(path)}"
                  + (", held-out accuracy " + " ".join(f"{k} {v:.2f}" for k, v in acc.items()) if acc else "") + ")"
                  + (f"; KL anchor {self.hp['kl_bc_coef']} -> {self.hp['kl_bc_final']}" if self.teacher else ""))

    # ------------------------------------------------------------------ logging

    def _log(self, train_stats: dict, start: int):
        eps, self._episodes = self._episodes, []
        elapsed = max(1e-6, time.time() - self._t0)
        row = {"total_timesteps": self.num_timesteps, "updates": self.n_updates,
               "fps": int((self.num_timesteps - start) / elapsed), "time_elapsed": int(elapsed),
               "stage": getattr(self.env, "stage", 0), "episodes": len(eps)}
        if eps:
            row["ep_rew_mean"] = float(np.mean([e["episode"]["r"] for e in eps]))
            row["ep_len_mean"] = float(np.mean([e["episode"]["l"] for e in eps]))
            gate = [e for e in eps if e.get("opponent", "rule") in ("rule", "passive")]
            selfp = [e["won"] for e in eps if e.get("opponent") in ("latest", "past")]
            if gate:
                row["win_rate"] = float(np.mean([e["won"] for e in gate]))       # against the stage's own opponent
                row["pass_rate"] = float(np.mean([e.get("passed", e["won"]) for e in gate]))   # its goal
            if selfp:
                row["win_rate_self_play"] = float(np.mean(selfp))  # against our own copies
        row.update(train_stats)
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
                if header != PROGRESS_FIELDS:              # a log from the previous method: keep it aside
                    os.replace(path, path.replace(".csv", "_old.csv"))
                    new = True
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)

    # ------------------------------------------------------------------ saving and loading

    def save(self, path: str) -> str:
        path = path if path.endswith(".pt") else path + ".pt"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"method": "commander-mappo", "policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "commander_optimizer": self.commander_optimizer.state_dict() if self.commander_optimizer else None,
                    "log_alpha": float(self.log_alpha) if self.log_alpha is not None else None,
                    "hyperparameters": self.hp, "value_norm": self.value_norm.state_dict(),
                    "cmd_value_norm": self.cmd_value_norm.state_dict(), "dims": self.dims, "heads": self.heads,
                    "head_names": self.head_names, "num_timesteps": self.num_timesteps, "n_updates": self.n_updates,
                    "stage": getattr(self.env, "stage", 0), "league": self.league.state_dict(),
                    "teacher": self.teacher.state_dict() if self.teacher is not None else None,
                    "n_updates_since_teacher": self.n_updates_since_teacher}, path)
        return path

    @classmethod
    def load(cls, path: str, env, device: str = "auto", **overrides) -> "MAPPO":
        """A saved model, ready to predict or to continue learning in env.
        The environment is moved to the curriculum stage the model was saved on."""
        path = path if path.endswith(".pt") else path + ".pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("method") != "commander-mappo":
            raise ValueError(f"{path} was saved by the earlier flat MAPPO; this version cannot continue it - "
                             f"start a new run (a new RUN_NAME)")
        if ck["dims"] != dict(env.dims) or [h["size"] for h in ck["heads"]] != list(env.head_sizes):
            raise ValueError(f"{path} was trained on a different observation layout: {ck['dims']} vs {dict(env.dims)}")
        model = cls(env, device=device, **{"verbose": 0, **ck["hyperparameters"], **overrides})
        model.policy.load_state_dict(ck["policy"])
        model.opponent.load_state_dict(ck["policy"])
        model.critic.load_state_dict(ck["critic"])
        model.value_norm.load_state_dict(ck["value_norm"])
        model.cmd_value_norm.load_state_dict(ck["cmd_value_norm"])
        model.actor_optimizer.load_state_dict(ck["actor_optimizer"])
        model.critic_optimizer.load_state_dict(ck["critic_optimizer"])
        if model.commander_optimizer is not None and ck.get("commander_optimizer"):
            model.commander_optimizer.load_state_dict(ck["commander_optimizer"])
        if model.log_alpha is not None and ck.get("log_alpha") is not None:
            model.log_alpha.data.fill_(ck["log_alpha"])
        model.league.load_state_dict(ck["league"])
        if ck.get("teacher") is not None:
            model.teacher = FleetPolicy(model.dims, model.heads, commander=False, **model.arch).to(model.device).eval()
            model.teacher.load_state_dict(ck["teacher"])
            model.n_updates_since_teacher = ck.get("n_updates_since_teacher", 0)
        model.num_timesteps, model.n_updates = ck["num_timesteps"], ck["n_updates"]
        if hasattr(env, "set_stage"):
            env.set_stage(ck.get("stage", 0))              # carry on with the curriculum where it was
        return model
