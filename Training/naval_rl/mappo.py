"""Recurrent MAPPO: rollout collection, GAE and the PPO update.

Data layout. Every worker contributes two "team rows" (Player, Enemy); every team row has
max_team agent slots. Buffers are [T, W, 2, N, ...]. A team row is *active* when its experience
trains the policy this episode (the learner's side, or both sides against the latest self); rule-AI
and snapshot-controlled sides are acted for but not trained on.

Agent death (MAPPO death masking). A slot that had a hull at the start of the episode stays in the
data until the episode ends. After the hull sinks the slot is excluded from the actor loss, but its
value keeps being learned from the shared team reward, so a ship's last decisions still get credit
for what its team achieved after it went down (a destroyer that dies spotting for the battle line).

Truncation vs termination. An episode ends only when the game ends (victory, defeat, or the game's
own clock - time remaining is in every observation, so running out of time is a real terminal).
A rollout that stops mid-battle is a truncation and is bootstrapped from the critic.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .league import EpisodePlan, League
from .model import Actor, Critic, LocalCritic, ValueNorm, evaluate_actions, sample_actions
from .rewards import RewardFunction
from .spec import Spec

ACTOR_KEYS = ["self", "allies", "ally_mask", "contacts", "contact_mask", "zones", "zone_mask"]
CRITIC_KEYS = ["critic_own", "critic_enemy", "critic_enemy_mask", "critic_zones", "critic_zone_mask", "critic_match"]


def to_t(x, device):
    return torch.as_tensor(np.ascontiguousarray(x), device=device)


class MAPPO:
    def __init__(self, cfg, spec: Spec, workers, league: League, device):
        self.cfg, self.spec, self.workers, self.league, self.device = cfg, spec, workers, league, device
        self.W, self.N, self.H = len(workers), spec.max_team, len(spec.heads)
        self.sizes = spec.head_sizes

        self.actor = Actor(spec, cfg.d_model, cfg.heads, cfg.layers, cfg.hidden).to(device)
        if cfg.algo == "ippo":
            self.critic = LocalCritic(spec, cfg.d_model, cfg.heads, cfg.layers).to(device)
        else:
            self.critic = Critic(spec, cfg.d_model, cfg.heads, cfg.layers).to(device)
        self.value_norm = ValueNorm().to(device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor, eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, eps=1e-5)
        self.reward_fn = RewardFunction(spec.team_reward_components, spec.agent_reward_components,
                                        cfg.team_weights, cfg.agent_weights, cfg.anneal_updates,
                                        cfg.shaping_floor, cfg.spotting_reward)
        self.snapshot_actors: dict[int, Actor] = {}
        self.update = 0
        self.env_steps = 0

        # live state per worker
        self.obs = [None] * self.W
        self.plans: list[EpisodePlan | None] = [None] * self.W
        self.hidden = torch.zeros(self.W, 2, self.N, cfg.hidden, device=device)
        self.starts = np.ones((self.W, 2), dtype=np.float32)
        self.episode_start_t = np.zeros(self.W, dtype=np.int64)
        self.completed: list[dict] = []
        self.collect_stats: dict = {}
        self._buf = None

    # ------------------------------------------------------------------ episodes

    def reset_all(self):
        for w, worker in enumerate(self.workers):
            self.plans[w] = self.league.sample(w)
            worker.reset_async(self.plans[w].reset)
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.recv()
        self.starts[:] = 1.0
        self.hidden.zero_()

    def _actor_for(self, plan: EpisodePlan, team: int):
        ctrl = plan.controller(team)
        if ctrl in ("learner", "latest"):
            return self.actor
        if ctrl == "snapshot":
            return self._snapshot(plan.snapshot_id)
        return None

    def _snapshot(self, sid: int) -> Actor:
        if sid not in self.snapshot_actors:
            path = next(s["path"] for s in self.league.snapshots if s["id"] == sid)
            a = Actor(self.spec, self.cfg.d_model, self.cfg.heads, self.cfg.layers, self.cfg.hidden).to(self.device)
            a.load_state_dict(torch.load(path, map_location=self.device))
            a.eval()
            self.snapshot_actors[sid] = a
            live = {s["id"] for s in self.league.snapshots}
            for k in list(self.snapshot_actors):
                if k not in live:
                    del self.snapshot_actors[k]
        return self.snapshot_actors[sid]

    def exists_mask(self, obs) -> np.ndarray:
        """[2, N]: slot held a hull at the start of this episode."""
        e = np.zeros((2, self.N), dtype=np.float32)
        for t in range(2):
            e[t, :min(int(obs.ships[t]), self.N)] = 1.0
        return e

    def critic_inputs(self, obs_arrays: dict, exists: np.ndarray) -> dict:
        c = {k: obs_arrays[k] for k in CRITIC_KEYS}
        if self.cfg.critic_mode == "belief":
            c["critic_enemy"] = c["critic_enemy"].copy()
            p0, p1 = self.spec.enemy_privileged
            c["critic_enemy"][..., p0:p1] = 0.0
        c["own_mask"] = exists
        return c

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def act(self, deterministic: bool = False):
        """Chooses actions for every policy-controlled team row. Returns per-worker action arrays and
        what the buffer needs about the rows that train."""
        W, N, H, dev = self.W, self.N, self.H, self.device
        actions = np.zeros((W, 2, N, H), dtype=np.int32)
        logp = np.zeros((W, 2, N), dtype=np.float32)
        new_hidden = self.hidden.clone()

        groups = defaultdict(list)          # actor module -> [(w, t)]
        for w in range(W):
            for t in range(2):
                a = self._actor_for(self.plans[w], t)
                if a is not None:
                    groups[id(a)].append((a, w, t))

        for items in groups.values():
            actor = items[0][0]
            rows = [(w, t) for _, w, t in items]
            obs = {k: to_t(np.stack([self.obs[w].arrays[k][t] for w, t in rows]), dev).flatten(0, 1) for k in ACTOR_KEYS}
            mask = to_t(np.stack([self.obs[w].arrays["action_mask"][t] for w, t in rows]), dev).flatten(0, 1)
            h = torch.stack([self.hidden[w, t] for w, t in rows]).flatten(0, 1)
            st = to_t(np.repeat(np.array([self.starts[w, t] for w, t in rows]), N), dev)
            logits, h2, _ = actor(obs, h, st)
            a, lp, _ = sample_actions(logits, mask, self.sizes, deterministic)
            a = a.view(len(rows), N, H).cpu().numpy()
            lp = lp.view(len(rows), N).cpu().numpy()
            h2 = h2.view(len(rows), N, -1)
            for i, (w, t) in enumerate(rows):
                actions[w, t] = a[i]
                logp[w, t] = lp[i]
                new_hidden[w, t] = h2[i]
        return actions, logp, new_hidden

    @torch.no_grad()
    def values(self, obs_list) -> np.ndarray:
        """[W, 2, N] denormalised values for every team row (zeros where unused)."""
        W, N, dev = self.W, self.N, self.device
        out = np.zeros((W, 2, N), dtype=np.float32)
        rows = [(w, t) for w in range(W) for t in range(2) if self.plans[w].trains_on(t)]
        if not rows:
            return out
        if self.cfg.algo == "ippo":
            obs = {k: to_t(np.stack([obs_list[w].arrays[k][t] for w, t in rows]), dev).flatten(0, 1) for k in ACTOR_KEYS}
            v, _ = self.critic(obs)
            v = self.value_norm.denormalize(v).view(len(rows), N).cpu().numpy()
        else:
            cin = [self.critic_inputs(obs_list[w].arrays, self.exists_mask(obs_list[w])) for w in range(W)]
            batch = {k: to_t(np.stack([cin[w][k][t] for w, t in rows]), dev) for k in cin[0]}
            batch = {k: v.repeat_interleave(N, dim=0) for k, v in batch.items()}
            idx = torch.arange(N, device=dev).repeat(len(rows))
            v, _ = self.critic(batch, idx)
            v = self.value_norm.denormalize(v).view(len(rows), N).cpu().numpy()
        for i, (w, t) in enumerate(rows):
            out[w, t] = v[i]
        return out

    # ------------------------------------------------------------------ collection

    def _alloc(self):
        T, W, N, H = self.cfg.rollout, self.W, self.N, self.H
        o = self.obs[0].arrays
        b = {}
        for k in ACTOR_KEYS + ["action_mask"]:
            b[k] = np.zeros((T, W, 2) + o[k].shape[1:], dtype=np.float32)
        for k in CRITIC_KEYS:
            b[k] = np.zeros((T, W, 2) + o[k].shape[1:], dtype=np.float32)
        b["own_mask"] = np.zeros((T, W, 2, N), dtype=np.float32)
        b["actions"] = np.zeros((T, W, 2, N, H), dtype=np.int64)
        b["logp"] = np.zeros((T, W, 2, N), dtype=np.float32)
        b["values"] = np.zeros((T + 1, W, 2, N), dtype=np.float32)
        b["rewards"] = np.zeros((T, W, 2, N), dtype=np.float32)
        b["dones"] = np.zeros((T, W, 2), dtype=np.float32)
        b["starts"] = np.zeros((T, W, 2), dtype=np.float32)
        b["active"] = np.zeros((T, W, 2), dtype=np.float32)
        b["alive"] = np.zeros((T, W, 2, N), dtype=np.float32)
        b["hidden"] = np.zeros((T, W, 2, N, self.cfg.hidden), dtype=np.float32)
        b["outcome"] = np.zeros((T, W, 2), dtype=np.float32)
        b["outcome_mask"] = np.zeros((T, W, 2), dtype=np.float32)
        self._buf = b

    def collect(self):
        cfg, T, W = self.cfg, self.cfg.rollout, self.W
        if self._buf is None:
            self._alloc()
        b = self._buf
        self.episode_start_t[:] = 0
        t0 = time.time()
        timing = {"policy_s": 0.0, "env_wait_s": 0.0, "reset_s": 0.0}
        sim_ms = []

        for t in range(T):
            tp = time.time()
            vals = self.values(self.obs)
            actions, logp, new_hidden = self.act()
            timing["policy_s"] += time.time() - tp

            for w in range(W):
                o = self.obs[w]
                ex = self.exists_mask(o)
                cin = self.critic_inputs(o.arrays, ex)
                for k in ACTOR_KEYS + ["action_mask"]:
                    b[k][t, w] = o.arrays[k]
                for k in CRITIC_KEYS:
                    b[k][t, w] = cin[k]
                b["own_mask"][t, w] = ex
                b["alive"][t, w] = o.arrays["alive"]
                b["starts"][t, w] = self.starts[w]
                b["active"][t, w] = [float(self.plans[w].trains_on(k)) for k in range(2)]
            b["actions"][t] = actions
            b["logp"][t] = logp
            b["values"][t] = vals
            b["hidden"][t] = self.hidden.cpu().numpy()

            self.hidden = new_hidden
            self.starts[:] = 0.0
            for w, worker in enumerate(self.workers):
                worker.step_async(actions[w])
            for w, worker in enumerate(self.workers):
                tw = time.time()
                o2 = worker.recv()
                timing["env_wait_s"] += time.time() - tw
                if o2.diag.get("sim_ms"):
                    sim_ms.append(o2.diag["sim_ms"])
                ex = self.exists_mask(self.obs[w])
                r = self.reward_fn(o2.arrays["team_reward"], o2.arrays["agent_reward"], self.update)
                b["rewards"][t, w] = r * ex
                if o2.terminal:
                    b["dones"][t, w] = 1.0
                    self._finish_episode(w, t, o2)
                    self.plans[w] = self.league.sample(w)
                    tr = time.time()
                    worker.reset_async(self.plans[w].reset)
                    o2 = worker.recv()
                    timing["reset_s"] += time.time() - tr
                    self.starts[w] = 1.0
                    self.hidden[w] = 0.0
                    self.episode_start_t[w] = t + 1
                self.obs[w] = o2
            self.env_steps += W

        b["values"][T] = self.values(self.obs)
        diag = [o.diag for o in self.obs if o.diag]
        self.collect_stats = {
            **timing,
            "sim_ms_mean": float(np.mean(sim_ms)) if sim_ms else 0.0,
            "sim_ms_max": float(np.max(sim_ms)) if sim_ms else 0.0,
            "env_heap_mb_max": max((d.get("heap_mb", 0) for d in diag), default=0.0),
            "env_objects_max": max((d.get("objects", 0) for d in diag), default=0),
            "env_gc0_max": max((d.get("gc0", 0) for d in diag), default=0),
        }
        return time.time() - t0

    def _finish_episode(self, w: int, t: int, obs):
        plan = self.plans[w]
        b = self._buf
        for team in range(2):
            if obs.draw:
                score = 0.5
            else:
                score = 1.0 if obs.winner == team else 0.0
            s0 = self.episode_start_t[w]
            b["outcome"][s0:t + 1, w, team] = score
            b["outcome_mask"][s0:t + 1, w, team] = 1.0
        L = plan.learner_team
        learner_score = 0.5 if obs.draw else float(obs.winner == L)
        self.league.record(plan, learner_score)
        rec = {"stage": plan.stage, "learner_team": L, "opponent": plan.opponent, "difficulty": plan.difficulty,
               "learner_score": learner_score, "battle_time": obs.battle_time, "reason": obs.reason}
        for k, v in (obs.stats or {}).items():
            if isinstance(v, list) and len(v) == 2:
                rec["learner_" + k] = v[L]
                rec["opponent_" + k] = v[1 - L]
        self.completed.append(rec)

    # ------------------------------------------------------------------ learning

    def advantages(self):
        cfg, b = self.cfg, self._buf
        T = cfg.rollout
        adv = np.zeros_like(b["rewards"])
        last = np.zeros_like(b["rewards"][0])
        for t in reversed(range(T)):
            nonterminal = (1.0 - b["dones"][t])[..., None]
            delta = b["rewards"][t] + cfg.gamma * b["values"][t + 1] * nonterminal - b["values"][t]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            adv[t] = last
        b["advantages"] = adv
        b["returns"] = adv + b["values"][:T]

    def learn(self) -> dict:
        cfg, b, dev = self.cfg, self._buf, self.device
        T, W, N, L = cfg.rollout, self.W, self.N, cfg.chunk
        self.advantages()

        actor_valid = b["active"][..., None] * b["alive"] * b["own_mask"]          # [T, W, 2, N]
        critic_valid = b["active"][..., None] * b["own_mask"]
        if actor_valid.sum() < 1:
            return {}

        adv = b["advantages"]
        m = actor_valid > 0
        adv_n = (adv - adv[m].mean()) / (adv[m].std() + 1e-8)
        self.value_norm.update(to_t(b["returns"][critic_valid > 0], dev))

        # sequences = (w, team, slot); chunks of length L along time
        n_chunks = T // L
        chunks = []
        for w in range(W):
            for tm in range(2):
                for n in range(N):
                    for c in range(n_chunks):
                        if critic_valid[c * L:(c + 1) * L, w, tm, n].sum() > 0:
                            chunks.append((c * L, w, tm, n))
        chunks = np.array(chunks, dtype=np.int64)
        progress = min(1.0, self.update / max(1, getattr(cfg, "ent_decay_updates", cfg.total_updates)))
        ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * progress

        stats = defaultdict(list)
        for _ in range(cfg.epochs):
            np.random.shuffle(chunks)
            for mb in np.array_split(chunks, cfg.minibatches):
                if len(mb) == 0:
                    continue
                ti = mb[:, 0][None, :] + np.arange(L)[:, None]                     # [L, B]
                wi = np.broadcast_to(mb[:, 1][None, :], ti.shape)
                mi = np.broadcast_to(mb[:, 2][None, :], ti.shape)
                ni = np.broadcast_to(mb[:, 3][None, :], ti.shape)

                obs = {k: to_t(b[k][ti, wi, mi, ni], dev) for k in ACTOR_KEYS}
                amask = to_t(b["action_mask"][ti, wi, mi, ni], dev)
                acts = to_t(b["actions"][ti, wi, mi, ni], dev)
                old_lp = to_t(b["logp"][ti, wi, mi, ni], dev)
                h0 = to_t(b["hidden"][mb[:, 0], mb[:, 1], mb[:, 2], mb[:, 3]], dev)
                starts = to_t(b["starts"][ti, wi, mi], dev)
                av = to_t(actor_valid[ti, wi, mi, ni], dev)
                cv = to_t(critic_valid[ti, wi, mi, ni], dev)
                A = to_t(adv_n[ti, wi, mi, ni], dev)
                R = to_t(b["returns"][ti, wi, mi, ni], dev)
                V_old = to_t(b["values"][ti, wi, mi, ni], dev)
                outc = to_t(b["outcome"][ti, wi, mi], dev)
                outm = to_t(b["outcome_mask"][ti, wi, mi], dev) * cv

                # ---- actor ----
                logits = self.actor.forward_sequence(obs, h0, starts)
                lp, ent = evaluate_actions(logits, amask, self.sizes, acts)
                ratio = torch.exp(lp - old_lp)
                s1 = ratio * A
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * A
                denom = av.sum().clamp_min(1.0)
                pol_loss = -(torch.min(s1, s2) * av).sum() / denom
                ent_mean = (ent * av).sum() / denom
                actor_loss = pol_loss - ent_coef * ent_mean

                # ---- critic ----
                if cfg.algo == "ippo":
                    flat = {k: v.flatten(0, 1) for k, v in obs.items()}
                    v_pred, win_logit = self.critic(flat)
                else:
                    cobs = {k: to_t(b[k][ti, wi, mi], dev).flatten(0, 1) for k in CRITIC_KEYS + ["own_mask"]}
                    v_pred, win_logit = self.critic(cobs, torch.as_tensor(ni.reshape(-1), device=dev))
                v_pred = v_pred.view(L, -1)
                win_logit = win_logit.view(L, -1)
                R_n = self.value_norm.normalize(R)
                V_old_n = self.value_norm.normalize(V_old)
                v_clip = V_old_n + torch.clamp(v_pred - V_old_n, -cfg.value_clip, cfg.value_clip)
                v_loss = torch.max((v_pred - R_n) ** 2, (v_clip - R_n) ** 2)
                cdenom = cv.sum().clamp_min(1.0)
                value_loss = (v_loss * cv).sum() / cdenom
                win_loss = (F.binary_cross_entropy_with_logits(win_logit, outc, reduction="none") * outm).sum() / outm.sum().clamp_min(1.0)
                critic_loss = cfg.value_coef * value_loss + cfg.win_coef * win_loss

                train_actor = self.update >= getattr(cfg, "actor_freeze_updates", 0)
                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                ((actor_loss if train_actor else 0.0) + critic_loss).backward()
                gn_a = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm) if train_actor else torch.zeros(())
                gn_c = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                if train_actor:
                    self.opt_actor.step()
                self.opt_critic.step()

                with torch.no_grad():
                    approx_kl = (((ratio - 1) - (lp - old_lp)) * av).sum() / denom
                    clipfrac = (((ratio - 1).abs() > cfg.clip).float() * av).sum() / denom
                stats["policy_loss"].append(pol_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["win_loss"].append(win_loss.item())
                stats["entropy"].append(ent_mean.item())
                stats["approx_kl"].append(approx_kl.item())
                stats["clipfrac"].append(clipfrac.item())
                stats["grad_norm_actor"].append(float(gn_a))
                stats["grad_norm_critic"].append(float(gn_c))

        self.update += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["ent_coef"] = ent_coef
        out["shaping"] = self.reward_fn.shaping(self.update)
        out["adv_std"] = float(adv[m].std())
        out["return_mean"] = float(b["returns"][critic_valid > 0].mean())
        out["reward_mean"] = float((b["rewards"] * critic_valid).sum() / max(1.0, critic_valid.sum()))
        out["actor_samples"] = int(actor_valid.sum())
        return out

    # ------------------------------------------------------------------ persistence

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "value_norm": self.value_norm.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "update": self.update, "env_steps": self.env_steps, "league": self.league.state_dict(),
            "spec": self.spec.to_json(), "config": self.cfg.to_dict(),
        }

    def load_state_dict(self, d: dict, weights_only: bool = False) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.value_norm.load_state_dict(d["value_norm"])
        if weights_only:
            return
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        self.update = d["update"]
        self.env_steps = d["env_steps"]
        self.league.load_state_dict(d["league"])
