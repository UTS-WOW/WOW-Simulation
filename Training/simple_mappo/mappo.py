"""MAPPO: PPO for a team of agents, with a centralised critic.

Reference: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games",
NeurIPS 2022 (https://arxiv.org/abs/2103.01955).

MAPPO is ordinary PPO (Week 10) for a team of agents:

* Parameter sharing. Every ship runs the same actor network. A battleship and a destroyer behave
  differently because their observations differ (ship class, guns, position), not because they
  have different networks.
* Centralised training, decentralised execution (CTDE). The actor only sees what its own ship knows
  (fog of war included), so the trained policy can play the real game. The critic, used only during
  training, sees the true state of the whole battle - which makes its value estimates, and so the
  advantages, far less noisy. centralised_critic=False gives the critic only the ship's own view
  instead: that is IPPO (independent PPO), the baseline MAPPO is compared against.

The paper's five implementation recommendations, and where they are here:

1. Value normalisation    ValueNorm: the critic learns returns rescaled to about unit size
2. Agent-specific global state   env.py: the critic input is the whole battle + this ship's own view + its id
3. Training data usage    few minibatches (n_minibatches=4) and several epochs (n_epochs=10)
4. PPO clipping           clip_range=0.2
5. Death masking          a sunk ship stops training the actor; its critic keeps learning the team's
                          value until the battle ends, because team rewards keep arriving

The interface follows Stable-Baselines3, so the notebook reads like the Week 9 and 10 ones:

    model = MAPPO(env, learning_rate=3e-4, n_steps=256, n_minibatches=4, n_epochs=10)
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

from .callbacks import BaseCallback, CallbackList

NEG_INF = -1e9


def layer(n_in: int, n_out: int, gain: float = np.sqrt(2)) -> nn.Linear:
    """Orthogonal initialisation, as in Stable-Baselines3 and the MAPPO paper."""
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


def mlp(n_in: int, n_out: int, hidden: int, out_gain: float) -> nn.Sequential:
    """Two hidden layers. The game already scales the inputs (almost all within -1.5..1.5), so they go
    in as they are; LayerNorm comes after the first layer, where the empty slots of a small battle
    (zeros) add nothing - on the raw input they would change the normalisation from stage to stage."""
    return nn.Sequential(
        layer(n_in, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        layer(hidden, hidden), nn.ReLU(),
        layer(hidden, n_out, gain=out_gain),
    )


class Actor(nn.Module):
    """Policy network: a ship's own observation -> a score (logit) for every option of every head."""

    def __init__(self, obs_dim: int, head_sizes: list[int], hidden: int = 256):
        super().__init__()
        self.head_sizes = head_sizes
        self.net = mlp(obs_dim, sum(head_sizes), hidden, out_gain=0.01)    # small: start close to uniformly random

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distributions(self, obs: torch.Tensor, action_mask: torch.Tensor) -> list[torch.distributions.Categorical]:
        """One categorical distribution per head; illegal options get probability 0."""
        logits = self(obs).masked_fill(action_mask < 0.5, NEG_INF)
        return [torch.distributions.Categorical(logits=part) for part in torch.split(logits, self.head_sizes, dim=-1)]


class Critic(nn.Module):
    """Value network: the whole battle (agent-specific global state) -> expected future reward."""

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden, out_gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ValueNorm:
    """Running mean and standard deviation of the value targets (MAPPO's value normalisation).

    Returns are a few units in a 1v1 duel and much larger in an 8v8 battle with shared team rewards.
    The critic always learns targets of about unit size: it predicts (return - mean) / std, and its
    output is turned back into game units for the advantages.

    beta = 0.99999 as in the MAPPO code: with the debiasing below this is close to an average over
    all training so far, so the critic's targets stay put between updates. (A fast 0.99 moved them
    every update, and in the duel the policy learned noticeably slower.)
    """

    def __init__(self, beta: float = 0.99999):
        self.beta, self.mean, self.mean_sq, self.debias = beta, 0.0, 0.0, 0.0

    def update(self, returns: np.ndarray) -> None:
        self.mean = self.beta * self.mean + (1 - self.beta) * float(returns.mean())
        self.mean_sq = self.beta * self.mean_sq + (1 - self.beta) * float((returns ** 2).mean())
        self.debias = self.beta * self.debias + (1 - self.beta)

    def stats(self) -> tuple[float, float]:
        if self.debias == 0.0:
            return 0.0, 1.0                            # nothing seen yet
        mean, mean_sq = self.mean / self.debias, self.mean_sq / self.debias
        return mean, max(mean_sq - mean ** 2, 1e-2) ** 0.5

    def normalize(self, x):
        mean, std = self.stats()
        return (x - mean) / std

    def denormalize(self, x):
        mean, std = self.stats()
        return x * std + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "mean_sq": self.mean_sq, "debias": self.debias}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.mean_sq, self.debias = d["mean"], d["mean_sq"], d["debias"]


class MAPPO:
    def __init__(self, env, learning_rate: float = 3e-4, n_steps: int = 256, n_minibatches: int = 4,
                 n_epochs: int = 10, gamma: float = 0.99, gae_lambda: float = 0.95, clip_range: float = 0.2,
                 ent_coef: float = 0.01, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                 hidden_size: int = 256, centralised_critic: bool = True, normalize_values: bool = True,
                 opponent_refresh: int = 10,
                 device: str = "auto", seed: int = 0, verbose: int = 1, log_dir: str | None = None):
        """
        n_steps     decisions collected from every battle before each update (a decision covers
                    env.decision_period seconds of game time)
        n_minibatches  the collected data is split into this many minibatches per epoch. A fixed
                    count (rather than a fixed minibatch size) gives every curriculum stage the same
                    n_epochs x n_minibatches gradient steps per update, whether 1 or 8 ships collected it
        n_epochs    passes over the collected data per update
        gamma       discount factor (0.99 looks ~100 s ahead)
        gae_lambda  bias/variance trade-off of the advantage estimate
        clip_range  how far one update may move the policy (PPO's clip)
        ent_coef    bonus for keeping the policy random enough to explore
        vf_coef     weight of the critic's loss
        centralised_critic  True: MAPPO (the critic sees the whole battle). False: IPPO (the critic
                    sees only the ship's own view) - for comparing the two
        normalize_values  MAPPO's value normalisation (ValueNorm); False lets the critic learn raw returns
        opponent_refresh  updates between refreshing the frozen copy that flies the enemy in
                    self-play battles (curriculum stages with "self_play")
        log_dir     where progress.csv (one row per update) is written
        """
        self.env = env
        self.hp = dict(learning_rate=learning_rate, n_steps=n_steps, n_minibatches=n_minibatches, n_epochs=n_epochs,
                       gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef,
                       vf_coef=vf_coef, max_grad_norm=max_grad_norm, hidden_size=hidden_size,
                       centralised_critic=centralised_critic, normalize_values=normalize_values,
                       opponent_refresh=opponent_refresh, seed=seed)
        self.verbose = verbose
        self.log_dir = log_dir
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.local_dim, self.state_dim = env.local_dim, env.state_dim
        self.head_sizes, self.head_names = list(env.head_sizes), list(env.head_names)
        self.actor = Actor(self.local_dim, self.head_sizes, hidden_size).to(self.device)
        self.critic_key = "state" if centralised_critic else "local"          # what the critic reads
        self.critic = Critic(self.state_dim if centralised_critic else self.local_dim, hidden_size).to(self.device)
        self.value_norm = ValueNorm()
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=learning_rate, eps=1e-5)
        # self-play: a frozen copy of the actor flies the enemy fleet in some battles. It is not
        # trained; every opponent_refresh updates it is overwritten with the current actor.
        self.opponent = copy.deepcopy(self.actor).eval()
        if hasattr(env, "set_opponent_policy"):
            env.set_opponent_policy(self.opponent_predict)
        if verbose:
            n_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
            critic_in = self.state_dim if centralised_critic else self.local_dim
            print(f"{'MAPPO' if centralised_critic else 'IPPO'} on {self.device}. Actor input {self.local_dim}, critic input {critic_in}, "
                  f"{len(self.head_sizes)} action heads {dict(zip(self.head_names, self.head_sizes))}, "
                  f"{n_params:,} parameters")

        self.num_timesteps = 0          # decisions, summed over the parallel battles
        self.n_updates = 0
        self._last_obs = None
        self._episodes = []             # results of battles finished since the last log line
        self._t0 = None
        self._start_timesteps, self._total_timesteps = 0, None

    # ------------------------------------------------------------------ acting

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _act(self, obs: dict, deterministic: bool = False):
        """Actions, their log-probabilities and the critic's values for every ship of every battle."""
        dists = self.actor.distributions(self._t(obs["local"]), self._t(obs["action_mask"]))
        actions = torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in dists], dim=-1)
        logp = sum(d.log_prob(actions[..., i]) for i, d in enumerate(dists))
        values = self.value_norm.denormalize(self.critic(self._t(obs[self.critic_key])))   # in reward units
        return actions.cpu().numpy(), logp.cpu().numpy(), values.cpu().numpy()

    def predict(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        """Actions [E, N, heads] for an observation from env.reset() / env.step()."""
        return self._act(obs, deterministic)[0]

    @torch.no_grad()
    def opponent_predict(self, obs: dict) -> np.ndarray:
        """The self-play opponent's actions, from the enemy fleet's own observation."""
        dists = self.opponent.distributions(self._t(obs["local"]), self._t(obs["action_mask"]))
        return torch.stack([d.sample() for d in dists], dim=-1).cpu().numpy()

    def reset_battles(self) -> None:
        """Starts fresh battles (e.g. after an evaluation took over the environment)."""
        self._last_obs = self.env.reset()

    # ------------------------------------------------------------------ training loop

    def learn(self, total_timesteps: int, callback: BaseCallback | list | None = None, log_interval: int = 1):
        """Alternate between playing n_steps decisions in every battle and updating the networks."""
        callback = CallbackList(callback if isinstance(callback, list) else [callback] if callback else [])
        callback.init_callback(self)
        if hasattr(self.env, "set_opponent_policy"):
            self.env.set_opponent_policy(self.opponent_predict)   # another loaded model may have taken it over
        self._last_obs = self.env.reset()           # fresh battles (an evaluation may have run in between)
        self._t0 = time.time()
        start = self._start_timesteps = self.num_timesteps
        self._total_timesteps = total_timesteps     # for status reports (speed, time left)
        while self.num_timesteps < total_timesteps:
            if not self.collect_rollouts(callback):
                break
            stats = self.train()
            if self.n_updates % self.hp["opponent_refresh"] == 0:
                self.opponent.load_state_dict(self.actor.state_dict())
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
        buf = {
            "local": np.zeros((T, E, N, self.local_dim), np.float32),
            "state": np.zeros((T, E, N, self.state_dim), np.float32),
            "action_mask": np.zeros((T, E, N, obs["action_mask"].shape[-1]), np.float32),
            "actions": np.zeros((T, E, N, len(self.head_sizes)), np.int64),
            "logp": np.zeros((T, E, N), np.float32),
            "values": np.zeros((T + 1, E, N), np.float32),
            "rewards": np.zeros((T, E, N), np.float32),
            "dones": np.zeros((T, E), np.float32),
            "alive": np.zeros((T, E, N), np.float32),
            "exists": np.zeros((T, E, N), np.float32),
        }
        for t in range(T):
            actions, logp, values = self._act(obs)
            for k in ("local", "state", "action_mask", "alive", "exists"):
                buf[k][t] = obs[k]
            buf["actions"][t], buf["logp"][t], buf["values"][t] = actions, logp, values

            obs, rewards, dones, infos = self.env.step(actions)
            buf["rewards"][t], buf["dones"][t] = rewards, dones
            self._episodes += [i for i in infos if i]
            self.num_timesteps += E
            if not callback.on_step(locals()):
                self._last_obs = obs
                return False
        buf["values"][T] = self._act(obs)[2]           # bootstrap: the battles are still going
        self._last_obs = obs
        self.buffer = buf
        self._advantages()
        return True

    def _advantages(self):
        """Generalised Advantage Estimation (GAE), per ship slot, backwards through time.

        delta_t     = r_t + gamma * V(s_t+1) * (1 - done_t) - V(s_t)
        advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_t+1
        """
        b, g, lam = self.buffer, self.hp["gamma"], self.hp["gae_lambda"]
        T = b["rewards"].shape[0]
        adv = np.zeros_like(b["rewards"])
        last = np.zeros_like(b["rewards"][0])
        for t in reversed(range(T)):
            not_done = (1.0 - b["dones"][t])[:, None]
            delta = b["rewards"][t] + g * b["values"][t + 1] * not_done - b["values"][t]
            last = delta + g * lam * not_done * last
            adv[t] = last
        b["advantages"] = adv
        b["returns"] = adv + b["values"][:T]

    def train(self) -> dict:
        """The PPO update: n_epochs passes of clipped policy loss + value loss + entropy bonus."""
        hp, b = self.hp, self.buffer
        # every (decision, battle, ship) sample whose slot holds a ship; the actor additionally needs it afloat
        idx = np.argwhere(b["exists"] > 0.5)
        flat = {k: b[k][idx[:, 0], idx[:, 1], idx[:, 2]] for k in
                ("local", "state", "action_mask", "actions", "logp", "advantages", "returns", "values", "alive")}
        live = flat["alive"] > 0.5
        adv = flat["advantages"]
        flat["advantages"] = (adv - adv[live].mean()) / (adv[live].std() + 1e-8)
        if self.hp["normalize_values"]:
            self.value_norm.update(flat["returns"])                               # else it stays mean 0, std 1
        flat["returns_norm"] = self.value_norm.normalize(flat["returns"])        # what the critic learns
        data = {k: self._t(v) if k != "actions" else torch.as_tensor(v, device=self.device) for k, v in flat.items()}

        stats = defaultdict(list)
        n = len(idx)
        for _ in range(hp["n_epochs"]):
            for mb in np.array_split(np.random.permutation(n), hp["n_minibatches"]):
                d = {k: v[mb] for k, v in data.items()}
                actor_w = d["alive"]                      # death masking: sunk ships do not train the actor

                dists = self.actor.distributions(d["local"], d["action_mask"])
                logp = sum(dist.log_prob(d["actions"][:, i]) for i, dist in enumerate(dists))
                entropy = sum(dist.entropy() for dist in dists)
                ratio = torch.exp(logp - d["logp"])
                clipped = torch.clamp(ratio, 1 - hp["clip_range"], 1 + hp["clip_range"])
                surrogate = torch.min(ratio * d["advantages"], clipped * d["advantages"])
                denom = actor_w.sum().clamp_min(1.0)
                policy_loss = -(surrogate * actor_w).sum() / denom
                entropy_mean = (entropy * actor_w).sum() / denom

                value_loss = ((self.critic(d[self.critic_key]) - d["returns_norm"]) ** 2).mean()
                loss = policy_loss - hp["ent_coef"] * entropy_mean + hp["vf_coef"] * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # clipped separately: early on the critic's gradients are large, and clipping both
                # together would shrink the actor's update with them
                nn.utils.clip_grad_norm_(self.actor.parameters(), hp["max_grad_norm"])
                nn.utils.clip_grad_norm_(self.critic.parameters(), hp["max_grad_norm"])
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = logp - d["logp"]
                    stats["approx_kl"].append((((ratio - 1) - log_ratio) * actor_w).sum().item() / denom.item())
                    stats["clip_fraction"].append((((ratio - 1).abs() > hp["clip_range"]).float() * actor_w).sum().item() / denom.item())
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy_mean.item())

        self.n_updates += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        y, v = flat["returns"], flat["values"]
        out["explained_variance"] = float(1 - np.var(y - v) / (np.var(y) + 1e-8))
        return out

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
            rule = [e["won"] for e in eps if e.get("opponent", "rule") == "rule"]
            selfp = [e["won"] for e in eps if e.get("opponent") == "self"]
            if rule:
                row["win_rate"] = float(np.mean(rule))             # against the rule AI
            if selfp:
                row["win_rate_self_play"] = float(np.mean(selfp))  # against the frozen copy
        row.update(train_stats)
        if self.verbose:
            print("-" * 44)
            for k, v in row.items():
                print(f"| {k:<22} | {v:>15.4g} |" if isinstance(v, float) else f"| {k:<22} | {v:>15} |")
            print("-" * 44)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, "progress.csv")
            fields = ["total_timesteps", "updates", "fps", "time_elapsed", "stage", "episodes", "ep_rew_mean",
                      "ep_len_mean", "win_rate", "win_rate_self_play", "policy_loss", "value_loss", "entropy",
                      "approx_kl", "clip_fraction", "explained_variance"]
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)

    # ------------------------------------------------------------------ saving and loading

    def save(self, path: str) -> str:
        path = path if path.endswith(".pt") else path + ".pt"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "hyperparameters": self.hp,
                    "value_norm": self.value_norm.state_dict(),
                    "local_dim": self.local_dim, "state_dim": self.state_dim,
                    "head_sizes": self.head_sizes, "head_names": self.head_names,
                    "num_timesteps": self.num_timesteps, "n_updates": self.n_updates,
                    "stage": getattr(self.env, "stage", 0)}, path)
        return path

    @classmethod
    def load(cls, path: str, env, device: str = "auto", **overrides) -> "MAPPO":
        """A saved model, ready to predict or to continue learning in env (same battle size).
        The environment is moved to the curriculum stage the model was saved on."""
        path = path if path.endswith(".pt") else path + ".pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if (ck["local_dim"], ck["state_dim"], ck["head_sizes"]) != (env.local_dim, env.state_dim, list(env.head_sizes)):
            raise ValueError(f"{path} was trained on a different battle size: observation "
                             f"{ck['local_dim']}/{ck['state_dim']} vs {env.local_dim}/{env.state_dim}")
        hp = {k: v for k, v in ck["hyperparameters"].items() if k != "batch_size"}   # older checkpoints
        model = cls(env, device=device, **{"verbose": 0, **hp, **overrides})
        model.actor.load_state_dict(ck["actor"])
        model.opponent.load_state_dict(ck["actor"])
        model.critic.load_state_dict(ck["critic"])
        if "value_norm" in ck:
            model.value_norm.load_state_dict(ck["value_norm"])
        model.optimizer.load_state_dict(ck["optimizer"])
        model.num_timesteps, model.n_updates = ck["num_timesteps"], ck["n_updates"]
        if hasattr(env, "set_stage"):
            env.set_stage(ck.get("stage", 0))              # carry on with the curriculum where it was
        return model
