"""The networks: what each ship (captain) and the fleet commander see and decide, and the critic.

Three ideas from the game-playing papers, each one a small module here:

1. EntityEncoder - a battle is a SET of things (ships, contacts, circles, islands), not an image.
   Every thing becomes a token; a small transformer lets the tokens look at each other; pooling
   (mean and max) turns any number of tokens into one fixed-size vector. AlphaStar (StarCraft) uses
   a transformer over units, OpenAI Five (Dota 2) max-pools over units. It is the job the custom
   CNN feature extractor did for Atari frames in Week 10, done for a set instead of a picture.

2. Captain - one network shared by every ship (parameter sharing). It reads only what its ship and
   its team know, keeps a GRU memory (fog of war: an enemy that slipped out of sight is still
   somewhere - OpenAI Five uses an LSTM for the same reason, Atari used frame stacking), and picks
   one option on each of six action heads. "Which target" and "which circle" are pointer heads:
   the options ARE the contact / circle tokens, scored by attention (AlphaStar's pointer network,
   Honor of Kings' target attention), so a battle with 2 or 8 enemies uses the same layer.

3. Commander - every few seconds it looks at the whole fleet (own ships, the contacts the team has
   spotted, the circles, the score - no hidden information) and gives every ship an order:

       0 free      the captain decides alone
       1 engage    fight; no circle duty
       2 + k       hold capture circle k

   This is the macro/micro split of Tencent's Hierarchical Macro Strategy model (Honor of Kings):
   the commander decides WHERE the fleet should be, the captains decide HOW to fight there. The
   order reaches the captain as part of its observation.

The CentralCritic is MAPPO's centralised critic: during training only, it sees the true battle
(enemies included) and predicts the return of every ship, split into reward groups (objective /
combat / outcome - Honor of Kings' multi-head value), plus the commander's value and the chance of
winning.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9
N_ORDER_TYPES = 3                     # free, engage, hold a circle
ORDER_FREE, ORDER_ENGAGE, ORDER_CIRCLE = 0, 1, 2


def layer(n_in: int, n_out: int, gain: float = math.sqrt(2)) -> nn.Linear:
    """Orthogonal initialisation, as in Stable-Baselines3 and the MAPPO paper."""
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


# ---------------------------------------------------------------------------------------------- entity encoder

class AttentionBlock(nn.Module):
    """One transformer block: every token looks at every real token (padding is masked out),
    then a small feed-forward layer. Pre-norm, with residual connections."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=mask < 0.5, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


def pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean and masked max over the tokens: [B, T, d] -> [B, 2d]."""
    m = mask.unsqueeze(-1)
    mean = (tokens * m).sum(1) / m.sum(1).clamp_min(1.0)
    biggest = tokens.masked_fill(m < 0.5, NEG_INF).max(1).values
    biggest = torch.where(m.sum(1) > 0, biggest, torch.zeros_like(biggest))
    return torch.cat([mean, biggest], dim=-1)


class EntityEncoder(nn.Module):
    """Token embedding (one small MLP per kind of thing) + `layers` transformer blocks.

    layers=0 leaves out attention: every token is embedded on its own and only pooled - the
    "deep sets" encoder of OpenAI Five. Handy as an ablation.
    """

    def __init__(self, token_dims: dict[str, int], d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.ModuleDict({name: nn.Sequential(layer(dim, d), nn.ReLU(), layer(d, d))
                                    for name, dim in token_dims.items()})
        self.blocks = nn.ModuleList([AttentionBlock(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, groups: list[tuple[str, torch.Tensor, torch.Tensor]]):
        """groups: (kind, tokens [B, n, dim], mask [B, n]) in a fixed order.
        Returns the encoded tokens [B, T, d] and their mask [B, T]."""
        x = torch.cat([self.embed[name](t) for name, t, _ in groups], dim=1)
        mask = torch.cat([m for _, _, m in groups], dim=1)
        for block in self.blocks:
            x = block(x, mask)
        return self.ln(x), mask


# ---------------------------------------------------------------------------------------------- captain

CAPTAIN_KEYS = ["self", "allies", "ally_mask", "contacts", "contact_mask", "zones", "zone_mask",
                "obstacles", "obstacle_mask"]


def order_features(order: torch.Tensor, n_zones: int) -> tuple[torch.Tensor, torch.Tensor]:
    """An order as captain inputs: a one-hot of its type (added to the self token) and a flag on the
    ordered circle's token. order [B] -> ([B, 3], [B, n_zones, 1])."""
    kind = order.clamp(max=ORDER_CIRCLE)
    onehot = F.one_hot(kind.long(), N_ORDER_TYPES).float()
    circle = order - ORDER_CIRCLE                                                # -1 / -2 for free / engage
    flag = (circle.unsqueeze(-1) == torch.arange(n_zones, device=order.device)).float().unsqueeze(-1)
    return onehot, flag


class Captain(nn.Module):
    """A ship's policy: its own view (+ its order) -> a score for every option of every head."""

    def __init__(self, dims: dict, heads: list[dict], d: int = 64, attn_heads: int = 4, layers: int = 2,
                 hidden: int = 128, recurrent: bool = True):
        super().__init__()
        self.d, self.hidden, self.recurrent = d, hidden, recurrent
        self.head_specs = [(h["name"], h["fixed"], h["pointer"]) for h in heads]
        self.encoder = EntityEncoder({"self": dims["self"] + N_ORDER_TYPES, "ally": dims["ally"],
                                      "contact": dims["contact"], "zone": dims["zone"] + 1,
                                      "obstacle": dims["obstacle"]}, d, attn_heads, layers)
        self.fuse = nn.Sequential(layer(3 * d, d), nn.ReLU())            # own token + mean + max of all
        # memory: a GRU cell carries what the ship has seen from decision to decision
        self.core = nn.GRUCell(d, hidden) if recurrent else nn.Sequential(layer(d, hidden), nn.ReLU())
        # the heads read the memory AND the current view: reacting to what is in front of the ship
        # does not have to go through the memory first
        z = hidden + d
        self.fixed = nn.ModuleList([layer(z, fixed, gain=0.01) for (_, fixed, _) in self.head_specs])
        self.query = nn.ModuleDict({"zones": layer(z, d, gain=0.01), "contacts": layer(z, d, gain=0.01)})
        self.key = nn.ModuleDict({"zones": layer(d, d), "contacts": layer(d, d)})

    def encode(self, obs: dict, order: torch.Tensor):
        """The per-decision part: obs tensors [B, ...] -> (vector [B, d], tokens, where groups start)."""
        onehot, flag = order_features(order, obs["zones"].shape[1])
        B = obs["self"].shape[0]
        groups = [("self", torch.cat([obs["self"], onehot], -1).unsqueeze(1), obs["self"].new_ones(B, 1)),
                  ("ally", obs["allies"], obs["ally_mask"]),
                  ("contact", obs["contacts"], obs["contact_mask"]),
                  ("zone", torch.cat([obs["zones"], flag], -1), obs["zone_mask"]),
                  ("obstacle", obs["obstacles"], obs["obstacle_mask"])]
        tokens, mask = self.encoder(groups)
        x = self.fuse(torch.cat([tokens[:, 0], pool(tokens, mask)], dim=-1))
        c0 = 1 + obs["allies"].shape[1]
        z0 = c0 + obs["contacts"].shape[1]
        pointer_tokens = {"contacts": tokens[:, c0:z0], "zones": tokens[:, z0:z0 + obs["zones"].shape[1]]}
        return x, pointer_tokens

    def heads(self, z: torch.Tensor, pointer_tokens: dict) -> torch.Tensor:
        """Logits of every head, concatenated in the environment's order. z: memory + current view."""
        out = []
        for (name, fixed, pointer), lin in zip(self.head_specs, self.fixed):
            out.append(lin(z))
            if pointer:                                    # one option per contact / circle token
                q = self.query[pointer](z)                                         # [B, d]
                k = self.key[pointer](pointer_tokens[pointer])                     # [B, K, d]
                out.append((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d))
        return torch.cat(out, dim=-1)

    def step(self, obs: dict, order: torch.Tensor, h: torch.Tensor):
        """One decision for B ships. h: memory [B, hidden] (zero it at battle start)."""
        x, ptr = self.encode(obs, order)
        h = self.core(x, h) if self.recurrent else self.core(x)
        return self.heads(torch.cat([h, x], -1), ptr), h

    def sequence(self, obs: dict, order: torch.Tensor, h0: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        """Training pass over chunks of consecutive decisions: obs [T, B, ...] -> logits [T, B, L].
        The memory starts from h0 (what it was during the rollout) and is cleared where starts=1."""
        T, B = order.shape
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
        x, ptr = self.encode(flat, order.reshape(T * B))
        x = x.view(T, B, -1)
        if self.recurrent:
            hs, h = [], h0
            for t in range(T):
                h = self.core(x[t], h * (1.0 - starts[t]).unsqueeze(-1))
                hs.append(h)
            z = torch.stack(hs).reshape(T * B, -1)
        else:
            z = self.core(x.reshape(T * B, -1))
        return self.heads(torch.cat([z, x.reshape(T * B, -1)], -1), ptr).view(T, B, -1)


# ---------------------------------------------------------------------------------------------- commander

class Commander(nn.Module):
    """The fleet's view -> an order for every own ship: free, engage, or hold circle k.

    Its view is what the team legitimately knows: the true state of its own ships, the enemies as
    the team believes them to be (spotted contacts, last known positions), the circles and the score.
    """

    def __init__(self, dims: dict, d: int = 64, attn_heads: int = 4, layers: int = 2):
        super().__init__()
        self.d = d
        self.encoder = EntityEncoder({"match": dims["fleet_match"], "own": dims["critic_own"],
                                      "enemy": dims["critic_enemy"], "zone": dims["critic_zone"]}, d, attn_heads, layers)
        self.fleet = nn.Sequential(layer(2 * d, d), nn.ReLU())
        self.ship = nn.Sequential(layer(2 * d, d), nn.ReLU())
        self.fixed = layer(d, 2, gain=0.01)                  # free, engage
        with torch.no_grad():
            self.fixed.bias[ORDER_FREE] = 2.0                # starts out mostly "free": captains decide
        self.query = layer(d, d, gain=0.01)
        self.key = layer(d, d)

    def forward(self, obs: dict) -> torch.Tensor:
        """obs: fleet tensors [B, ...] -> order logits [B, N, 2 + Z] (illegal ones not masked yet)."""
        own = obs["critic_own"]
        B, N, _ = own.shape
        groups = [("match", obs["fleet_match"].unsqueeze(1), own.new_ones(B, 1)),
                  ("own", own, obs["own_mask"]),
                  ("enemy", obs["fleet_enemy"], obs["critic_enemy_mask"]),
                  ("zone", obs["critic_zones"], obs["critic_zone_mask"])]
        tokens, mask = self.encoder(groups)
        fleet = self.fleet(pool(tokens, mask))                                           # [B, d]
        ships = self.ship(torch.cat([tokens[:, 1:1 + N], fleet.unsqueeze(1).expand(B, N, -1)], -1))
        Z = obs["critic_zones"].shape[1]
        zones = self.key(tokens[:, 1 + 2 * N:1 + 2 * N + Z])                              # [B, Z, d]
        circle = (self.query(ships) @ zones.transpose(1, 2)) / math.sqrt(self.d)         # [B, N, Z]
        return torch.cat([self.fixed(ships), circle], dim=-1)

    @staticmethod
    def order_mask(obs: dict) -> torch.Tensor:
        """[B, N, 2 + Z]: circles that exist; a sunk or empty slot can only be "free"."""
        live = obs["own_mask"] * obs["alive"]
        B, N = live.shape
        zones = obs["critic_zone_mask"].unsqueeze(1).expand(B, N, -1)
        mask = torch.cat([torch.ones(B, N, 2, device=live.device), zones], -1) * live.unsqueeze(-1)
        mask[..., ORDER_FREE] = 1.0
        return mask


# ---------------------------------------------------------------------------------------------- critic

class CentralCritic(nn.Module):
    """MAPPO's centralised critic (training only): the TRUE battle -> for every own ship a value per
    reward group, plus the commander's value and the probability of winning.

    The battle is encoded once; ship i's values come from its own token next to the pooled battle
    ("agent-specific global state"), so all ships of a battle cost one pass.
    """

    def __init__(self, dims: dict, n_groups: int, d: int = 64, attn_heads: int = 4, layers: int = 2):
        super().__init__()
        self.encoder = EntityEncoder({"match": dims["critic_match"], "own": dims["critic_own"],
                                      "enemy": dims["critic_enemy"], "zone": dims["critic_zone"]}, d, attn_heads, layers)
        self.trunk = nn.Sequential(layer(3 * d, d), nn.ReLU())
        self.values = layer(d, n_groups, gain=1.0)
        self.fleet = nn.Sequential(layer(2 * d, d), nn.ReLU())
        self.commander_value = layer(d, 1, gain=1.0)
        self.win = layer(d, 1, gain=1.0)

    def forward(self, obs: dict):
        """obs: battle tensors [B, ...] -> (values [B, N, groups], commander value [B], win logit [B])."""
        own = obs["critic_own"]
        B, N, _ = own.shape
        groups = [("match", obs["critic_match"].unsqueeze(1), own.new_ones(B, 1)),
                  ("own", own, obs["own_mask"]),
                  ("enemy", obs["critic_enemy"], obs["critic_enemy_mask"]),
                  ("zone", obs["critic_zones"], obs["critic_zone_mask"])]
        tokens, mask = self.encoder(groups)
        battle = pool(tokens, mask)                                                       # [B, 2d]
        ships = self.trunk(torch.cat([tokens[:, 1:1 + N], battle.unsqueeze(1).expand(B, N, -1)], -1))
        fleet = self.fleet(battle)
        return self.values(ships), self.commander_value(fleet).squeeze(-1), self.win(fleet).squeeze(-1)


class LocalCritic(nn.Module):
    """IPPO ablation: each ship's critic sees only that ship's own view (what its actor sees)."""

    def __init__(self, dims: dict, n_groups: int, d: int = 64, attn_heads: int = 4, layers: int = 2):
        super().__init__()
        self.encoder = EntityEncoder({"self": dims["self"], "ally": dims["ally"], "contact": dims["contact"],
                                      "zone": dims["zone"], "obstacle": dims["obstacle"]}, d, attn_heads, layers)
        self.trunk = nn.Sequential(layer(3 * d, d), nn.ReLU())
        self.values = layer(d, n_groups, gain=1.0)

    def forward(self, obs: dict):
        """obs: per-ship tensors [B, N, ...] -> (values [B, N, groups], None, None)."""
        B, N = obs["self"].shape[:2]
        flat = {k: obs[k].reshape(B * N, *obs[k].shape[2:]) for k in CAPTAIN_KEYS}
        groups = [("self", flat["self"].unsqueeze(1), flat["self"].new_ones(B * N, 1)),
                  ("ally", flat["allies"], flat["ally_mask"]), ("contact", flat["contacts"], flat["contact_mask"]),
                  ("zone", flat["zones"], flat["zone_mask"]), ("obstacle", flat["obstacles"], flat["obstacle_mask"])]
        tokens, mask = self.encoder(groups)
        x = self.trunk(torch.cat([tokens[:, 0], pool(tokens, mask)], -1))
        return self.values(x).view(B, N, -1), None, None


# ---------------------------------------------------------------------------------------------- distributions

def split_logits(logits: torch.Tensor, mask: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Illegal options get a score of -1e9 (probability 0); one tensor per head."""
    return list(torch.split(logits.masked_fill(mask < 0.5, NEG_INF), sizes, dim=-1))


def sample(logits: torch.Tensor, mask: torch.Tensor, sizes: list[int], deterministic: bool = False):
    """One option per head: (actions [..., heads], summed log-probability [...])."""
    acts, logp = [], 0.0
    for part in split_logits(logits, mask, sizes):
        dist = torch.distributions.Categorical(logits=part)
        a = part.argmax(-1) if deterministic else dist.sample()
        acts.append(a)
        logp = logp + dist.log_prob(a)
    return torch.stack(acts, dim=-1), logp


def log_prob_entropy(logits: torch.Tensor, mask: torch.Tensor, sizes: list[int], actions: torch.Tensor):
    """For the PPO update: log-probability of the taken actions and the entropy, summed over heads."""
    logp, ent = 0.0, 0.0
    for i, part in enumerate(split_logits(logits, mask, sizes)):
        dist = torch.distributions.Categorical(logits=part)
        logp = logp + dist.log_prob(actions[..., i])
        ent = ent + dist.entropy()
    return logp, ent


def kl_to(logits_ref: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """KL(reference || policy), summed over heads - how far the policy has moved from the reference."""
    kl = 0.0
    for ref, cur in zip(split_logits(logits_ref, mask, sizes), split_logits(logits, mask, sizes)):
        p_ref = F.softmax(ref, -1)
        kl = kl + (p_ref * (F.log_softmax(ref, -1) - F.log_softmax(cur, -1))).sum(-1)
    return kl


# ---------------------------------------------------------------------------------------------- the fleet

class Memory:
    """What a fleet policy carries between decisions, for each ship slot of each battle: the
    captains' GRU state and the commander's current orders. Cleared when a battle starts."""

    def __init__(self, n_envs: int, n_agents: int, hidden: int, device):
        self.h = torch.zeros(n_envs, n_agents, hidden, device=device)
        self.orders = torch.zeros(n_envs, n_agents, dtype=torch.long, device=device)

    def reset(self, starts: torch.Tensor) -> None:
        keep = (1.0 - starts).view(-1, 1, 1)
        self.h = self.h * keep
        self.orders = self.orders * (1 - starts.long()).view(-1, 1)

    def take(self, envs) -> "Memory":
        """The memory of some battles only (for an opponent that flies just those)."""
        part = Memory.__new__(Memory)
        part.h, part.orders = self.h[envs].clone(), self.orders[envs].clone()
        return part

    def put(self, envs, part: "Memory") -> None:
        self.h[envs], self.orders[envs] = part.h, part.orders


class FleetPolicy(nn.Module):
    """Everything that ACTS: the captain network (every ship) and, optionally, the commander.
    Training, evaluation, self-play opponents and the imitation-learning teacher all use it."""

    def __init__(self, dims: dict, heads: list[dict], d: int = 64, attn_heads: int = 4, layers: int = 2,
                 hidden: int = 128, recurrent: bool = True, commander: bool = True, commander_period: int = 10):
        super().__init__()
        self.head_sizes = [h["size"] for h in heads]
        self.hidden = hidden
        self.commander_period = commander_period
        self.captain = Captain(dims, heads, d, attn_heads, layers, hidden, recurrent)
        self.commander = Commander(dims, d, attn_heads, layers) if commander else None

    def new_memory(self, n_envs: int, n_agents: int) -> Memory:
        return Memory(n_envs, n_agents, self.hidden, next(self.parameters()).device)

    @torch.no_grad()
    def act(self, obs: dict, memory: Memory, deterministic: bool = False) -> dict:
        """One decision for every ship of every battle. obs: tensors [E, N, ...] / [E, ...].

        Returns actions [E, N, heads] and what training needs to remember (log-probabilities, the
        memory before this step, the orders and the commander's decision)."""
        E, N = obs["alive"].shape
        memory.reset(obs["starts"])
        out = {"h": memory.h.clone()}

        # the commander: every commander_period decisions, in battles where it is on
        decide = torch.zeros(E, dtype=torch.bool, device=memory.h.device)
        out["cmd_logp"] = torch.zeros(E, device=memory.h.device)
        if self.commander is not None:
            decide = (obs["commander_on"] > 0.5) & (obs["t"].long() % self.commander_period == 0)
            if decide.any():
                sub = {k: v[decide] for k, v in obs.items()}
                mask = Commander.order_mask(sub)
                logits = self.commander(sub).masked_fill(mask < 0.5, NEG_INF)
                dist = torch.distributions.Categorical(logits=logits)
                orders = logits.argmax(-1) if deterministic else dist.sample()
                live = sub["own_mask"] * sub["alive"]
                out["cmd_logp"][decide] = (dist.log_prob(orders) * live).sum(-1)
                memory.orders[decide] = orders
            # a battle without its commander (a 1-ship stage) keeps every ship "free"
            memory.orders[obs["commander_on"] < 0.5] = ORDER_FREE
        out["cmd_decide"] = decide
        out["orders"] = memory.orders.clone()

        # the captains: every ship, on its own view and its order
        flat = {k: obs[k].reshape(E * N, *obs[k].shape[2:]) for k in CAPTAIN_KEYS}
        logits, h = self.captain.step(flat, memory.orders.reshape(E * N), memory.h.reshape(E * N, -1))
        memory.h = h.view(E, N, -1)
        actions, logp = sample(logits, obs["action_mask"].reshape(E * N, -1), self.head_sizes, deterministic)
        out["actions"] = actions.view(E, N, -1)
        out["logp"] = logp.view(E, N)
        out["logits"] = logits.view(E, N, -1)
        return out


def to_tensors(obs: dict, device) -> dict:
    return {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device=device) for k, v in obs.items()}
