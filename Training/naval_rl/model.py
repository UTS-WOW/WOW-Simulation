"""Actor and critic networks.

Both are entity encoders: every ship, contact and zone is a token, a small transformer lets the
tokens attend to each other, and the result is independent of how many tokens there are. That is
what lets a policy trained at 3v3 or 6v6 command a 12v12 or 30v30 fleet without retraining.

The actor's discrete choices over entities are pointer heads: the "target" head scores every
contact token against the agent's query, and the zone part of the "move" head scores every zone
token, so the number of options grows with the battle instead of being baked into a layer.

Every operation here (Linear, ReLU, LayerNorm, masked softmax attention, GRUCell) is re-implemented
in Assets/Scripts/RL/RLPolicy.cs for in-game inference; keep the two in step (export.py writes a
parity case the Unity side checks).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import Spec

NEG = -1e9


class MLP(nn.Module):
    def __init__(self, n_in: int, d: int):
        super().__init__()
        self.l1 = nn.Linear(n_in, d)
        self.l2 = nn.Linear(d, d)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, mask):
        # x [B, T, d], mask [B, T] with 1 = real token
        B, T, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).view(B, T, 3, h, d // h).permute(2, 0, 3, 1, 4)     # 3 x [B, h, T, dh]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d // h)
        scores = scores.masked_fill(mask[:, None, None, :] < 0.5, NEG)
        attn = torch.softmax(scores, dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return self.out(y), attn


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 2 * d)
        self.ff2 = nn.Linear(2 * d, d)

    def forward(self, x, mask):
        a, attn = self.attn(self.ln1(x), mask)
        x = x + a
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x, attn


def masked_mean(x, mask):
    m = mask.unsqueeze(-1)
    return (x * m).sum(dim=-2) / m.sum(dim=-2).clamp_min(1.0)


class EntityEncoder(nn.Module):
    """Token embedding + transformer. Returns the final tokens and the last layer's attention."""

    def __init__(self, token_dims: list[int], d: int, heads: int, layers: int):
        super().__init__()
        self.embeds = nn.ModuleList([MLP(n, d) for n in token_dims])
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, groups: list[torch.Tensor], masks: list[torch.Tensor]):
        # groups[i] is [B, n_i, dim_i]; masks[i] is [B, n_i]
        x = torch.cat([emb(g) for emb, g in zip(self.embeds, groups)], dim=1)
        mask = torch.cat(masks, dim=1)
        attn = None
        for blk in self.blocks:
            x, attn = blk(x, mask)
        return self.ln_f(x), mask, attn


# ---------------------------------------------------------------------------------------------- actor

class Actor(nn.Module):
    """obs -> (per-head logits, new GRU state). Shared by every ship of every class."""

    def __init__(self, spec: Spec, d: int = 128, heads: int = 4, layers: int = 2, hidden: int = 128):
        super().__init__()
        self.d, self.hidden = d, hidden
        self.head_specs = [(h.name, h.fixed, h.pointer) for h in spec.heads]
        dm = spec.dims
        self.encoder = EntityEncoder([dm["self"], dm["ally"], dm["contact"], dm["zone"]], d, heads, layers)
        self.fuse = nn.Linear(2 * d, d)
        self.gru = nn.GRUCell(d, hidden)
        self.fixed = nn.ModuleList([nn.Linear(hidden, fixed) for (_, fixed, _) in self.head_specs])
        self.ptr_q = nn.ModuleDict({"zones": nn.Linear(hidden, d), "contacts": nn.Linear(hidden, d)})
        self.ptr_k = nn.ModuleDict({"zones": nn.Linear(d, d), "contacts": nn.Linear(d, d)})

    def encode(self, obs: dict):
        """The per-step part (no recurrence): obs tensors with any leading batch shape flattened to B."""
        B = obs["self"].shape[0]
        ones = obs["self"].new_ones(B, 1)
        tokens, mask, attn = self.encoder(
            [obs["self"].unsqueeze(1), obs["allies"], obs["contacts"], obs["zones"]],
            [ones, obs["ally_mask"], obs["contact_mask"], obs["zone_mask"]],
        )
        x = F.relu(self.fuse(torch.cat([tokens[:, 0], masked_mean(tokens, mask)], dim=-1)))
        return x, tokens, attn

    def heads(self, z, tokens, n_allies: int, n_contacts: int):
        """Logits for every head, concatenated in spec order."""
        c0 = 1 + n_allies
        z0 = c0 + n_contacts
        groups = {"contacts": tokens[:, c0:z0], "zones": tokens[:, z0:]}
        out = []
        for (name, fixed, pointer), lin in zip(self.head_specs, self.fixed):
            out.append(lin(z))
            if pointer:
                q = self.ptr_q[pointer](z)                                   # [B, d]
                k = self.ptr_k[pointer](groups[pointer])                     # [B, K, d]
                out.append((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d))
        return torch.cat(out, dim=-1)

    def forward(self, obs: dict, h, starts):
        """One step. starts[b] = 1 resets that row's memory (new episode)."""
        x, tokens, attn = self.encode(obs)
        h = h * (1.0 - starts).unsqueeze(-1)
        h = self.gru(x, h)
        logits = self.heads(h, tokens, obs["allies"].shape[1], obs["contacts"].shape[1])
        return logits, h, attn

    def forward_sequence(self, obs: dict, h0, starts):
        """Chunked training pass. obs tensors are [T, B, ...]; returns logits [T, B, L]."""
        T, B = obs["self"].shape[:2]
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
        x, tokens, _ = self.encode(flat)
        x = x.view(T, B, -1)
        hs = []
        h = h0
        for t in range(T):
            h = h * (1.0 - starts[t]).unsqueeze(-1)
            h = self.gru(x[t], h)
            hs.append(h)
        hseq = torch.stack(hs).reshape(T * B, -1)
        logits = self.heads(hseq, tokens, flat["allies"].shape[1], flat["contacts"].shape[1])
        return logits.view(T, B, -1)

    def initial_state(self, n: int, device=None):
        return torch.zeros(n, self.hidden, device=device)


# ---------------------------------------------------------------------------------------------- critics

class Critic(nn.Module):
    """Centralised critic: the team's true picture plus "which ship am I" -> value and P(win)."""

    def __init__(self, spec: Spec, d: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        dm = spec.dims
        self.encoder = EntityEncoder(
            [dm["critic_match"], dm["critic_own"] + 1, dm["critic_enemy"], dm["critic_zone"]], d, heads, layers)
        self.fuse = nn.Linear(2 * d, d)
        self.value = nn.Linear(d, 1)
        self.win = nn.Linear(d, 1)

    def forward(self, cobs: dict, agent_index):
        """cobs tensors are per team row [B, ...]; agent_index [B] picks the "me" token."""
        own = cobs["critic_own"]
        B, N, _ = own.shape
        is_me = F.one_hot(agent_index.long(), N).to(own.dtype).unsqueeze(-1)
        ones = own.new_ones(B, 1)
        tokens, mask, _ = self.encoder(
            [cobs["critic_match"].unsqueeze(1), torch.cat([own, is_me], dim=-1), cobs["critic_enemy"], cobs["critic_zones"]],
            [ones, cobs["own_mask"], cobs["critic_enemy_mask"], cobs["critic_zone_mask"]],
        )
        me = tokens[torch.arange(B, device=own.device), 1 + agent_index.long()]
        x = F.relu(self.fuse(torch.cat([me, masked_mean(tokens, mask)], dim=-1)))
        return self.value(x).squeeze(-1), self.win(x).squeeze(-1)


class LocalCritic(nn.Module):
    """IPPO ablation: the critic sees only what the actor sees."""

    def __init__(self, spec: Spec, d: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        dm = spec.dims
        self.encoder = EntityEncoder([dm["self"], dm["ally"], dm["contact"], dm["zone"]], d, heads, layers)
        self.fuse = nn.Linear(2 * d, d)
        self.value = nn.Linear(d, 1)
        self.win = nn.Linear(d, 1)

    def forward(self, obs: dict):
        B = obs["self"].shape[0]
        tokens, mask, _ = self.encoder(
            [obs["self"].unsqueeze(1), obs["allies"], obs["contacts"], obs["zones"]],
            [obs["self"].new_ones(B, 1), obs["ally_mask"], obs["contact_mask"], obs["zone_mask"]],
        )
        x = F.relu(self.fuse(torch.cat([tokens[:, 0], masked_mean(tokens, mask)], dim=-1)))
        return self.value(x).squeeze(-1), self.win(x).squeeze(-1)


# ---------------------------------------------------------------------------------------------- distributions

def split_masked(logits, mask, sizes):
    logits = logits.masked_fill(mask < 0.5, NEG)
    return torch.split(logits, sizes, dim=-1)


def sample_actions(logits, mask, sizes, deterministic: bool = False):
    parts = split_masked(logits, mask, sizes)
    acts, logp, ent = [], 0.0, 0.0
    for p in parts:
        dist = torch.distributions.Categorical(logits=p)
        a = p.argmax(-1) if deterministic else dist.sample()
        acts.append(a)
        logp = logp + dist.log_prob(a)
        ent = ent + dist.entropy()
    return torch.stack(acts, dim=-1), logp, ent


def evaluate_actions(logits, mask, sizes, actions):
    parts = split_masked(logits, mask, sizes)
    logp, ent = 0.0, 0.0
    for i, p in enumerate(parts):
        dist = torch.distributions.Categorical(logits=p)
        logp = logp + dist.log_prob(actions[..., i])
        ent = ent + dist.entropy()
    return logp, ent


class ValueNorm(nn.Module):
    """Running mean/variance of value targets (MAPPO's value normalisation)."""

    def __init__(self, beta: float = 0.99999, eps: float = 1e-5):
        super().__init__()
        self.beta, self.eps = beta, eps
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("mean_sq", torch.zeros(()))
        self.register_buffer("debias", torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        x = x.detach().float()
        self.mean.mul_(self.beta).add_(x.mean() * (1 - self.beta))
        self.mean_sq.mul_(self.beta).add_((x ** 2).mean() * (1 - self.beta))
        self.debias.mul_(self.beta).add_(1 - self.beta)

    def stats(self):
        mean = self.mean / self.debias.clamp_min(self.eps)
        mean_sq = self.mean_sq / self.debias.clamp_min(self.eps)
        var = (mean_sq - mean ** 2).clamp_min(1e-2)
        return mean, var.sqrt()

    def normalize(self, x):
        m, s = self.stats()
        return (x - m) / s

    def denormalize(self, x):
        m, s = self.stats()
        return x * s + m
