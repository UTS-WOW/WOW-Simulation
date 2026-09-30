"""Exports a trained policy for in-game inference (Assets/Scripts/RL/RLPolicy.cs).

File layout (little-endian):
    "NAVP" | int32 version | int32 header_bytes | UTF-8 JSON header | float32 tensors

The header describes the architecture, the observation dims, the action heads, the value
normaliser and where each tensor sits in the blob. The game runs the actor to command ships and,
when present, the critic's win-probability head for the live win meter.

export_parity_case() writes random inputs and PyTorch's outputs so the C# forward pass can be
checked against this one (Naval > RL > Check Policy Parity in the editor, or batch mode).
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch

VERSION = 1


def _tensors(prefix: str, module: torch.nn.Module):
    for name, t in module.state_dict().items():
        yield f"{prefix}.{name}", t.detach().float().cpu().numpy()


def export_policy(actor, critic, spec, cfg, path: str, value_norm=None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tensors, blobs, offset = [], [], 0
    parts = [("actor", actor)]
    if critic is not None and critic.__class__.__name__ == "Critic":
        parts.append(("critic", critic))
    for prefix, mod in parts:
        for name, arr in _tensors(prefix, mod):
            tensors.append({"name": name, "shape": list(arr.shape), "offset": offset})
            blobs.append(arr.astype("<f4").tobytes())
            offset += arr.size
    vn_mean, vn_std = 0.0, 1.0
    if value_norm is not None:
        m, s = value_norm.stats()
        vn_mean, vn_std = float(m), float(s)
    header = {
        "version": VERSION,
        "d": cfg.d_model, "heads": cfg.heads, "layers": cfg.layers, "hidden": cfg.hidden,
        "dims": spec.dims,
        "action_heads": [{"name": h.name, "fixed": h.fixed, "pointer": h.pointer} for h in spec.heads],
        "action_mode": spec.action_mode,
        "decision_period": spec.decision_period,
        "has_critic": len(parts) > 1,
        "vn_mean": vn_mean, "vn_std": vn_std,
        "tensors": tensors,
    }
    hb = json.dumps(header).encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"NAVP")
        f.write(struct.pack("<ii", VERSION, len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)          # the game may be reading the old file; swap atomically
    return path


@torch.no_grad()
def export_parity_case(actor, critic, spec, path: str, seed: int = 0, n_allies=3, n_contacts=4, n_zones=3, n_own=4, n_enemy=4):
    """Unpadded random inputs (exactly the counts the game would feed) and the reference outputs."""
    g = torch.Generator().manual_seed(seed)
    d = spec.dims
    r = lambda *s: torch.randn(*s, generator=g)
    obs = {
        "self": r(1, d["self"]), "allies": r(1, n_allies, d["ally"]), "ally_mask": torch.ones(1, n_allies),
        "contacts": r(1, n_contacts, d["contact"]), "contact_mask": torch.ones(1, n_contacts),
        "zones": r(1, n_zones, d["zone"]), "zone_mask": torch.ones(1, n_zones),
    }
    h = r(1, actor.hidden) * 0.5
    actor.eval()
    logits, h2, attn = actor(obs, h, torch.zeros(1))
    case = {
        "self": obs["self"][0].tolist(), "allies": obs["allies"][0].flatten().tolist(),
        "contacts": obs["contacts"][0].flatten().tolist(), "zones": obs["zones"][0].flatten().tolist(),
        "n_allies": n_allies, "n_contacts": n_contacts, "n_zones": n_zones,
        "hidden_in": h[0].tolist(), "logits": logits[0].tolist(), "hidden_out": h2[0].tolist(),
        "attention_self": attn[0, :, 0, :].mean(0).tolist(),
    }
    if critic is not None and critic.__class__.__name__ == "Critic":
        cobs = {
            "critic_own": r(1, n_own, d["critic_own"]), "own_mask": torch.ones(1, n_own),
            "critic_enemy": r(1, n_enemy, d["critic_enemy"]), "critic_enemy_mask": torch.ones(1, n_enemy),
            "critic_zones": r(1, n_zones, d["critic_zone"]), "critic_zone_mask": torch.ones(1, n_zones),
            "critic_match": r(1, d["critic_match"]),
        }
        critic.eval()
        v, win = critic(cobs, torch.tensor([1]))
        case.update({
            "critic_own": cobs["critic_own"][0].flatten().tolist(), "critic_enemy": cobs["critic_enemy"][0].flatten().tolist(),
            "critic_zones": cobs["critic_zones"][0].flatten().tolist(), "critic_match": cobs["critic_match"][0].tolist(),
            "n_own": n_own, "n_enemy": n_enemy, "agent_index": 1, "value": float(v[0]), "win_logit": float(win[0]),
        })
    with open(path, "w") as f:
        json.dump(case, f)
    return path
