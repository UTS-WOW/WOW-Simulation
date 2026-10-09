"""Exports a trained fleet to the Unity game, which runs it in plain C# (Assets/Scripts/RL/RLPolicy.cs).

    python export_simple.py --model runs_simple/fleet/models/model_best.pt
        -> Assets/StreamingAssets/RL/naval_policy.bin (the previous file is kept as naval_policy.bin.bak)

Then in Unity: Play, and on the setup screen set "Enemy AI: Learned" (or "Your fleet: Learned"); the
game's RLPolicyDriver flies those ships with the exported captains and, in fleets of two or more,
the exported commander. No Python is needed while the game runs.

File layout (little-endian), version 2 of the format naval_rl/export.py introduced:
    "NAVP" | int32 version | int32 header_bytes | UTF-8 JSON header | float32 tensors

The tensors are renamed to the names RLPolicy.cs reads (actor.encoder.embeds.0.l1, ...blocks.0.attn.qkv,
actor.gru, actor.fixed.<head>, actor.ptr_q.zones, commander.* ...). export_parity_case() writes random
inputs and PyTorch's outputs, so tests/test_policy_parity.py can check the C# forward pass against
this one.
"""

from __future__ import annotations

import json
import os
import shutil
import struct

import numpy as np
import torch

from .networks import Captain, Commander

VERSION = 2
TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_POLICY = os.path.join(TRAINING_DIR, "..", "Assets", "StreamingAssets", "RL", "naval_policy.bin")
GAME_STAGES = os.path.join(TRAINING_DIR, "..", "Assets", "StreamingAssets", "RL", "stages.json")

# What the commander may not see (simple_mappo/env.py zeroes the same): the true-state columns of
# critic_enemy and the enemy's true strength in critic_match. Used when a checkpoint does not record them.
DEFAULT_ENEMY_PRIVILEGED = (1, 45)
PRIVILEGED_MATCH = ("their_fleet_strength", "their_alive_frac_true")
DEFAULT_MATCH_HIDDEN_FROM_END = 2             # critic_match ends with my/their strength and their true alive share

CAPTAIN_TOKENS = ["self", "ally", "contact", "zone", "obstacle"]
COMMANDER_TOKENS = ["match", "own", "enemy", "zone"]


def _encoder_names(prefix: str, key: str, tokens: list[str]) -> str | None:
    """encoder.* of networks.EntityEncoder -> the name RLPolicy.cs reads."""
    parts = key.split(".")
    if parts[1] == "embed":                                  # encoder.embed.<token>.<0|2>.<w|b>
        layer = {"0": "l1", "2": "l2"}[parts[3]]
        return f"{prefix}.encoder.embeds.{tokens.index(parts[2])}.{layer}.{parts[4]}"
    if parts[1] == "ln":
        return f"{prefix}.encoder.ln_f.{parts[2]}"
    if parts[1] == "blocks":                                 # encoder.blocks.<i>.<...>
        i, rest = parts[2], parts[3:]
        if rest[0] == "attn":
            if rest[1] == "in_proj_weight":
                return f"{prefix}.encoder.blocks.{i}.attn.qkv.weight"
            if rest[1] == "in_proj_bias":
                return f"{prefix}.encoder.blocks.{i}.attn.qkv.bias"
            return f"{prefix}.encoder.blocks.{i}.attn.out.{rest[-1]}"          # out_proj.<w|b>
        if rest[0] == "ff":
            return f"{prefix}.encoder.blocks.{i}.{ {'0': 'ff1', '2': 'ff2'}[rest[1]] }.{rest[2]}"
        return f"{prefix}.encoder.blocks.{i}.{rest[0]}.{rest[1]}"           # ln1 / ln2
    return None


def captain_tensors(captain: Captain):
    if not captain.recurrent:
        raise ValueError("the game's C# policy needs the GRU captain (recurrent=True)")
    for key, t in captain.state_dict().items():
        p = key.split(".")
        if p[0] == "encoder":
            name = _encoder_names("actor", key, CAPTAIN_TOKENS)
        elif p[0] == "fuse":                                 # fuse.0.<w|b>
            name = f"actor.fuse.{p[2]}"
        elif p[0] == "core":                                 # core.weight_ih ...
            name = f"actor.gru.{p[1]}"
        elif p[0] == "fixed":
            name = f"actor.fixed.{p[1]}.{p[2]}"
        elif p[0] == "query":
            name = f"actor.ptr_q.{p[1]}.{p[2]}"
        elif p[0] == "key":
            name = f"actor.ptr_k.{p[1]}.{p[2]}"
        else:
            name = None
        if name is None:
            raise KeyError(f"captain tensor {key} has no place in the C# policy")
        yield name, t


def commander_tensors(commander: Commander):
    for key, t in commander.state_dict().items():
        p = key.split(".")
        if p[0] == "encoder":
            name = _encoder_names("commander", key, COMMANDER_TOKENS)
        elif p[0] in ("fleet", "ship"):                      # fleet.0.<w|b>
            name = f"commander.{p[0]}.{p[2]}"
        else:                                                # fixed / query / key
            name = f"commander.{p[0]}.{p[1]}"
        if name is None:
            raise KeyError(f"commander tensor {key} has no place in the C# policy")
        yield name, t


def _load(source):
    """A MAPPO model, or the path of a MAPPO / imitation checkpoint -> what the export needs."""
    if not isinstance(source, str):                          # a MAPPO model in memory
        env = source.env
        return {"captain": source.policy.captain, "commander": source.policy.commander, "dims": source.dims,
                "heads": source.heads, "arch": source.arch, "commander_period": source.hp["commander_period"],
                "decision_period": getattr(env, "decision_period", 1.0),
                "enemy_privileged": tuple(getattr(getattr(env, "spec", None), "enemy_privileged", DEFAULT_ENEMY_PRIVILEGED)),
                "match_hidden": _match_hidden(env), "about": {"timesteps": source.num_timesteps,
                                                              "stage": getattr(env, "stage", 0)}}
    ck = torch.load(source, map_location="cpu", weights_only=False)
    dims, heads = ck["dims"], ck["heads"]
    if "captain" in ck:                                      # an imitation checkpoint: the clone, no commander
        arch = ck["arch"]
        captain = Captain(dims, heads, arch["d"], arch["attn_heads"], arch["layers"], arch["hidden"], arch["recurrent"])
        captain.load_state_dict(ck["captain"])
        commander, period, about = None, 10, {"imitation": True}
    else:
        from .networks import FleetPolicy
        hp = ck["hyperparameters"]
        arch = dict(d=hp["d_model"], attn_heads=hp["attn_heads"], layers=hp["attn_layers"], hidden=hp["hidden_size"],
                    recurrent=hp["recurrent"])
        policy = FleetPolicy(dims, heads, commander=hp["commander"], commander_period=hp["commander_period"], **arch)
        policy.load_state_dict(ck["policy"])
        captain, commander, period = policy.captain, policy.commander, hp["commander_period"]
        about = {"timesteps": ck["num_timesteps"], "stage": ck.get("stage", 0)}
    match_hidden = ck.get("match_hidden")
    if match_hidden is None:
        m = dims["critic_match"]
        match_hidden = [0.0] * (m - DEFAULT_MATCH_HIDDEN_FROM_END) + [1.0] * DEFAULT_MATCH_HIDDEN_FROM_END
    return {"captain": captain, "commander": commander, "dims": dims, "heads": heads, "arch": arch,
            "commander_period": period, "decision_period": ck.get("decision_period", 1.0),
            "enemy_privileged": tuple(ck.get("enemy_privileged", DEFAULT_ENEMY_PRIVILEGED)),
            "match_hidden": match_hidden, "about": about}


def _match_hidden(env) -> list[float]:
    keep = getattr(env, "_fleet_match_keep", None)
    if keep is None:
        m = env.dims["critic_match"]
        return [0.0] * (m - DEFAULT_MATCH_HIDDEN_FROM_END) + [1.0] * DEFAULT_MATCH_HIDDEN_FROM_END
    return [float(1.0 - k) for k in keep]


def export_policy(source, path: str = GAME_POLICY, commander: bool = True, backup: bool = True,
                  max_obstacles: int = 4) -> str:
    """Writes the fleet (captain + commander) as the game's policy file. source: a MAPPO model, or a
    checkpoint path (model_*.pt from training, or imitation.pt for the clone alone). Returns the path."""
    m = _load(source)
    tensors, blobs, offset = [], [], 0
    parts = list(captain_tensors(m["captain"]))
    with_commander = commander and m["commander"] is not None
    if with_commander:
        parts += list(commander_tensors(m["commander"]))
    for name, t in parts:
        arr = t.detach().float().cpu().numpy()
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset})
        blobs.append(arr.astype("<f4").tobytes())
        offset += arr.size
    dims = {k: int(v) for k, v in m["dims"].items() if k != "fleet_match"}
    header = {
        "version": VERSION, "method": "commander-mappo",
        "d": m["arch"]["d"], "heads": m["arch"]["attn_heads"], "layers": m["arch"]["layers"], "hidden": m["arch"]["hidden"],
        "dims": dims, "max_obstacles": max_obstacles,
        "action_heads": [{"name": h["name"], "fixed": h["fixed"], "pointer": h["pointer"]} for h in m["heads"]],
        "action_mode": "intent", "decision_period": float(m["decision_period"]), "has_critic": False,
        "pool": "mean_max", "head_input": "memory_and_view", "order_features": 3,
        "has_commander": with_commander, "commander_period": int(m["commander_period"]),
        "enemy_privileged": [int(x) for x in m["enemy_privileged"]], "match_hidden": [float(x) for x in m["match_hidden"]],
        "trained": m["about"], "tensors": tensors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    hb = json.dumps(header).encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"NAVP")
        f.write(struct.pack("<ii", VERSION, len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)                  # the game may be reading the old file: swap atomically
    return path


@torch.no_grad()
def export_parity_case(source, path: str, n_allies=2, n_contacts=3, n_zones=2, n_obstacles=2, order=0,
                       n_own=3, n_enemy=3, seed=0) -> str:
    """Random inputs and PyTorch's outputs for one captain decision (two steps, so the GRU memory
    is checked too) and one commander decision - for the C# parity check."""
    m = _load(source)
    cap, com, d = m["captain"].eval(), m["commander"], m["dims"]
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g) * 0.5
    A, C, Z, O = 7, 8, 5, 4                          # the training paddings
    obs = {"self": r(1, d["self"]), "allies": r(1, A, d["ally"]), "contacts": r(1, C, d["contact"]),
           "zones": r(1, Z, d["zone"]), "obstacles": r(1, O, d["obstacle"])}
    for key, n, size in (("ally_mask", n_allies, A), ("contact_mask", n_contacts, C), ("zone_mask", n_zones, Z),
                         ("obstacle_mask", n_obstacles, O)):
        obs[key] = (torch.arange(size) < n).float().unsqueeze(0)
    h = r(1, cap.hidden)
    out = {"n_allies": n_allies, "n_contacts": n_contacts, "n_zones": n_zones, "n_obstacles": n_obstacles,
           "order": order, "self": obs["self"][0].tolist(),
           "allies": obs["allies"][0, :n_allies].flatten().tolist(), "contacts": obs["contacts"][0, :n_contacts].flatten().tolist(),
           "zones": obs["zones"][0, :n_zones].flatten().tolist(), "obstacles": obs["obstacles"][0, :n_obstacles].flatten().tolist(),
           "hidden_in": h[0].tolist()}
    order_t = torch.tensor([order])
    logits, h1 = cap.step(obs, order_t, h)
    logits2, h2 = cap.step(obs, order_t, h1)                 # a second decision from the new memory
    out["logits"] = _compact(logits[0], m["heads"], n_contacts, n_zones)
    out["logits_step2"] = _compact(logits2[0], m["heads"], n_contacts, n_zones)
    out["hidden_out"] = h1[0].tolist()
    out["hidden_out_step2"] = h2[0].tolist()

    if com is not None:
        com = com.eval()
        N = 8
        own, enemy = r(1, N, d["critic_own"]), r(1, N, d["critic_enemy"])
        zones, match = r(1, Z, d["critic_zone"]), r(1, d["critic_match"])
        p0, p1 = m["enemy_privileged"]
        fleet_enemy = enemy.clone()
        fleet_enemy[..., p0:p1] = 0.0
        fleet_match = match * (1.0 - torch.tensor(m["match_hidden"]))
        fobs = {"critic_own": own, "own_mask": (torch.arange(N) < n_own).float().unsqueeze(0),
                "fleet_enemy": fleet_enemy, "critic_enemy_mask": (torch.arange(N) < n_enemy).float().unsqueeze(0),
                "critic_zones": zones, "critic_zone_mask": (torch.arange(Z) < n_zones).float().unsqueeze(0),
                "fleet_match": fleet_match}
        orders = com(fobs)[0, :n_own, :2 + n_zones]
        out.update({"n_own": n_own, "n_enemy": n_enemy, "critic_match": match[0].tolist(),
                    "critic_own": own[0, :n_own].flatten().tolist(), "critic_enemy": enemy[0, :n_enemy].flatten().tolist(),
                    "critic_zones": zones[0, :n_zones].flatten().tolist(), "order_logits": orders.flatten().tolist()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return path


def _mix_text(mix: dict, level: str) -> str:
    """{"rule": 0.7, "mirror": 0.3} -> "rule AI (Veteran) 70% · itself (both sides learn) 30%"."""
    names = {"rule": f"rule AI ({level})", "mirror": "itself (both sides learn)", "latest": "a frozen copy",
             "past": "its past selves"}
    total = sum(mix.values()) or 1.0
    return " · ".join(f"{names.get(k, k)} {v / total:.0%}" for k, v in mix.items())


def export_stages(path: str = GAME_STAGES, curriculum="default") -> str:
    """Writes the curriculum for the game's TRAINING STAGES screen (Assets/Scripts/RL/RLStages.cs):
    every stage's battle (its scenario file embedded, or the generator's settings), its opponent and
    what passes it - so the game starts a stage exactly the way training does."""
    from .curriculum import RANDOM_CHOICES, load_curriculum

    def pick(v, default):              # "random" or a list -> -1 (the game picks one); otherwise the value
        v = default if v is None else v
        return -1 if v == "random" or isinstance(v, (list, tuple)) else int(v)

    stages = []
    for i, st in enumerate(load_curriculum(curriculum)):
        goal = st.get("pass_if", "won")
        rate, window = st.get("promote_win_rate", st.get("stop_win_rate")), st.get("window", st.get("stop_window"))
        passed = {"won": "won", "sunk": "every enemy ship sunk", "captured": "a circle captured"}[goal]
        if st.get("within"):
            passed += f" within {int(st['within'])} s"
        opponent = st.get("opponent", "rule")
        difficulty = int(st.get("difficulty", 0))
        level = ["Recruit", "Veteran", "Elite"][min(max(difficulty, 0), 2)]
        out = {"index": i, "name": st["name"], "battle": st.get("battle", ""), "description": st.get("description", ""),
               "opponent": opponent, "difficulty": difficulty,
               "opponent_text": {"passive": "a passive target", "rule": f"rule AI ({level})"}.get(opponent)
                                or _mix_text(st.get("opponents") or {"rule": 1.0}, level),
               "pass_text": f"passed: {passed}" + (f" in {rate:.0%} of the last {window}" if rate else ""),
               "commander": bool(st.get("commander", False)), "time_limit": float(st.get("time_limit") or 0)}
        if "procedural" in st:
            p = st["procedural"]
            ships = p.get("ships", [6, 6])
            lo, hi = (ships[0], ships[-1]) if isinstance(ships, (list, tuple)) else (ships, ships)
            radius = p.get("capture_radius", 0)
            r_lo, r_hi = (radius[0], radius[-1]) if isinstance(radius, (list, tuple)) else (radius, radius)
            out["procedural"] = {"mode": pick(p.get("mode"), 0), "preset": pick(p.get("preset"), 0),
                                 "density": pick(p.get("density"), 1), "weather": pick(p.get("weather"), 0),
                                 "ships_min": int(lo), "ships_max": int(hi), "capture_radius_min": float(r_lo),
                                 "capture_radius_max": float(r_hi), "modes": RANDOM_CHOICES["mode"]}
            out["time_limit"] = float(p.get("time_limit") or out["time_limit"])
        else:
            scenario = st["scenario"][0] if isinstance(st["scenario"], list) else st["scenario"]
            with open(os.path.join(TRAINING_DIR, scenario)) as f:
                out["scenario_json"] = json.dumps(json.load(f))
        stages.append(out)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"version": 1, "stages": stages}, f, indent=1)
    return path


def _compact(logits: torch.Tensor, heads: list[dict], n_contacts: int, n_zones: int) -> list[float]:
    """The padded logits of all heads -> the game's compact order: each head's fixed options, then
    one option per contact / circle that exists."""
    out, off = [], 0
    for h in heads:
        n = h["fixed"] + (n_zones if h["pointer"] == "zones" else n_contacts if h["pointer"] == "contacts" else 0)
        out += logits[off:off + n].tolist()
        off += h["size"]
    return out
