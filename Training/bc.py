"""Behaviour cloning: teach the MAPPO actor to imitate the rule-based fleet AI, as a starting point
for MAPPO fine-tuning (the same recipe as AlphaStar: supervised first, then reinforcement learning).

    python bc.py collect --battles 120            # rule AI vs rule AI, both fleets recorded
    python bc.py train                            # fit the actor to the recorded decisions
    python train.py --init-from runs/bc/bc.pt ... # then improve it with MAPPO

collect: plays battles where both fleets are the Elite rule AI and the environment labels every
         ship's decisions in the policy's six action heads (Assets/Scripts/RL/RLExpertLabels.cs).
         Saves runs/bc/demos.npz: one contiguous sequence per ship per battle.
train:   cross-entropy on each head, masked exactly as in play, over 64-step sequences so the GRU
         learns to use its memory. Reports per-head accuracy on held-out battles and writes
         runs/bc/bc.pt, a checkpoint train.py can start from.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from naval_rl.config import Config
from naval_rl.env import init_message, make_workers
from naval_rl.league import League
from naval_rl.mappo import ACTOR_KEYS
from naval_rl.model import Actor, Critic, ValueNorm
from naval_rl.spec import Spec

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "bc")
KEYS = ACTOR_KEYS + ["action_mask"]


# ---------------------------------------------------------------------------------------- collect

def collect(a):
    os.makedirs(OUT, exist_ok=True)
    cfg = Config()
    init = init_message(cfg.max_team, cfg.max_allies, cfg.max_contacts, cfg.max_zones, "intent",
                        cfg.decision_period, cfg.sim_dt, cfg.reflexes)
    workers = make_workers(a.workers, init, os.path.abspath(a.unity_binary), a.base_port,
                           log_dir=os.path.join(OUT, "unity_logs"))
    spec = workers[0].spec
    with open(os.path.join(OUT, "spec.json"), "w") as f:
        json.dump(spec.to_json(), f)

    stages = [int(s) for s in a.stages.split(",")]
    weights = np.array([float(w) for w in a.weights.split(",")]) if a.weights else np.ones(len(stages))
    weights = weights / weights.sum()
    rng = np.random.default_rng(a.seed)
    cfg.selfplay_from_stage = 99
    league = League(cfg, HERE, seed=a.seed)
    N, H = spec.max_team, len(spec.heads)

    # per worker, per team: the running sequence of this battle
    seqs = [[{"obs": {k: [] for k in KEYS}, "labels": [], "valid": []} for _ in range(2)] for _ in workers]
    finished = []            # dicts: obs {k: [T, N, ...]}, labels [T, N, H], valid [T, N]
    battles_started = 0
    t0 = time.time()

    def start(w):
        nonlocal battles_started
        league.stage = int(rng.choice(stages, p=weights))
        plan = league.sample(w)
        msg = dict(plan.reset, learned_teams=0, record_teams=3, opponent_difficulty=a.difficulty)
        workers[w].reset_async(msg)
        battles_started += 1

    for w in range(len(workers)):
        start(w)
    obs = [wk.recv() for wk in workers]
    zero = np.zeros((2, N, H), dtype=np.int32)
    active = set(range(len(workers)))
    while active:
        for w in active:
            workers[w].step_async(zero)
        for w in list(active):
            o2 = workers[w].recv()
            o = obs[w]
            for t in range(2):
                sq = seqs[w][t]
                for k in KEYS:
                    sq["obs"][k].append(o.arrays[k][t].astype(np.float16))
                sq["labels"].append(o2.arrays["expert_actions"][t].astype(np.int8))
                sq["valid"].append(o2.arrays["expert_valid"][t].astype(np.bool_))
            if o2.terminal:
                for t in range(2):
                    sq = seqs[w][t]
                    if len(sq["labels"]) > 8:
                        finished.append({"obs": {k: np.stack(v) for k, v in sq["obs"].items()},
                                         "labels": np.stack(sq["labels"]), "valid": np.stack(sq["valid"])})
                    seqs[w][t] = {"obs": {k: [] for k in KEYS}, "labels": [], "valid": []}
                if battles_started < a.battles:
                    start(w)
                    o2 = workers[w].recv()
                else:
                    active.discard(w)
                done = len(finished) // 2
                print(f"  battle {done}/{a.battles} finished ({o2.reason or 'reset'}), "
                      f"{sum(len(x['labels']) for x in finished)} team-steps, {(time.time() - t0) / 60:.1f} min", flush=True)
            obs[w] = o2
    for wk in workers:
        wk.close()

    # flatten to per-ship sequences to keep only slots that held a ship
    out = {k: [] for k in KEYS}
    labels, valid, starts = [], [], []
    for ep in finished:
        n_ship = int(ep["valid"].any(axis=0).sum())
        for i in range(ep["labels"].shape[1]):
            if not ep["valid"][:, i].any():
                continue
            for k in KEYS:
                out[k].append(ep["obs"][k][:, i])
            labels.append(ep["labels"][:, i])
            valid.append(ep["valid"][:, i])
            s = np.zeros(len(ep["labels"]), dtype=np.bool_)
            s[0] = True
            starts.append(s)
    lengths = np.array([len(l) for l in labels])
    path = a.out
    np.savez_compressed(path, lengths=lengths, labels=np.concatenate(labels), valid=np.concatenate(valid),
                        starts=np.concatenate(starts), **{k: np.concatenate(v) for k, v in out.items()})
    print(f"saved {path}: {len(lengths)} ship sequences, {lengths.sum()} labelled decisions "
          f"({np.concatenate(valid).mean():.0%} valid)")


# ---------------------------------------------------------------------------------------- train

def chunks_of(lengths, L, rng, held_out_frac):
    """(start, length) windows inside each sequence, split into train / held-out by sequence."""
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    order = rng.permutation(len(lengths))
    n_hold = max(1, int(len(lengths) * held_out_frac))
    split = {"hold": order[:n_hold], "train": order[n_hold:]}
    out = {}
    for name, idx in split.items():
        w = []
        for i in idx:
            for s in range(0, lengths[i], L):
                w.append((offsets[i] + s, min(L, lengths[i] - s)))
        out[name] = w
    return out


def batch(data, windows, L, device):
    B = len(windows)
    idx = np.zeros((L, B), dtype=np.int64)
    pad = np.zeros((L, B), dtype=np.float32)
    for b, (s, n) in enumerate(windows):
        idx[:n, b] = np.arange(s, s + n)
        idx[n:, b] = s + n - 1
        pad[:n, b] = 1.0
    obs = {k: torch.as_tensor(data[k][idx].astype(np.float32), device=device) for k in ACTOR_KEYS}
    mask = torch.as_tensor(data["action_mask"][idx].astype(np.float32), device=device)
    labels = torch.as_tensor(data["labels"][idx].astype(np.int64), device=device)
    valid = torch.as_tensor(data["valid"][idx].astype(np.float32), device=device) * torch.as_tensor(pad, device=device)
    starts = torch.zeros(L, B, device=device)
    starts[0] = 1.0
    return obs, mask, labels, valid, starts


def head_losses(logits, mask, labels, valid, sizes):
    logits = logits.masked_fill(mask < 0.5, -1e9)
    losses, correct = [], []
    off = 0
    for h, s in enumerate(sizes):
        lg = logits[..., off:off + s]
        ce = F.cross_entropy(lg.reshape(-1, s), labels[..., h].reshape(-1), reduction="none").view(valid.shape)
        losses.append((ce * valid).sum() / valid.sum().clamp_min(1.0))
        correct.append(((lg.argmax(-1) == labels[..., h]).float() * valid).sum() / valid.sum().clamp_min(1.0))
        off += s
    return losses, correct


def train(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(OUT, "spec.json")) as f:
        spec = Spec.from_json(json.load(f))
    raw = np.load(a.demos)
    data = {k: raw[k] for k in raw.files}
    cfg = Config()
    rng = np.random.default_rng(a.seed)
    L = a.chunk
    split = chunks_of(data["lengths"], L, rng, 0.1)
    print(f"{int(data['valid'].sum())} labelled decisions; {len(split['train'])} training windows, "
          f"{len(split['hold'])} held-out windows")

    actor = Actor(spec, cfg.d_model, cfg.heads, cfg.layers, cfg.hidden).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=a.lr)
    sizes = spec.head_sizes
    names = [h.name for h in spec.heads]

    def evaluate():
        actor.eval()
        tot = np.zeros(len(sizes))
        loss = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(split["hold"]), a.batch):
                obs, mask, labels, valid, starts = batch(data, split["hold"][i:i + a.batch], L, device)
                logits = actor.forward_sequence(obs, actor.initial_state(valid.shape[1], device), starts)
                ls, cs = head_losses(logits, mask, labels, valid, sizes)
                tot += np.array([c.item() for c in cs])
                loss += sum(l.item() for l in ls)
                n += 1
        actor.train()
        return loss / max(1, n), tot / max(1, n)

    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(split["train"]))
        t0 = time.time()
        for i in range(0, len(order), a.batch):
            wins = [split["train"][j] for j in order[i:i + a.batch]]
            obs, mask, labels, valid, starts = batch(data, wins, L, device)
            logits = actor.forward_sequence(obs, actor.initial_state(valid.shape[1], device), starts)
            ls, _ = head_losses(logits, mask, labels, valid, sizes)
            loss = sum(ls)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
        hl, acc = evaluate()
        print(f"epoch {epoch}: held-out loss {hl:.3f}  accuracy " +
              "  ".join(f"{n} {v:.2f}" for n, v in zip(names, acc)) + f"   ({time.time() - t0:.0f}s)", flush=True)

    critic = Critic(spec, cfg.d_model, cfg.heads, cfg.layers)
    ck = {"actor": actor.state_dict(), "critic": critic.state_dict(), "value_norm": ValueNorm().state_dict(),
          "config": cfg.to_dict(), "spec": spec.to_json(), "update": 0, "bc_heldout_accuracy": dict(zip(names, map(float, acc)))}
    path = a.save
    torch.save(ck, path)
    print("saved", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--battles", type=int, default=120)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--stages", default="1,2,3,4")
    c.add_argument("--weights", default="0.2,0.2,0.4,0.2")
    c.add_argument("--difficulty", type=int, default=2, help="rule AI skill for both fleets (2 = Elite)")
    c.add_argument("--unity-binary", default=os.path.join(HERE, "../Builds/NavalTrainer/NavalTrainer.x86_64"))
    c.add_argument("--base-port", type=int, default=5400)
    c.add_argument("--seed", type=int, default=5)
    c.add_argument("--out", default=os.path.join(OUT, "demos.npz"))
    t = sub.add_parser("train")
    t.add_argument("--demos", default=os.path.join(OUT, "demos.npz"))
    t.add_argument("--save", default=os.path.join(OUT, "bc.pt"))
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--chunk", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    collect(a) if a.cmd == "collect" else train(a)


if __name__ == "__main__":
    main()
