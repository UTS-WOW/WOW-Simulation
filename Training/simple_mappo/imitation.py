"""Imitation learning: copy the game's best rule-based AI first, then improve on it with RL.

This is how AlphaStar started (supervised learning on human games, then reinforcement learning), and
the reason is simple: a random policy almost never wins a naval battle, so RL from scratch spends
most of its time learning what "sail, aim, shoot" even means. The game already has an Elite AI, so:

  1. collect_expert   the Elite AI flies BOTH fleets; for every decision of every ship of ours the
                      game reports which option it chose on each of the six action heads
                      (Assets/Scripts/RL/RLExpertLabels.cs). That is a labelled dataset.
  2. behaviour_cloning  train the captain network to predict those labels - a classifier, trained
                      with exactly the Week 9 Part A loop (DataLoader, CrossEntropyLoss, Adam,
                      a loss curve and an accuracy curve), one cross-entropy per action head.
  3. MAPPO.init_from_imitation  RL starts from the cloned captain. A KL penalty keeps it close to
                      the clone at first and fades out, so the fleet keeps the rule AI's skills while
                      it learns to beat it (AlphaStar's KL to the supervised policy).

In this project's first pipeline (naval_rl) cloning alone already won 58 % of King of the Hill
battles, where RL from scratch won 20 %.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .networks import CAPTAIN_KEYS, Captain

DATA_KEYS = CAPTAIN_KEYS + ["action_mask"]


# ---------------------------------------------------------------------------------------------- 1. collect

def collect_expert(env, n_battles: int = 60, stages=(3, 4, 5, 6), seed: int = 0, verbose: int = 1) -> dict:
    """Records n_battles battles of the rule AI. env: NavalEnv(..., record_expert=True).

    stages: curriculum stages to draw each battle from. Returns one sequence per ship per battle:
    {"obs": {key: [decisions, ...]}, "labels" [decisions, 6], "valid" [decisions], "lengths" [sequences]}.
    """
    if not env.record_expert:
        raise ValueError("collect_expert needs NavalEnv(..., record_expert=True)")
    rng = np.random.default_rng(seed)
    E, N, H = env.n_envs, env.n_agents, len(env.head_sizes)
    env.set_stage(int(rng.choice(stages)))
    obs = env.reset()
    running = [[] for _ in range(E)]          # per battle slot: (obs slices, labels, valid) per decision
    sequences = []
    finished, t0 = 0, time.time()
    no_actions = np.zeros((E, N, H), dtype=np.int32)          # ignored: the rule AI flies both fleets
    while finished < n_battles:
        env.set_stage(int(rng.choice(stages)))                # the next battle to start draws its stage
        next_obs, _, dones, infos = env.step(no_actions)
        labels, valid = env.last_expert                       # what the rule AI did on `obs`
        for e in range(E):
            running[e].append(({k: obs[k][e].astype(np.float16) for k in DATA_KEYS}, labels[e].astype(np.int8),
                               valid[e].astype(bool)))
            if dones[e]:
                sequences += _ship_sequences(running[e])
                running[e] = []
                finished += 1
                if verbose:
                    r = infos[e]
                    print(f"  battle {finished}/{n_battles}: stage {r['stage']}, {r['episode']['l']} decisions "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        obs = next_obs
    data = {"obs": {k: np.concatenate([s["obs"][k] for s in sequences]) for k in DATA_KEYS},
            "labels": np.concatenate([s["labels"] for s in sequences]),
            "valid": np.concatenate([s["valid"] for s in sequences]),
            "lengths": np.array([len(s["labels"]) for s in sequences])}
    if verbose:
        print(f"{len(sequences)} ship sequences, {int(data['valid'].sum()):,} labelled decisions")
    return data


def _ship_sequences(steps: list) -> list[dict]:
    """One battle's decisions -> one sequence per ship slot that has labels."""
    valid = np.stack([v for _, _, v in steps])                             # [T, N]
    out = []
    for i in np.flatnonzero(valid.any(axis=0)):
        out.append({"obs": {k: np.stack([o[k][i] for o, _, _ in steps]) for k in DATA_KEYS},
                    "labels": np.stack([l[i] for _, l, _ in steps]), "valid": valid[:, i]})
    return out


def save_dataset(data: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, labels=data["labels"], valid=data["valid"], lengths=data["lengths"],
                        **{"obs_" + k: v for k, v in data["obs"].items()})
    return path


def load_dataset(path: str) -> dict:
    raw = np.load(path)
    return {"obs": {k: raw["obs_" + k] for k in DATA_KEYS}, "labels": raw["labels"], "valid": raw["valid"],
            "lengths": raw["lengths"]}


def merge_datasets(*datas: dict) -> dict:
    return {"obs": {k: np.concatenate([d["obs"][k] for d in datas]) for k in DATA_KEYS},
            "labels": np.concatenate([d["labels"] for d in datas]), "valid": np.concatenate([d["valid"] for d in datas]),
            "lengths": np.concatenate([d["lengths"] for d in datas])}


# ---------------------------------------------------------------------------------------------- 2. clone

class WindowDataset(torch.utils.data.Dataset):
    """Windows of `length` consecutive decisions of one ship (the GRU needs sequences, not single
    decisions). Item: (obs {key: [length, ...]}, labels [length, 6], weight [length], starts [length])."""

    def __init__(self, data: dict, sequences: np.ndarray, length: int):
        self.data, self.length = data, length
        offsets = np.concatenate([[0], np.cumsum(data["lengths"])[:-1]])
        self.windows = [(offsets[i] + s, min(length, data["lengths"][i] - s))
                        for i in sequences for s in range(0, data["lengths"][i], length)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, n = self.windows[idx]
        rows = np.minimum(np.arange(start, start + self.length), start + n - 1)   # pad by repeating the last
        weight = (np.arange(self.length) < n) * self.data["valid"][rows]
        starts = np.zeros(self.length, np.float32)
        starts[0] = 1.0
        obs = {k: torch.as_tensor(self.data["obs"][k][rows].astype(np.float32)) for k in DATA_KEYS}
        return obs, torch.as_tensor(self.data["labels"][rows].astype(np.int64)), \
            torch.as_tensor(weight.astype(np.float32)), torch.as_tensor(starts)


def head_losses(logits, mask, labels, weight, sizes):
    """Cross-entropy and accuracy of each action head (illegal options masked, as in play)."""
    logits = logits.masked_fill(mask < 0.5, -1e9)
    losses, accuracy, off = [], [], 0
    total = weight.sum().clamp_min(1.0)
    for h, size in enumerate(sizes):
        part = logits[..., off:off + size]
        ce = F.cross_entropy(part.reshape(-1, size), labels[..., h].reshape(-1), reduction="none").view(weight.shape)
        losses.append((ce * weight).sum() / total)
        accuracy.append(((part.argmax(-1) == labels[..., h]).float() * weight).sum() / total)
        off += size
    return losses, accuracy


def behaviour_cloning(data: dict, dims: dict, heads: list[dict], d_model: int = 64, attn_heads: int = 4,
                      attn_layers: int = 2, hidden_size: int = 128, recurrent: bool = True, num_epochs: int = 8,
                      batch_size: int = 64, window: int = 32, learning_rate: float = 3e-4, held_out: float = 0.1,
                      device: str = "auto", seed: int = 0, verbose: int = 1):
    """Trains a captain to copy the recorded decisions. Returns (captain, history).

    history: per epoch the training loss and accuracy and the held-out accuracy per head - the
    loss curve and accuracy curve of Week 9 Part A."""
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                          "cpu" if device == "auto" else device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data["lengths"]))
    n_hold = max(1, int(len(order) * held_out))
    train_set = WindowDataset(data, order[n_hold:], window)
    test_set = WindowDataset(data, order[:n_hold], window)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = Captain(dims, heads, d_model, attn_heads, attn_layers, hidden_size, recurrent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sizes, names = [h["size"] for h in heads], [h["name"] for h in heads]
    history = {"train_loss": [], "train_accuracy": [], "test_accuracy": []}

    def run(batch):
        obs, labels, weight, starts = batch
        obs = {k: v.transpose(0, 1).to(device) for k, v in obs.items()}               # [window, batch, ...]
        labels, weight, starts = labels.transpose(0, 1).to(device), weight.transpose(0, 1).to(device), starts.transpose(0, 1).to(device)
        B = labels.shape[1]
        orders = torch.zeros(window, B, dtype=torch.long, device=device)              # every ship "free"
        logits = model.sequence({k: obs[k] for k in CAPTAIN_KEYS}, orders, torch.zeros(B, hidden_size, device=device), starts)
        return head_losses(logits, obs["action_mask"], labels, weight, sizes)

    for epoch in range(num_epochs):
        model.train()
        losses, accs = [], []
        for i, batch in enumerate(train_loader):
            head_loss, head_acc = run(batch)
            loss = sum(head_loss)                                 # one cross-entropy per action head
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            accs.append([a.item() for a in head_acc])
        model.eval()
        with torch.no_grad():
            test = np.mean([[a.item() for a in run(batch)[1]] for batch in test_loader], axis=0)
        history["train_loss"].append(float(np.mean(losses)))
        history["train_accuracy"].append(np.mean(accs, axis=0).tolist())
        history["test_accuracy"].append(test.tolist())
        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(losses):.4f}, held-out accuracy: "
                  + "  ".join(f"{n} {a:.2f}" for n, a in zip(names, test)), flush=True)
    model.arch = dict(d=d_model, attn_heads=attn_heads, layers=attn_layers, hidden=hidden_size, recurrent=recurrent)
    return model, history


def save_imitation(path: str, captain: Captain, dims: dict, heads: list[dict], history: dict) -> str:
    """The checkpoint MAPPO.init_from_imitation reads."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    names = [h["name"] for h in heads]
    accuracy = dict(zip(names, history["test_accuracy"][-1])) if history["test_accuracy"] else {}
    torch.save({"captain": captain.state_dict(), "dims": dict(dims), "heads": heads, "arch": captain.arch,
                "accuracy": accuracy, "history": history}, path)
    return path


def plot_imitation(history: dict, head_names: list[str]):
    """The Week 9 Part A plots: the training loss, and the held-out accuracy of every head."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (sum over heads)")
    plt.title("Imitation: training loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    acc = np.array(history["test_accuracy"])
    for h, name in enumerate(head_names):
        plt.plot(range(1, len(acc) + 1), acc[:, h], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Held-out accuracy")
    plt.title("Imitation: does the captain choose what the Elite AI chose?")
    plt.legend()
    plt.tight_layout()
    plt.show()
