"""Evaluate a checkpoint against the rule-based AI: win rate and damage exchange over N battles.

    python evaluate.py --checkpoint runs/<run>/checkpoints/latest.pt --stage 0 --episodes 96
    python evaluate.py --random --stage 0 --episodes 96          # uniform-random baseline
    python evaluate.py --checkpoint ... --stage 3 --max-team 12  # zero-shot on bigger fleets

The learner's side is randomised per battle, as in training. --greedy takes the policy's likeliest
action instead of sampling. Prints win rate with its standard error, and writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from naval_rl.config import Config
from naval_rl.env import init_message, make_workers
from naval_rl.league import League
from naval_rl.mappo import ACTOR_KEYS, to_t
from naval_rl.model import Actor, sample_actions
from naval_rl.spec import Spec

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--random", action="store_true", help="uniform-random legal actions instead of a policy")
    p.add_argument("--scripted", action="store_true",
                   help="a fixed simple tactic: auto target, fire at will, turn broadside, full speed, no consumables")
    p.add_argument("--unity-binary", default=os.path.join(HERE, "../Builds/NavalTrainer/NavalTrainer.x86_64"))
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--difficulty", type=int, help="rule AI difficulty 0-2 (default: the stage's)")
    p.add_argument("--episodes", type=int, default=96)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--max-team", type=int, help="bigger padding caps for zero-shot evaluation")
    p.add_argument("--base-port", type=int, default=5300)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out")
    a = p.parse_args()
    if not (a.random or a.scripted or a.checkpoint):
        p.error("pass --checkpoint, --random or --scripted")

    cfg = Config()
    actor, spec = None, None
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
        spec = Spec.from_json(ck["spec"])
    n = a.max_team or (spec.max_team if spec else cfg.max_team)
    caps = dict(max_team=n, max_allies=max(n - 1, cfg.max_allies), max_contacts=max(n, cfg.max_contacts),
                max_zones=cfg.max_zones)
    init = init_message(caps["max_team"], caps["max_allies"], caps["max_contacts"], caps["max_zones"],
                        spec.action_mode if spec else cfg.action_mode, cfg.decision_period, cfg.sim_dt, cfg.reflexes)
    workers = make_workers(a.workers, init, os.path.abspath(a.unity_binary), a.base_port)
    wspec = workers[0].spec
    if a.checkpoint:
        actor = Actor(wspec, cfg.d_model, cfg.heads, cfg.layers, cfg.hidden)
        actor.load_state_dict(ck["actor"])        # pointer heads make the weights independent of the caps
        actor.eval()

    cfg.start_stage = a.stage
    cfg.selfplay_from_stage = 99                   # rule-AI opponents only
    league = League(cfg, HERE, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    sizes = wspec.head_sizes
    N, H = wspec.max_team, len(wspec.heads)

    plans, obs = [None] * len(workers), [None] * len(workers)
    hidden = torch.zeros(len(workers), N, cfg.hidden)
    starts = np.ones(len(workers), dtype=np.float32)
    results = []

    def start(w):
        plans[w] = league.sample(w)
        if a.difficulty is not None:
            plans[w].reset["opponent_difficulty"] = a.difficulty
        workers[w].reset_async(plans[w].reset)

    for w in range(len(workers)):
        start(w)
    for w in range(len(workers)):
        obs[w] = workers[w].recv()
    launched = len(workers)
    active = set(range(len(workers)))

    while active:
        for w in list(active):
            L = plans[w].learner_team
            acts = np.zeros((2, N, H), dtype=np.int32)
            m = obs[w].arrays["action_mask"][L]
            if actor is not None:
                with torch.no_grad():
                    o = {k: to_t(obs[w].arrays[k][L], "cpu") for k in ACTOR_KEYS}
                    logits, hidden[w], _ = actor(o, hidden[w], torch.full((N,), float(starts[w])))
                    acts[L] = sample_actions(logits, to_t(m, "cpu"), sizes, a.greedy)[0].numpy()
            elif a.scripted:
                broadside = 10                      # RLLayout.MoveBroadside; option 0 elsewhere = auto / fire / none
                for i in range(N):
                    acts[L, i, 0] = broadside if m[i, broadside] > 0.5 else 0
            else:
                for i in range(N):
                    off = 0
                    for h, s in enumerate(sizes):
                        legal = np.flatnonzero(m[i, off:off + s] > 0.5)
                        acts[L, i, h] = rng.choice(legal) if len(legal) else 0
                        off += s
            starts[w] = 0.0
            workers[w].step_async(acts)
        for w in list(active):
            o = workers[w].recv()
            if o.terminal:
                L = plans[w].learner_team
                st = o.stats or {}
                results.append({
                    "score": 0.5 if o.draw else float(o.winner == L),
                    "dealt": st.get("damage_dealt_hulls", [0, 0])[L],
                    "taken": st.get("damage_dealt_hulls", [0, 0])[1 - L],
                    "reason": o.reason,
                })
                if launched < a.episodes:
                    start(w)
                    o = workers[w].recv()
                    hidden[w] = 0.0
                    starts[w] = 1.0
                    launched += 1
                else:
                    active.discard(w)
            obs[w] = o

    for wk in workers:
        wk.close()
    s = np.array([r["score"] for r in results])
    summary = {
        "who": "random" if a.random else "scripted" if a.scripted else a.checkpoint, "stage": a.stage, "episodes": len(results),
        "win_rate": float(s.mean()), "stderr": float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
        "damage_dealt": float(np.mean([r["dealt"] for r in results])),
        "damage_taken": float(np.mean([r["taken"] for r in results])),
        "greedy": a.greedy,
    }
    print(f"{summary['who']}: win rate {summary['win_rate']:.2f} ± {summary['stderr']:.2f} over {len(results)} battles, "
          f"damage dealt {summary['damage_dealt']:.2f} vs taken {summary['damage_taken']:.2f} (hulls)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "battles": results}, f, indent=1)


if __name__ == "__main__":
    main()
