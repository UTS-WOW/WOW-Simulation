"""Watch a trained policy play one battle: renders frames and stitches them into a video.

    python watch.py --checkpoint runs/first/checkpoints/latest.pt --stage 0
    python watch.py --checkpoint ... --scenario scenarios/stage2_archipelago_3v3.json --difficulty 1

The learner flies the Player fleet (cyan); the rule-based AI flies the Enemy (crimson). Fog of war
is revealed in the frames so both fleets are visible, but the policy itself still only sees what
its team has spotted. Writes <out>/frames/*.png, <out>/battle.mp4 and <out>/battle.gif (ffmpeg).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import torch

from naval_rl.config import Config, DEFAULT_STAGES
from naval_rl.env import UnityWorker, init_message, launch_player
from naval_rl.model import Actor, sample_actions
from naval_rl.mappo import ACTOR_KEYS, to_t
from naval_rl.spec import Spec

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unity-binary", default=os.path.join(HERE, "../Builds/NavalTrainer/NavalTrainer.x86_64"))
    p.add_argument("--stage", type=int, default=0, help="curriculum stage whose battle to play")
    p.add_argument("--scenario", help="scenario JSON instead of a stage")
    p.add_argument("--difficulty", type=int, help="rule AI difficulty 0-2 (default: the stage's)")
    p.add_argument("--every", type=int, default=3, help="render every N decisions (seconds of game time)")
    p.add_argument("--greedy", action="store_true", help="always take the policy's likeliest action")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "watch"))
    p.add_argument("--port", type=int, default=5950)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    spec = Spec.from_json(ck["spec"])
    actor = Actor(spec, cfg.d_model, cfg.heads, cfg.layers, cfg.hidden)
    actor.load_state_dict(ck["actor"])
    actor.eval()

    frames = os.path.join(a.out, "frames")
    shutil.rmtree(frames, ignore_errors=True)
    os.makedirs(frames)
    proc = launch_player(os.path.abspath(a.unity_binary), a.port, a.out, graphics=True)
    init = init_message(spec.max_team, spec.max_allies, spec.max_contacts, spec.max_zones, spec.action_mode,
                        spec.decision_period, spec.sim_dt, cfg.reflexes)
    w = UnityWorker(a.port, init, process=proc)
    try:
        stage = DEFAULT_STAGES[a.stage]
        reset = {"learned_teams": 1, "episode_seed": a.seed,
                 "opponent_difficulty": a.difficulty if a.difficulty is not None else stage.get("difficulty", 2)}
        scen = a.scenario or stage.get("scenario")
        if scen:
            with open(scen if os.path.isabs(scen) else os.path.join(HERE, scen)) as f:
                reset.update({"use_scenario": True, "scenario": json.load(f)})
        else:
            pr = stage["procedural"]
            reset.update({"use_scenario": False, "mode": pr.get("mode", 0), "preset": 0, "density": 1,
                          "player_ships": min(6, spec.max_team), "enemy_ships": min(6, spec.max_team),
                          "seed": a.seed, "time_limit": pr.get("time_limit", 0)})
        w.reset_async(reset)
        o = w.recv()
        N, H = spec.max_team, len(spec.heads)
        h = torch.zeros(N, cfg.hidden)
        starts = torch.ones(N)
        step, frame = 0, 0
        while True:
            if step % a.every == 0:
                w.render(os.path.join(frames, f"{frame:05d}.png"))
                frame += 1
            with torch.no_grad():
                obs = {k: to_t(o.arrays[k][0], "cpu") for k in ACTOR_KEYS}
                logits, h, _ = actor(obs, h, starts)
                acts, _, _ = sample_actions(logits, to_t(o.arrays["action_mask"][0], "cpu"), spec.head_sizes, a.greedy)
            starts = torch.zeros(N)
            actions = np.zeros((2, N, H), dtype=np.int32)
            actions[0] = acts.numpy()
            w.step_async(actions)
            o = w.recv()
            step += 1
            if o.terminal:
                w.render(os.path.join(frames, f"{frame:05d}.png"))
                break
        result = "draw" if o.draw else ("learner won" if o.winner == 0 else "rule AI won")
        print(f"{result}: {o.reason} after {o.battle_time:.0f} s, {frame + 1} frames")
        print(json.dumps(o.stats))
    finally:
        w.close()

    fps = 10
    mp4 = os.path.join(a.out, "battle.mp4")
    gif = os.path.join(a.out, "battle.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames, "%05d.png"),
                    "-vf", "scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", gif], check=True)
    print("wrote", mp4, "and", gif)


if __name__ == "__main__":
    main()
