"""Checks that each kind of order does what it should in the real game, and films it.

Plays the 3v3 archipelago scenario with hand-written orders for the Player fleet (no trained
policy involved) against a Recruit AI, and verifies from the game's own state:

    move     every ship sails for zone B through the islands without running aground, and one gets in
    radar    the cruiser's surveillance radar comes on
    smoke    the destroyer's smoke screen comes on
    torpedo  the destroyer launches a spread at a contact (ammo goes down)
    hold     a battleship holding fire does not fire, then opens fire once allowed

    python probe_actions.py            # writes runs/probe/battle.mp4 and prints PASS/FAIL per check
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np

from naval_rl.env import UnityWorker, init_message, launch_player

HERE = os.path.dirname(os.path.abspath(__file__))
MOVE_FIXED = 13
ZONE_B = MOVE_FIXED + 1
ABILITY = {"smoke": 3, "radar": 5}


def main():
    out = os.path.join(HERE, "runs", "probe")
    frames = os.path.join(out, "frames")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(frames)
    binary = os.environ.get("PROBE_BINARY", os.path.join(HERE, "../Builds/NavalTrainer/NavalTrainer.x86_64"))
    port = int(os.environ.get("PROBE_PORT", 5960))
    w = UnityWorker(port, init_message(8, 7, 8, 5, "intent", 1.0, 0.02, True),
                    process=launch_player(os.path.abspath(binary), port, out, graphics=True))
    spec = w.spec
    f = {name: i for i, name in enumerate(spec.features["self"])}
    zf = {name: i for i, name in enumerate(spec.features["zone"])}
    offsets = np.cumsum([0] + spec.head_sizes)
    results = {}

    def check(name, ok, detail):
        results[name] = (bool(ok), detail)

    try:
        with open(os.path.join(HERE, "scenarios/stage2_archipelago_3v3.json")) as fh:
            scenario = json.load(fh)
        w.reset_async({"use_scenario": True, "scenario": scenario, "learned_teams": 1,
                       "opponent_difficulty": 0, "episode_seed": 11, "time_limit": 900})
        o = w.recv()
        ships = int(o.ships[0])
        cls = ["dd" if o.arrays["self"][0, i, f["cls_dd"]] else "ca" if o.arrays["self"][0, i, f["cls_ca"]] else "bb"
               for i in range(ships)]
        dd, ca, bb = cls.index("dd"), cls.index("ca"), cls.index("bb")
        start_dist = o.arrays["zones"][0, :ships, 1, zf["dist"]].copy()
        best_dist = start_dist.copy()           # closest each ship got while afloat
        min_keel = np.ones(ships)
        reached = [False] * ships
        torp_before = None
        bb_fired_while_holding = False
        bb_fired_after = False

        def legal(i, head, option):
            return o.arrays["action_mask"][0, i, offsets[head] + option] > 0.5

        for step in range(420):
            if step % 5 == 0:
                w.render(os.path.join(frames, f"{step // 5:05d}.png"), margin=180)
            a = np.zeros((2, 8, len(spec.heads)), dtype=np.int32)
            for i in range(ships):
                if o.arrays["alive"][0, i] < 0.5:
                    continue
                a[0, i, 0] = ZONE_B if legal(i, 0, ZONE_B) else 0
                if o.arrays["contact_mask"][0, i, 0] > 0.5:
                    a[0, i, 2] = 1                                   # shoot the nearest contact
                a[0, i, 3] = 1 if (i == bb and step < 120) else 0    # battleship holds fire at first
            if step == 20 and legal(ca, 5, ABILITY["radar"]):
                a[0, ca, 5] = ABILITY["radar"]
            if step == 25 and legal(dd, 5, ABILITY["smoke"]):
                a[0, dd, 5] = ABILITY["smoke"]
            if torp_before is None and legal(dd, 4, 1) and a[0, dd, 2] == 1:
                a[0, dd, 4] = 1
                torp_before = o.arrays["self"][0, dd, f["torp_ammo"]]
            w.step_async(a)
            prev = o
            o = w.recv()
            s = o.arrays["self"][0]
            for i in range(ships):
                if o.arrays["alive"][0, i] > 0.5:
                    min_keel[i] = min(min_keel[i], s[i, f["depth_under_keel"]])
                    best_dist[i] = min(best_dist[i], o.arrays["zones"][0, i, 1, zf["dist"]])
                    reached[i] |= bool(o.arrays["zones"][0, i, 1, zf["i_am_inside"]] > 0.5)
            if step == 21:
                check("radar", s[ca, f["surveillanceradar_active"]] > 0.5, "cruiser radar active after the order")
            if step == 26:
                check("smoke", s[dd, f["smokescreen_active"]] > 0.5, "destroyer smoke active after the order")
            if torp_before is not None and "torpedo" not in results:
                check("torpedo", s[dd, f["torp_ammo"]] < torp_before,
                      f"destroyer torpedo ammo {torp_before:.2f} -> {s[dd, f['torp_ammo']]:.2f}")
            fired = s[bb, f["recently_fired"]] > 0.5 and prev.arrays["self"][0, bb, f["recently_fired"]] < 0.5
            if step < 120 and fired:
                bb_fired_while_holding = True
            if step >= 120 and fired:
                bb_fired_after = True
            if o.terminal:
                break

        closer = [bool(best_dist[i] < start_dist[i] - 0.05) or reached[i] for i in range(ships)]
        check("move", all(closer) and any(reached),
              f"distance to zone B in km, start {np.round(start_dist * 28, 1).tolist()} closest {np.round(best_dist * 28, 1).tolist()}, "
              f"inside the ring {reached}")
        check("no grounding", float(min_keel.min()) > 0.0, f"least water under any keel {min_keel.min():.3f}")
        check("hold fire", not bb_fired_while_holding and bb_fired_after,
              f"fired while holding: {bb_fired_while_holding}, fired after release: {bb_fired_after}")
        for k in ("radar", "smoke", "torpedo"):
            results.setdefault(k, (False, "the order never became legal"))
    finally:
        w.close()

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", os.path.join(frames, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(out, "battle.mp4")], check=True)
    ok = True
    for k, (passed, detail) in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {k:<13} {detail}")
        ok &= passed
    print("video:", os.path.join(out, "battle.mp4"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
