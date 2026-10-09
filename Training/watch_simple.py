"""See a trained simple-MAPPO fleet in the real Unity game - on a computer with a screen.

Training runs the game headless (no graphics), so the replays it records are a 2D map drawn from
the game's data. This shows the actual game instead:

    # a video rendered by Unity: one battle, fog of war lifted so both fleets show
    python watch_simple.py --model runs_simple/curriculum/models/model_final.pt

    # play it live in the Unity editor: open the project, tick GameBootstrap > Rl Training Server,
    # press Play, then
    python watch_simple.py --model runs_simple/curriculum/models/model_final.pt --live

A model trained on SageMaker: download it from the team folder (or the Studio file browser,
right-click > Download) and pass its path. --stage picks the battle (default: the stage the model
reached). Not for SageMaker spaces: they have no screen, so Unity cannot render there.
"""

from __future__ import annotations

import argparse
import os

from simple_mappo import MAPPO, NavalEnv, record_unity_video

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="a saved model, e.g. runs_simple/<run>/models/model_final.pt")
    p.add_argument("--stage", type=int, help="curriculum stage to play (default: the model's)")
    p.add_argument("--out", help="video file (default: runs_simple/watch/<model>_stage<k>.mp4)")
    p.add_argument("--battles", type=int, default=1, help="battles to record")
    p.add_argument("--live", action="store_true", help="play in the Unity editor (port 5005) instead")
    p.add_argument("--port", type=int, default=5005, help="the editor's port with --live")
    a = p.parse_args()

    if a.live:
        # the editor renders every frame; sim_dt 0.02 is the game's normal step, so it plays at a watchable pace
        env = NavalEnv(stages="default", ports=[a.port], sim_dt=0.02)
    else:
        env = NavalEnv(stages="default", n_envs=1, graphics=True, base_port=5900,
                       unity_log_dir=os.path.join(HERE, "runs_simple", "watch", "unity_logs"))
    model = MAPPO.load(a.model, env=env)
    if a.stage is not None:
        env.set_stage(a.stage)
    print(f"{os.path.basename(a.model)} ({model.num_timesteps:,} timesteps) on stage '{env.stages[env.stage]['name']}'")

    try:
        if a.live:
            print("playing in the editor - watch the Game view; Ctrl+C to stop")
            obs = env.reset()
            while True:
                obs, _, dones, infos = env.step(model.predict(obs))
                if dones[0]:
                    r = infos[0]
                    print(f"{'won' if r['won'] else 'lost'}: {r['reason']} after {r['battle_time']:.0f} s")
        else:
            name = os.path.splitext(os.path.basename(a.model))[0]
            for i in range(a.battles):
                out = a.out or os.path.join(HERE, "runs_simple", "watch", f"{name}_stage{env.stage}_{i + 1}.mp4")
                record_unity_video(model, env, out)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
