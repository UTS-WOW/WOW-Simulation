"""Put a trained fleet into the Unity game - no Python needed while the game runs.

    # from this repository, after training here:
    python export_simple.py                                   # runs_simple/fleet: model_best.pt (else model_latest.pt)
    python export_simple.py --model runs_simple/fleet/models/model_latest.pt
    python export_simple.py --model runs_simple/fleet/models/imitation.pt      # just the clone of the Elite AI

    # a run trained on SageMaker / by a teammate: fetch its checkpoint from the team folder first
    python export_simple.py --team-storage s3://<bucket>/wow-mappo --run-name fleet

It writes Assets/StreamingAssets/RL/naval_policy.bin (the previous one is kept as naval_policy.bin.bak).
Then open the project in Unity, press Play, and on the setup screen set ENEMY AI: LEARNED (fight
against it) or YOUR FLEET: LEARNED (watch it fight). RLPolicyDriver flies those ships with the
trained captains - and, in fleets of two or more, with the trained fleet commander. The file is read
when the game starts: after exporting again, stop and press Play again. The line under those
buttons says which policy is loaded ("commander MAPPO with fleet commander").
"""

from __future__ import annotations

import argparse
import os

from simple_mappo.export import GAME_POLICY, GAME_STAGES, export_policy, export_stages

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="a checkpoint (model_*.pt or imitation.pt); default: the run's best, else latest")
    p.add_argument("--run-name", default="fleet")
    p.add_argument("--team-storage", help="download the run's checkpoint from this team folder first (s3://... or a folder)")
    p.add_argument("--out", default=GAME_POLICY, help="the policy file (default: the game's StreamingAssets/RL/naval_policy.bin)")
    p.add_argument("--no-commander", action="store_true", help="export the captains only (every ship free)")
    p.add_argument("--stages-only", action="store_true",
                   help="only write the curriculum for the game's TRAINING STAGES screen (stages.json)")
    a = p.parse_args()

    # the curriculum, for the game's TRAINING STAGES screen (kept in step with simple_mappo/curriculum.py)
    stages = export_stages(os.path.join(os.path.dirname(os.path.abspath(a.out)), "stages.json")
                           if a.out != GAME_POLICY else GAME_STAGES)
    print(f"stages     -> {stages}")
    if a.stages_only:
        return

    run_dir = os.path.join(HERE, "runs_simple", a.run_name)
    model = a.model
    if model is None and a.team_storage:
        from simple_mappo import TeamStorage
        storage = TeamStorage(a.team_storage, a.run_name)
        for name in ("model_best.pt", "model_latest.pt"):
            local = os.path.join(run_dir, "models", name)
            if storage.backend.download(f"models/{name}", local):
                model = local
                print(f"downloaded {name} from {storage.location}")
                break
    if model is None:
        for name in ("model_best.pt", "model_latest.pt", "imitation.pt"):
            if os.path.exists(os.path.join(run_dir, "models", name)):
                model = os.path.join(run_dir, "models", name)
                break
    if model is None:
        raise SystemExit(f"no checkpoint found in {run_dir}/models - pass --model")

    path = export_policy(model, os.path.abspath(a.out), commander=not a.no_commander)
    print(f"exported {model}\n      -> {path}")
    print("In Unity: press Play (again), then on the setup screen set ENEMY AI: LEARNED or YOUR FLEET: LEARNED.")


if __name__ == "__main__":
    main()
