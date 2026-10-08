"""Train the simple MAPPO fleet from the command line - on your machine or as a SageMaker training job.

    python train_simple.py --total-timesteps 5000000                 # the curriculum, stages 0-5
    python train_simple.py --start-stage 3                           # begin at a later stage
    python train_simple.py --curriculum my_stages.json               # your own stages (any number)
    python train_simple.py --scenario scenarios/stage0_bb_duel.json --run-name duel   # one battle only
    python train_simple.py --resume runs_simple/koth/models/model_final.pt --total-timesteps 1000000
    python train_simple.py --mock --total-timesteps 20000          # no Unity: checks the code only

Everything goes to runs_simple/<run-name>/: logs/monitor.csv (one line per battle),
logs/progress.csv (one line per update), models/model_<timesteps>.pt, eval.json.

Team training: with --team-storage the run lives in a shared folder (normally on S3) that every
teammate continues from - the latest checkpoint is pulled at the start, pushed every
--save-interval timesteps and at the end, and a lock stops two people training the same run at once:

    python train_simple.py --team-storage s3://<bucket>/wow-mappo --run-name curriculum --user Lukita

As a SageMaker training job (see the last section of WOW-MAPPO-Simple.ipynb) hyperparameters
arrive as these same --flags and the headless player comes from the "unity" input channel. Use
--team-storage there too: it is how the job saves checkpoints while it runs. The final model and
logs are also copied to /opt/ml/model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import OrderedDict

from simple_mappo import (MAPPO, CurriculumCallback, Monitor, NavalEnv, REWARD_WEIGHTS, SaveOnIntervalCallback,
                          TeamStorage, TeamSyncCallback, evaluate)

HERE = os.path.dirname(os.path.abspath(__file__))
ON_SAGEMAKER = "SM_MODEL_DIR" in os.environ


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # the battle
    p.add_argument("--curriculum", default="default",
                   help='"default" (stages 0-5, see simple_mappo/curriculum.py) or a JSON file of stages')
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--scenario", nargs="+", help="train on these battles only, instead of the curriculum")
    p.add_argument("--opponent-difficulty", type=int, default=0, help="with --scenario: 0 Recruit, 1 Veteran, 2 Elite")
    p.add_argument("--n-envs", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)),
                   help="battles in parallel, one Unity process each (default: half the CPU cores)")
    p.add_argument("--decision-period", type=float, default=1.0, help="game seconds between decisions (action repeat)")
    p.add_argument("--sim-dt", type=float, default=0.08, help="game seconds per frame: 0.02 normal, 0.08 fastest stable")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--reward-weights", type=json.loads, default=None,
                   help='JSON overrides, e.g. \'{"in_circle": 0.05, "win": 10}\'')
    # MAPPO
    p.add_argument("--total-timesteps", type=int, default=500_000,
                   help="the run's total: a resumed run trains until it reaches this")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--ippo", action="store_true", help="IPPO baseline: the critic sees only each ship's own view")
    p.add_argument("--no-value-norm", action="store_true", help="switch off MAPPO's value normalisation")
    p.add_argument("--seed", type=int, default=0)
    # bookkeeping
    p.add_argument("--run-name", default="curriculum")
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--eval-battles", type=int, default=40, help="battles for the final evaluation (0: skip)")
    p.add_argument("--resume", help="a saved model to continue training")
    p.add_argument("--team-storage", help="shared folder for the team's runs, e.g. s3://<bucket>/wow-mappo")
    p.add_argument("--user", help="your name, shown to teammates in the shared run")
    p.add_argument("--force-lock", action="store_true", help="take over a run whose lock is held by a dead session")
    p.add_argument("--unity-binary")
    p.add_argument("--base-port", type=int, default=5005)
    p.add_argument("--device", default="auto")
    p.add_argument("--mock", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    storage = TeamStorage(a.team_storage, a.run_name, user=a.user) if a.team_storage else None
    if ON_SAGEMAKER and not storage:
        print("WARNING: no --team-storage: this job saves its model only when it finishes (to /opt/ml/model)")
    run_dir = os.path.join(HERE, "runs_simple", a.run_name)
    log_dir, models_dir = os.path.join(run_dir, "logs"), os.path.join(run_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    pulled = None
    if storage:
        pulled = storage.pull(run_dir)                  # the team's latest checkpoint and logs
        storage.lock(force=a.force_lock)                # refuses while a teammate is training this run
        print(f"training the team's run at {storage.location} as {storage.user}")

    weights = OrderedDict(REWARD_WEIGHTS)
    weights.update(a.reward_weights or {})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(a), "reward_weights": weights}, f, indent=2)

    env = Monitor(NavalEnv(a.scenario or "", n_envs=a.n_envs, opponent_difficulty=a.opponent_difficulty,
                           stages=None if a.scenario else a.curriculum, start_stage=a.start_stage,
                           decision_period=a.decision_period, sim_dt=a.sim_dt,
                           time_limit=a.time_limit, reward_weights=weights, unity_binary=a.unity_binary,
                           base_port=a.base_port, seed=a.seed, mock=a.mock,
                           unity_log_dir=os.path.join(run_dir, "unity_logs")), log_dir)
    try:
        resume = a.resume or pulled
        if resume:
            model = MAPPO.load(resume, env=env, device=a.device, log_dir=log_dir, verbose=1)
            print(f"resuming {resume} at {model.num_timesteps} timesteps")
        else:
            model = MAPPO(env, learning_rate=a.learning_rate, n_steps=a.n_steps, n_minibatches=a.n_minibatches,
                          n_epochs=a.n_epochs, gamma=a.gamma, gae_lambda=a.gae_lambda, clip_range=a.clip_range,
                          ent_coef=a.ent_coef, vf_coef=a.vf_coef, hidden_size=a.hidden_size,
                          centralised_critic=not a.ippo, normalize_values=not a.no_value_norm, device=a.device,
                          seed=a.seed, log_dir=log_dir)

        t0 = time.time()
        print("stages:", " -> ".join(s["name"] for s in env.stages), f"(starting at stage {env.stage})")
        callbacks = [SaveOnIntervalCallback(a.save_interval, models_dir), CurriculumCallback()]
        if storage:
            callbacks.append(TeamSyncCallback(storage, run_dir, push_every=a.save_interval))
        model.learn(total_timesteps=a.total_timesteps, callback=callbacks)
        model.save(os.path.join(models_dir, "model_final"))
        model.save(os.path.join(models_dir, "model_latest"))
        if storage and callbacks[-1].conflict is None:
            storage.push(run_dir, model)                 # model_final as well
        print(f"trained {model.num_timesteps} timesteps in {(time.time() - t0) / 60:.1f} min")

        if a.eval_battles > 0:
            # on the stage training reached
            result = {"trained": evaluate(model, env, a.eval_battles),
                      "random": evaluate(None, env, a.eval_battles)}
            print(json.dumps(result, indent=2))
            with open(os.path.join(run_dir, "eval.json"), "w") as f:
                json.dump(result, f, indent=2)
    finally:
        env.close()
        if storage:
            storage.release()

    if ON_SAGEMAKER:
        # /opt/ml/model is packed into model.tar.gz on S3 when the job ends
        out = os.environ["SM_MODEL_DIR"]
        shutil.copy(os.path.join(models_dir, "model_final.pt"), out)
        shutil.copytree(log_dir, os.path.join(out, "logs"), dirs_exist_ok=True)
        for f in ("config.json", "eval.json"):
            if os.path.exists(os.path.join(run_dir, f)):
                shutil.copy(os.path.join(run_dir, f), out)


if __name__ == "__main__":
    main()
