"""Train the fleet (MAPPO with a commander) from the command line - on your machine or as a SageMaker job.

    python train_simple.py --total-timesteps 5000000                 # imitation, then the curriculum, stages 0-7
    python train_simple.py --start-stage 3                           # begin at a later stage
    python train_simple.py --curriculum my_stages.json               # your own stages (any number)
    python train_simple.py --scenario scenarios/stage0_bb_duel.json --run-name duel   # one battle only
    python train_simple.py --resume runs_simple/koth/models/model_final.pt --total-timesteps 1000000
    python train_simple.py --mock --total-timesteps 20000          # no Unity: checks the code only
    python train_simple.py --no-commander --no-memory --attn-layers 0 --run-name ablation_flat   # an ablation

A new run starts with imitation (--imitation-battles, default 60): the Elite rule AI is recorded and
the captains are cloned from it (simple_mappo/imitation.py) before reinforcement learning begins.

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

from simple_mappo import (MAPPO, BestModelCallback, CurriculumCallback, Monitor, NavalEnv, REWARD_WEIGHTS,
                          SaveOnIntervalCallback, TeamStorage, TeamSyncCallback, VideoCallback, behaviour_cloning,
                          collect_expert, evaluate, save_imitation, training_status)

HERE = os.path.dirname(os.path.abspath(__file__))
ON_SAGEMAKER = "SM_MODEL_DIR" in os.environ


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # the battle
    p.add_argument("--curriculum", default="default",
                   help='"default" (stages 0-7, see simple_mappo/curriculum.py) or a JSON file of stages')
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
    # imitation (a new run only)
    p.add_argument("--imitation-battles", type=int, default=60,
                   help="battles of the Elite rule AI to clone before RL (0: start from scratch)")
    p.add_argument("--imitation-epochs", type=int, default=8)
    p.add_argument("--no-imitation", action="store_true", help="ignore an existing imitation checkpoint")
    # MAPPO
    p.add_argument("--total-timesteps", type=int, default=500_000,
                   help="the run's total: a resumed run trains until it reaches this")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--critic-learning-rate", type=float, default=5e-4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--dual-clip", type=float, default=3.0, help="Honor of Kings' dual clip (0: off)")
    p.add_argument("--ent-coef", default="0.01", help='entropy bonus at the start, or "auto" (SAC-style)')
    p.add_argument("--d-model", type=int, default=64, help="entity encoder width")
    p.add_argument("--attn-layers", type=int, default=2, help="transformer layers (0: pooling only)")
    p.add_argument("--hidden-size", type=int, default=128, help="the captain's memory (GRU) size")
    p.add_argument("--no-memory", action="store_true", help="ablation: no GRU")
    p.add_argument("--no-commander", action="store_true", help="ablation: every ship always free")
    p.add_argument("--commander-period", type=int, default=10, help="decisions between the commander's orders")
    p.add_argument("--kl-bc", type=float, default=0.1, help="KL anchor to the imitation policy at the start")
    p.add_argument("--ippo", action="store_true", help="IPPO baseline: the critic sees only each ship's own view "
                                                        "(needs --no-commander)")
    p.add_argument("--no-value-norm", action="store_true", help="switch off MAPPO's value normalisation")
    p.add_argument("--seed", type=int, default=0)
    # bookkeeping
    p.add_argument("--run-name", default="fleet")
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--video-every", type=int, default=None,
                   help="record a replay video every N timesteps (default: every save-interval; 0: never)")
    p.add_argument("--status-every", type=int, default=10_000,
                   help="with --team-storage: push a status update (no checkpoint) every N timesteps")
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

    resume = a.resume or pulled
    cloned = os.path.join(models_dir, "imitation.pt")
    if not resume and not a.no_imitation and not os.path.exists(cloned) and a.imitation_battles > 0 and not a.mock:
        imitate(a, run_dir, cloned, storage)          # before the training battles start: one set of players at a time

    env = Monitor(NavalEnv(a.scenario or "", n_envs=a.n_envs, opponent_difficulty=a.opponent_difficulty,
                           stages=None if a.scenario else a.curriculum, start_stage=a.start_stage,
                           decision_period=a.decision_period, sim_dt=a.sim_dt,
                           time_limit=a.time_limit, reward_weights=weights, unity_binary=a.unity_binary,
                           base_port=a.base_port, seed=a.seed, mock=a.mock,
                           unity_log_dir=os.path.join(run_dir, "unity_logs")), log_dir)
    try:
        if resume:
            model = MAPPO.load(resume, env=env, device=a.device, log_dir=log_dir, verbose=1)
            print(f"resuming {resume} at {model.num_timesteps} timesteps")
        else:
            ent = a.ent_coef if a.ent_coef == "auto" else float(a.ent_coef)
            model = MAPPO(env, learning_rate=a.learning_rate, critic_learning_rate=a.critic_learning_rate,
                          n_steps=a.n_steps, n_minibatches=a.n_minibatches, n_epochs=a.n_epochs, gamma=a.gamma,
                          gae_lambda=a.gae_lambda, clip_range=a.clip_range, dual_clip=a.dual_clip or None,
                          ent_coef=ent, d_model=a.d_model, attn_layers=a.attn_layers, hidden_size=a.hidden_size,
                          recurrent=not a.no_memory, commander=not a.no_commander,
                          commander_period=a.commander_period, kl_bc_coef=a.kl_bc, centralised_critic=not a.ippo,
                          normalize_values=not a.no_value_norm, device=a.device, seed=a.seed, log_dir=log_dir)
            if not a.no_imitation and os.path.exists(cloned):
                model.init_from_imitation(cloned)

        t0 = time.time()
        print("stages:", " -> ".join(s["name"] for s in env.stages), f"(starting at stage {env.stage})")
        callbacks = [SaveOnIntervalCallback(a.save_interval, models_dir), CurriculumCallback(),
                     BestModelCallback(models_dir)]
        video_every = a.save_interval if a.video_every is None else a.video_every
        if video_every and not a.mock:                        # the mock environment cannot be drawn
            callbacks.append(VideoCallback(video_every, os.path.join(run_dir, "videos")))
        if storage:                                           # last: it also pushes the newest video
            callbacks.append(TeamSyncCallback(storage, run_dir, push_every=a.save_interval, status_every=a.status_every))
        model.learn(total_timesteps=a.total_timesteps, callback=callbacks)
        model.save(os.path.join(models_dir, "model_final"))
        model.save(os.path.join(models_dir, "model_latest"))
        if storage and callbacks[-1].conflict is None:
            storage.push(run_dir, model, status=training_status(model, run_dir))     # model_final as well
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
        if os.path.isdir(os.path.join(run_dir, "videos")):
            shutil.copytree(os.path.join(run_dir, "videos"), os.path.join(out, "videos"), dirs_exist_ok=True)
        for f in ("config.json", "eval.json"):
            if os.path.exists(os.path.join(run_dir, f)):
                shutil.copy(os.path.join(run_dir, f), out)


def imitate(a, run_dir: str, path: str, storage=None) -> None:
    """Records the Elite rule AI and clones it into a captain (simple_mappo/imitation.py) with the
    same network settings the run's MAPPO will use."""
    print(f"imitation: recording {a.imitation_battles} battles of the Elite rule AI")
    rec = NavalEnv(stages=a.curriculum, n_envs=a.n_envs, decision_period=a.decision_period, sim_dt=a.sim_dt,
                   unity_binary=a.unity_binary, base_port=a.base_port + 100, seed=a.seed + 1, record_expert=True,
                   unity_log_dir=os.path.join(run_dir, "unity_logs"))
    try:
        team_stages = [i for i, st in enumerate(rec.stages) if st.get("opponent", "rule") != "passive"] or [0]
        data = collect_expert(rec, a.imitation_battles, stages=team_stages, seed=a.seed)
        dims, heads = rec.dims, rec.heads
    finally:
        rec.close()
    captain, history = behaviour_cloning(data, dims, heads, d_model=a.d_model, attn_layers=a.attn_layers,
                                         hidden_size=a.hidden_size, recurrent=not a.no_memory,
                                         num_epochs=a.imitation_epochs, device=a.device)
    save_imitation(path, captain, dims, heads, history)
    print("saved", path)
    if storage:
        storage.push(run_dir, None, status={"imitation": history["test_accuracy"][-1]})


if __name__ == "__main__":
    main()
