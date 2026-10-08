"""Team checkpoints: one shared folder per training run, on S3 (or any shared folder).

Everyone on the team trains the same lines of training in turns - from a SageMaker notebook, a
SageMaker training job or their own computer - and they all read and write the same layout:

    s3://<bucket>/wow-mappo/runs/<run-name>/
        run.json              who pushed last, when, how far (timesteps, curriculum stage)
        LOCK.json             who is training this run right now
        config.json           the settings the run was started with
        logs/monitor.csv      one line per battle, from every session - the team's learning curve
        logs/progress.csv     one line per update
        models/model_latest.pt     the checkpoint to continue from
        models/model_<n>.pt        kept checkpoints, models/model_final.pt
        videos/battle_<timesteps>_stage<k>.mp4   replays recorded while training

    storage.show_status()                    # how the run is doing: stage, progress, win rate, ETA
    storage.pull_view("team_view")           # its logs and videos, to plot and watch

    storage = TeamStorage("s3://my-bucket/wow-mappo", "curriculum", user="Lukita")
    latest = storage.pull("runs_simple/curriculum")      # None if the run is new
    storage.lock()                                        # refuses if a teammate is training it now
    ... train, with TeamSyncCallback(storage, run_dir) pushing every so often ...
    storage.push("runs_simple/curriculum"); storage.release()

Two rules keep teammates from overwriting each other:
* lock() refuses while someone else holds the run and has pushed within lock_minutes.
* push() refuses if someone else pushed after you pulled - pull again (you lose only your own
  unpushed training, never theirs).

A plain folder path (a shared drive, or any local folder for testing) works in place of s3://.
"""

from __future__ import annotations

import csv
import datetime
import getpass
import glob
import json
import os
import shutil
import socket
import time
import uuid

from .callbacks import BaseCallback


class TeamStorageBusy(RuntimeError):
    """Someone else is training this run, or pushed since you pulled."""


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class _Local:
    def __init__(self, root: str):
        self.root = root

    def read(self, key: str) -> bytes | None:
        path = os.path.join(self.root, key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def write(self, key: str, data: bytes) -> None:
        path = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def download(self, key: str, path: str) -> bool:
        src = os.path.join(self.root, key)
        if not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copyfile(src, path)
        return True

    def upload(self, path: str, key: str) -> None:
        dst = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(path, dst)

    def list(self, folder: str) -> list[str]:
        return sorted(f"{folder}/{n}" for n in os.listdir(os.path.join(self.root, folder))) \
            if os.path.isdir(os.path.join(self.root, folder)) else []


class _S3:
    def __init__(self, uri: str):
        import boto3                                   # on SageMaker; locally: pip install boto3
        bucket, _, prefix = uri[len("s3://"):].partition("/")
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.s3 = boto3.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _missing(self, error) -> bool:
        """True for "no such file". Without s3:ListBucket permission S3 says AccessDenied instead."""
        code = error.response["Error"]["Code"]
        if code in ("404", "NoSuchKey", "NotFound"):
            return True
        if code in ("403", "AccessDenied"):
            raise PermissionError(f"no access to s3://{self.bucket}/{self.prefix} - the SageMaker execution role "
                                  f"needs s3:GetObject, s3:PutObject and s3:ListBucket on this bucket (the default "
                                  f"sagemaker-<region>-<account> bucket has them)") from error
        return False

    def read(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except ClientError as e:
            if self._missing(e):
                return None
            raise

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as e:
            if self._missing(e):
                return False
            raise

    def write(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def download(self, key: str, path: str) -> bool:
        if not self.exists(key):
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.s3.download_file(self.bucket, self._key(key), path)
        return True

    def upload(self, path: str, key: str) -> None:
        self.s3.upload_file(path, self.bucket, self._key(key))

    def list(self, folder: str) -> list[str]:
        keys, cut = [], len(self.prefix) + 1 if self.prefix else 0
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self._key(folder) + "/"):
            keys += [o["Key"][cut:] for o in page.get("Contents", [])]
        return sorted(keys)


class TeamStorage:
    def __init__(self, uri: str, run_name: str, user: str | None = None, lock_minutes: float = 30.0):
        """
        uri           the team's folder, e.g. s3://sagemaker-ap-southeast-2-123456789012/wow-mappo
                      (or a shared / local folder path)
        run_name      one line of training; the team takes turns extending it
        user          your name, shown to teammates (on SageMaker every login is "sagemaker-user",
                      so set it)
        lock_minutes  a lock older than this (no push in that time) counts as abandoned
        """
        base = f"runs/{run_name}"
        self.uri, self.run_name, self.lock_minutes = uri.rstrip("/"), run_name, lock_minutes
        self.backend = _S3(f"{self.uri}/{base}") if uri.startswith("s3://") else _Local(os.path.join(uri, base))
        self.user = user or os.environ.get("TEAM_USER") or getpass.getuser()
        self.session = f"{self.user}@{socket.gethostname()}#{uuid.uuid4().hex[:6]}"
        self._base_push = None          # the push our training continues from
        self._uploaded: dict[str, float] = {}

    @property
    def location(self) -> str:
        return f"{self.uri}/runs/{self.run_name}"

    # ------------------------------------------------------------------ status

    def info(self) -> dict | None:
        """run.json: who pushed last, when, timesteps and stage - or None for a new run."""
        raw = self.backend.read("run.json")
        return json.loads(raw) if raw else None

    def _lock(self) -> dict | None:
        raw = self.backend.read("LOCK.json")
        return json.loads(raw) if raw else None

    def _lock_is_live(self, lock: dict | None) -> bool:
        if not lock or lock.get("released") or lock.get("session") == self.session:
            return False
        age = _now() - datetime.datetime.fromisoformat(lock["heartbeat"])
        return age.total_seconds() < self.lock_minutes * 60

    # ------------------------------------------------------------------ pull / lock / push

    def pull(self, run_dir: str) -> str | None:
        """Downloads the run's latest checkpoint, logs and config into run_dir.
        Returns the path of models/model_latest.pt, or None when the run has no checkpoint yet."""
        info = self.info()
        self._base_push = info.get("push_id") if info else None
        if not info:
            return None
        for key in ("config.json", "logs/monitor.csv", "logs/progress.csv", "models/model_latest.pt"):
            path = os.path.join(run_dir, key)
            if self.backend.download(key, path):
                self._uploaded[key] = os.path.getmtime(path)          # already in the team's folder
        latest = os.path.join(run_dir, "models", "model_latest.pt")
        if "models/model_latest.pt" not in self._uploaded:
            return None
        print(f"pulled {self.location}: {info['num_timesteps']:,} timesteps, stage '{info.get('stage_name', '')}', "
              f"last pushed by {info['user']} at {info['updated'][:16].replace('T', ' ')} UTC")
        return latest

    def lock(self, force: bool = False) -> None:
        """Claims the run. force=True takes over a lock you know is stale (e.g. a crashed session)."""
        held = self._lock()
        if self._lock_is_live(held) and not force:
            raise TeamStorageBusy(f"{held['user']} is training '{self.run_name}' (last heartbeat "
                                  f"{held['heartbeat'][:16].replace('T', ' ')} UTC). Wait for them, train a different "
                                  f"run_name, or lock(force=True) if that session is dead.")
        self._write_lock(released=False)

    def release(self) -> None:
        held = self._lock()
        if held and held.get("session") == self.session:
            self._write_lock(released=True)

    def _write_lock(self, released: bool) -> None:
        self.backend.write("LOCK.json", json.dumps({"user": self.user, "session": self.session,
                                                    "heartbeat": _now().isoformat(), "released": released}).encode())

    def push(self, run_dir: str, model=None, status: dict | None = None, models: bool = True) -> None:
        """Uploads new and changed files (checkpoints, logs, videos, config) and records the run's
        status in run.json. models=False sends everything but the checkpoints (a quick status update)."""
        info = self.info()
        remote_push = info.get("push_id") if info else None
        if remote_push not in (None, self._base_push):
            raise TeamStorageBusy(f"{info['user']} pushed '{self.run_name}' at {info['updated'][:16].replace('T', ' ')} "
                                  f"UTC, after you pulled ({info['num_timesteps']:,} timesteps). Your progress since "
                                  f"then cannot be merged: pull again and continue from theirs, or use another run_name.")
        files = [os.path.join(run_dir, "config.json")] + sorted(glob.glob(os.path.join(run_dir, "logs", "*.csv"))) \
            + sorted(glob.glob(os.path.join(run_dir, "videos", "*.mp4")) + glob.glob(os.path.join(run_dir, "videos", "*.gif")))
        if models:
            files += sorted(glob.glob(os.path.join(run_dir, "models", "*.pt")))
        sent = 0
        for path in files:
            if not os.path.exists(path):
                continue
            rel = os.path.relpath(path, run_dir).replace(os.sep, "/")
            if self._uploaded.get(rel) == os.path.getmtime(path):
                continue
            self.backend.upload(path, rel)
            self._uploaded[rel] = os.path.getmtime(path)
            sent += 1
        push_id = uuid.uuid4().hex
        record = {"run_name": self.run_name, "user": self.user, "session": self.session, "push_id": push_id,
                  "updated": _now().isoformat()}
        if model is not None:
            stage = getattr(model.env, "stage", 0)
            stages = getattr(model.env, "stages", [{"name": ""}])
            record.update({"num_timesteps": model.num_timesteps, "updates": model.n_updates, "stage": stage,
                           "stage_name": stages[stage]["name"]})
        elif info:
            record.update({k: info[k] for k in ("num_timesteps", "updates", "stage", "stage_name") if k in info})
        record.setdefault("num_timesteps", 0)
        # the timesteps of the checkpoint a teammate would continue from
        record["checkpoint_timesteps"] = record["num_timesteps"] if models else (info or {}).get("checkpoint_timesteps", 0)
        record.update(status or {})
        self.backend.write("run.json", json.dumps(record, indent=2).encode())
        self._base_push = push_id
        self._write_lock(released=False)                 # heartbeat
        what = "checkpoint" if models else "status"
        print(f"pushed {what} ({sent} file(s)) to {self.location} at {record['num_timesteps']:,} timesteps")

    # ------------------------------------------------------------------ watching a run

    def show_status(self) -> dict | None:
        """Prints how the run is doing - from anywhere, while a teammate or a training job trains it."""
        info = self.info()
        if not info:
            print(f"{self.location}: no training pushed yet")
            return None
        lock = self._lock()
        age_min = (_now() - datetime.datetime.fromisoformat(info["updated"])).total_seconds() / 60
        training = bool(lock and not lock.get("released")
                        and (_now() - datetime.datetime.fromisoformat(lock["heartbeat"])).total_seconds() < self.lock_minutes * 60)
        lines = [f"run           {self.location}",
                 f"state         {'TRAINING now by ' + lock['user'] if training else 'idle - free to continue'}",
                 f"last update   {info['updated'][:16].replace('T', ' ')} UTC by {info['user']} ({age_min:.0f} min ago)",
                 f"stage         {info.get('stage_name', info.get('stage', '?'))}"]
        if info.get("promote_win_rate") is not None:
            lines[-1] += (f"  (stops at {info['promote_win_rate']:.0%} over {info.get('window', 250)} held-out battles)"
                          if info.get("final") else
                          f"  (moves on at {info['promote_win_rate']:.0%} over {info.get('window', 100)} battles)")
        total = info.get("total_timesteps")
        prog = f"{info['num_timesteps']:,}" + (f" / {total:,} timesteps ({info['num_timesteps'] / total:.0%})" if total else " timesteps")
        if info.get("fps"):
            prog += f", {info['fps']:.0f} timesteps/s"
        if total and info["num_timesteps"] >= total:
            prog += ", finished"
        elif info.get("eta_min") is not None:
            prog += f", about {info['eta_min'] / 60:.1f} h left" if info["eta_min"] >= 90 else f", about {info['eta_min']:.0f} min left"
        lines.append(f"progress      {prog}")
        lines.append(f"checkpoint    model_latest.pt at {info.get('checkpoint_timesteps', info['num_timesteps']):,} timesteps")
        if info.get("win_rate_recent") is not None:
            lines.append(f"win rate      {info['win_rate_recent']:.2f} over the last {info['recent_battles']} battles against the "
                         f"rule AI on this stage ({info['stage_battles']} played on it); reward/ship {info['reward_recent']:+.2f}")
        if info.get("latest_video"):
            lines.append(f"latest video  videos/{info['latest_video']}  ({info.get('videos', 1)} so far)")
        print("\n".join(lines))
        return info

    def pull_view(self, local_dir: str, videos: int = 3) -> str:
        """Downloads the run's logs and its newest `videos` replays into local_dir (to plot and watch
        without disturbing whoever is training it). Returns local_dir."""
        for key in ("logs/monitor.csv", "logs/progress.csv"):
            self.backend.download(key, os.path.join(local_dir, key))
        for key in [k for k in self.backend.list("videos") if k.endswith((".mp4", ".gif"))][-videos:] if videos else []:
            self.backend.download(key, os.path.join(local_dir, key))
        return local_dir


def training_status(model, run_dir: str) -> dict:
    """What run.json reports about a run in progress: speed, time left, recent results, videos."""
    env = model.env
    stage = getattr(env, "stage", 0)
    st = getattr(env, "stages", [{}])[stage]
    elapsed = max(1e-6, time.time() - (model._t0 or time.time()))
    fps = (model.num_timesteps - model._start_timesteps) / elapsed
    total = model._total_timesteps
    out = {"total_timesteps": total, "fps": round(fps, 1), "elapsed_min": round(elapsed / 60, 1),
           "eta_min": round(max(0, total - model.num_timesteps) / fps / 60, 1) if total and fps > 0 else None,
           "promote_win_rate": st.get("promote_win_rate", st.get("stop_win_rate")),
           "window": st.get("window", st.get("stop_window")),
           "final": "promote_win_rate" not in st and "stop_win_rate" in st}
    path = os.path.join(run_dir, "logs", "monitor.csv")
    if os.path.exists(path):
        with open(path) as f:
            next(f)                                      # the '#{...}' header line
            rows = [r for r in csv.DictReader(f) if r.get("opponent", "rule") == "rule"
                    and int(float(r.get("stage") or 0)) == stage]
        recent = rows[-int(out["window"] or 100):]
        out["stage_battles"] = len(rows)
        if recent:
            out.update({"recent_battles": len(recent),
                        "win_rate_recent": round(sum(float(r["won"]) for r in recent) / len(recent), 3),
                        "reward_recent": round(sum(float(r["r"]) for r in recent) / len(recent), 3)})
    vids = sorted(glob.glob(os.path.join(run_dir, "videos", "*.mp4")) + glob.glob(os.path.join(run_dir, "videos", "*.gif")))
    if vids:
        out.update({"videos": len(vids), "latest_video": os.path.basename(vids[-1])})
    return out


class TeamSyncCallback(BaseCallback):
    """Keeps the team's copy of the run up to date while training:

    * every status_every timesteps a quick status update - logs, new videos and run.json's live
      figures (stage, progress, speed, time left, recent win rate), no checkpoint;
    * every push_every timesteps and at the end, the same plus models/model_latest.pt (and any
      other new checkpoints), so the team always has a recent checkpoint to continue from.
    """

    def __init__(self, storage: TeamStorage, run_dir: str, push_every: int = 50_000, status_every: int = 10_000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.storage, self.run_dir, self.push_every, self.status_every = storage, run_dir, push_every, status_every
        self._last = self._last_status = 0
        self.conflict = None

    def _on_training_start(self) -> None:
        self._last = self._last_status = self.num_timesteps

    def _on_rollout_end(self) -> bool:
        if self.num_timesteps - self._last >= self.push_every:
            return self._push(models=True)
        if self.status_every and self.num_timesteps - self._last_status >= self.status_every:
            return self._push(models=False)
        return True

    def _on_training_end(self) -> None:
        if self.conflict is None:
            self._push(models=True)

    def _push(self, models: bool) -> bool:
        self._last_status = self.num_timesteps
        if models:
            self._last = self.num_timesteps
            self.model.save(os.path.join(self.run_dir, "models", "model_latest"))
        try:
            self.storage.push(self.run_dir, self.model, status=training_status(self.model, self.run_dir), models=models)
            return True
        except TeamStorageBusy as e:
            # a teammate pushed this run while we trained: stop rather than build on a stale line
            self.conflict = str(e)
            print(f"*** not pushed - stopping: {e} ***")
            return False
