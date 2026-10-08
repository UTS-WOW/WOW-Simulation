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

import datetime
import getpass
import glob
import json
import os
import shutil
import socket
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

    def push(self, run_dir: str, model=None) -> None:
        """Uploads new and changed files (checkpoints, logs, config) and records the run's status."""
        info = self.info()
        remote_push = info.get("push_id") if info else None
        if remote_push not in (None, self._base_push):
            raise TeamStorageBusy(f"{info['user']} pushed '{self.run_name}' at {info['updated'][:16].replace('T', ' ')} "
                                  f"UTC, after you pulled ({info['num_timesteps']:,} timesteps). Your progress since "
                                  f"then cannot be merged: pull again and continue from theirs, or use another run_name.")
        files = [os.path.join(run_dir, "config.json")] + sorted(glob.glob(os.path.join(run_dir, "logs", "*.csv"))) \
            + sorted(glob.glob(os.path.join(run_dir, "models", "*.pt")))
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
        status = {"run_name": self.run_name, "user": self.user, "session": self.session, "push_id": push_id,
                  "updated": _now().isoformat()}
        if model is not None:
            stage = getattr(model.env, "stage", 0)
            stages = getattr(model.env, "stages", [{"name": ""}])
            status.update({"num_timesteps": model.num_timesteps, "updates": model.n_updates, "stage": stage,
                           "stage_name": stages[stage]["name"]})
        elif info:
            status.update({k: info[k] for k in ("num_timesteps", "updates", "stage", "stage_name") if k in info})
        status.setdefault("num_timesteps", 0)
        self.backend.write("run.json", json.dumps(status, indent=2).encode())
        self._base_push = push_id
        self._write_lock(released=False)                 # heartbeat
        print(f"pushed {sent} file(s) to {self.location} at {status['num_timesteps']:,} timesteps")


class TeamSyncCallback(BaseCallback):
    """Pushes the run to the team's storage every push_every timesteps and when training ends.
    It saves models/model_latest.pt first, so the team always has an up-to-date checkpoint."""

    def __init__(self, storage: TeamStorage, run_dir: str, push_every: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.storage, self.run_dir, self.push_every = storage, run_dir, push_every
        self._last = 0
        self.conflict = None

    def _on_training_start(self) -> None:
        self._last = self.num_timesteps

    def _on_rollout_end(self) -> bool:
        if self.num_timesteps - self._last >= self.push_every:
            return self._push()
        return True

    def _on_training_end(self) -> None:
        if self.conflict is None:
            self._push()

    def _push(self) -> bool:
        self._last = self.num_timesteps
        self.model.save(os.path.join(self.run_dir, "models", "model_latest"))
        try:
            self.storage.push(self.run_dir, self.model)
            return True
        except TeamStorageBusy as e:
            # a teammate pushed this run while we trained: stop rather than build on a stale line
            self.conflict = str(e)
            print(f"*** not pushed - stopping: {e} ***")
            return False
