"""SageMaker helpers built on boto3 only, so they work with any SageMaker Python SDK version.

(SageMaker Distribution 4.x ships SDK v3, where `sagemaker.Session` and `sagemaker.get_execution_role`
no longer exist at the top level and the old `PyTorch` estimator is gone.)

    TEAM_STORAGE = team_storage_uri()            # s3://sagemaker-<region>-<account>/wow-mappo
    job = launch_training_job("curriculum", TEAM_STORAGE, user="Lukita-job",
                              hyperparameters={"total-timesteps": 20_000_000, "n-envs": 8})
    job_status(job)                              # AWS's view of the job + its newest log lines
    stop_training_job(job)

A training job runs train_simple.py in AWS's PyTorch CPU container with the headless player as its
"unity" input channel, and trains the team run in TEAM_STORAGE - pulling its latest checkpoint,
pushing status, videos and checkpoints while it runs, exactly like the notebook does.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import re
import tarfile

from .env import TRAINING_DIR, find_unity_binary

# AWS's PyTorch training image, CPU build: Ubuntu 22.04 (the Unity player needs its glibc 2.35), Python 3.12
TRAINING_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker"
JOB_CODE = ["simple_mappo", "naval_rl", "scenarios", "train_simple.py"]
JOB_REQUIREMENTS = "matplotlib\npillow\n"     # for the replay videos; PyTorch, NumPy and boto3 are in the image
# numbers from the training table, charted by SageMaker on the job's page (Monitor -> Algorithm metrics)
METRICS = [{"Name": name, "Regex": rf"\| {name} +\| +([-0-9.e+]+) +\|"}
           for name in ("total_timesteps", "stage", "win_rate", "win_rate_self_play", "ep_rew_mean", "fps",
                        "entropy", "explained_variance", "orders_free", "elo")]


def _session():
    import boto3
    return boto3.session.Session()


def region() -> str:
    r = _session().region_name or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not r:
        raise RuntimeError("no AWS region configured")
    return r


def account() -> str:
    return _session().client("sts").get_caller_identity()["Account"]


def default_bucket() -> str:
    """The account's SageMaker bucket, sagemaker-<region>-<account>; created if it does not exist yet."""
    from botocore.exceptions import ClientError
    name, reg = f"sagemaker-{region()}-{account()}", region()
    s3 = _session().client("s3")
    try:
        s3.head_bucket(Bucket=name)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise PermissionError(f"cannot use the bucket {name} ({code}). Ask for a bucket your SageMaker role can "
                                  f"read and write, and set TEAM_STORAGE = 's3://<that bucket>/wow-mappo'") from e
        s3.create_bucket(Bucket=name, **({} if reg == "us-east-1" else
                                         {"CreateBucketConfiguration": {"LocationConstraint": reg}}))
    return name


def team_storage_uri(prefix: str = "wow-mappo") -> str:
    return f"s3://{default_bucket()}/{prefix}"


def execution_role() -> str:
    """The IAM role training jobs run as - this Studio user's execution role."""
    meta = "/opt/ml/metadata/resource-metadata.json"       # Studio: the user profile's (or domain's) role
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                m = json.load(f)
            sm, domain = _session().client("sagemaker"), m.get("DomainId")
            profile = m.get("UserProfileName")
            if not profile and m.get("SpaceName"):
                profile = sm.describe_space(DomainId=domain, SpaceName=m["SpaceName"]).get(
                    "OwnershipSettings", {}).get("OwnerUserProfileName")
            if profile:
                role = sm.describe_user_profile(DomainId=domain, UserProfileName=profile).get(
                    "UserSettings", {}).get("ExecutionRole")
                if role:
                    return role
            role = sm.describe_domain(DomainId=domain).get("DefaultUserSettings", {}).get("ExecutionRole")
            if role:
                return role
        except Exception:
            pass                                           # e.g. not allowed to describe the domain
    try:                                                   # SageMaker Python SDK v3
        from sagemaker.core.helper.session_helper import get_execution_role
        return get_execution_role()
    except Exception:
        pass
    try:                                                   # SageMaker Python SDK v2
        import sagemaker
        return sagemaker.get_execution_role()
    except Exception:
        pass
    raise RuntimeError("could not find the SageMaker execution role - pass role='arn:aws:iam::...:role/...'")


# ------------------------------------------------------------------ training jobs

def _upload_player(s3, bucket: str, prefix: str) -> str:
    """Uploads the headless player once per build; later jobs reuse it."""
    player = os.path.dirname(find_unity_binary())
    stamp = json.dumps(sorted((os.path.relpath(os.path.join(r, n), player), os.path.getsize(os.path.join(r, n)))
                              for r, _, files in os.walk(player) for n in files if "DoNotShip" not in r))
    try:
        if s3.get_object(Bucket=bucket, Key=f"{prefix}/.build")["Body"].read().decode() == stamp:
            return f"s3://{bucket}/{prefix}/"
    except s3.exceptions.NoSuchKey:
        pass
    print("uploading the headless player (about 100 MB, once per build)...")
    for root, _, files in os.walk(player):
        if "DoNotShip" in root:
            continue
        for n in files:
            path = os.path.join(root, n)
            s3.upload_file(path, bucket, f"{prefix}/{os.path.relpath(path, player)}")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/.build", Body=stamp.encode())
    return f"s3://{bucket}/{prefix}/"


def _code_archive() -> bytes:
    """sourcedir.tar.gz: the code the job runs, and the extra packages it installs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in JOB_CODE:
            tar.add(os.path.join(TRAINING_DIR, item), arcname=item,
                    filter=lambda t: None if "__pycache__" in t.name else t)
        req = tarfile.TarInfo("requirements.txt")
        req.size = len(JOB_REQUIREMENTS)
        tar.addfile(req, io.BytesIO(JOB_REQUIREMENTS.encode()))
    return buf.getvalue()


def training_job_request(job_name: str, role: str, region_name: str, code_uri: str, player_uri: str, output_uri: str,
                         hyperparameters: dict, instance_type: str, max_hours: float, volume_gb: int,
                         image: str | None = None) -> dict:
    """The CreateTrainingJob request. Hyperparameters become train_simple.py's --flags."""
    hp = {k: json.dumps(v) for k, v in hyperparameters.items()}
    hp.update({"sagemaker_program": json.dumps("train_simple.py"), "sagemaker_submit_directory": json.dumps(code_uri),
               "sagemaker_region": json.dumps(region_name)})
    return {
        "TrainingJobName": job_name,
        "RoleArn": role,
        "AlgorithmSpecification": {"TrainingImage": image or TRAINING_IMAGE.format(region=region_name),
                                   "TrainingInputMode": "File", "MetricDefinitions": METRICS},
        "HyperParameters": hp,
        "InputDataConfig": [{"ChannelName": "unity", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": player_uri, "S3DataDistributionType": "FullyReplicated"}}}],
        "OutputDataConfig": {"S3OutputPath": output_uri},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": volume_gb},
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_hours * 3600)},
        "Tags": [{"Key": "project", "Value": "wow-mappo"}],
    }


def launch_training_job(run_name: str, team_storage: str, user: str, hyperparameters: dict | None = None,
                        instance_type: str = "ml.c5.4xlarge", max_hours: float = 48, volume_gb: int = 30,
                        role: str | None = None, image: str | None = None) -> str:
    """Starts a SageMaker training job that continues the team run `run_name`. Returns the job name.

    instance_type  CPUs matter (one battle per core pair); ml.c5.4xlarge has 16 vCPU for 8 battles
    max_hours      the job stops after this (it pushes a checkpoint every save-interval, so a new job
                   simply continues); SageMaker's limit is 5 days
    """
    if not team_storage or not team_storage.startswith("s3://"):
        raise ValueError("a training job needs TEAM_STORAGE on S3 - it is where the job saves its checkpoints")
    bucket, _, prefix = team_storage[len("s3://"):].partition("/")
    prefix = prefix.strip("/") or "wow-mappo"
    reg, s3 = region(), _session().client("s3")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = re.sub(r"[^A-Za-z0-9-]", "-", f"wow-{run_name}")[:44] + f"-{stamp}"     # SageMaker's naming rules
    player_uri = _upload_player(s3, bucket, f"{prefix}/unity")
    code_key = f"{prefix}/jobs/{job_name}/sourcedir.tar.gz"
    s3.put_object(Bucket=bucket, Key=code_key, Body=_code_archive())
    request = training_job_request(job_name, role or execution_role(), reg, f"s3://{bucket}/{code_key}", player_uri,
                                   f"s3://{bucket}/{prefix}/jobs", {"run-name": run_name, "team-storage": team_storage,
                                                                   "user": user, **(hyperparameters or {})},
                                   instance_type, max_hours, volume_gb, image)
    from botocore.exceptions import ClientError
    try:
        _session().client("sagemaker").create_training_job(**request)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("AccessDeniedException", "AccessDenied"):
            raise PermissionError(
                "this AWS account does not allow SageMaker training jobs for your role (course accounts often "
                "block them). Train in the notebook instead: the team folder keeps every checkpoint, so each "
                "notebook session - yours or a teammate's - continues where the last one stopped.") from e
        if code == "ResourceLimitExceeded":
            raise RuntimeError(f"no training-job quota for {instance_type} in this account - try another "
                               f"instance_type, ask for a quota increase, or train in the notebook") from e
        raise
    print(f"started training job {job_name} on {instance_type}")
    print(f"  console: https://{reg}.console.aws.amazon.com/sagemaker/home?region={reg}#/jobs/{job_name}")
    return job_name


def job_status(job_name: str, log_lines: int = 15) -> dict:
    """Prints the job's state as AWS sees it and the newest lines of its log."""
    sm = _session().client("sagemaker")
    d = sm.describe_training_job(TrainingJobName=job_name)
    start, end = d.get("TrainingStartTime"), d.get("TrainingEndTime")
    now = datetime.datetime.now(datetime.timezone.utc)
    ran = ((end or now) - start).total_seconds() / 3600 if start else 0.0
    print(f"job       {job_name}")
    print(f"status    {d['TrainingJobStatus']} - {d.get('SecondaryStatus', '')}"
          + (f" after {ran:.1f} h" if start else "") + f" on {d['ResourceConfig']['InstanceType']}")
    if d.get("FailureReason"):
        print(f"failure   {d['FailureReason']}")
    for m in d.get("FinalMetricDataList", [])[:6]:
        print(f"metric    {m['MetricName']} = {m['Value']:.4g}")
    if log_lines:
        logs = _session().client("logs")
        group = "/aws/sagemaker/TrainingJobs"
        try:
            streams = logs.describe_log_streams(logGroupName=group, logStreamNamePrefix=job_name)["logStreams"]
        except logs.exceptions.ResourceNotFoundException:
            streams = []
        if streams:
            events = logs.get_log_events(logGroupName=group, logStreamName=streams[0]["logStreamName"],
                                         limit=log_lines, startFromHead=False)["events"]
            print("log (newest lines):")
            for e in events:
                print("   ", e["message"].rstrip()[:160])
        else:
            print("log       not written yet (the job is still starting)")
    return d


def stop_training_job(job_name: str) -> None:
    """Stops the job. It is stopped without a final push: the team keeps its last checkpoint push, and
    its lock frees itself after 30 minutes (or continue at once with lock(force=True) / --force-lock)."""
    _session().client("sagemaker").stop_training_job(TrainingJobName=job_name)
    print(f"stopping {job_name}")


def list_training_jobs(max_results: int = 10) -> list[dict]:
    """The account's most recent training jobs of this project."""
    jobs = _session().client("sagemaker").list_training_jobs(NameContains="wow-", SortBy="CreationTime",
                                                             SortOrder="Descending", MaxResults=max_results)
    for j in jobs["TrainingJobSummaries"]:
        print(f"{j['TrainingJobName']:<60} {j['TrainingJobStatus']:<12} {j['CreationTime']:%Y-%m-%d %H:%M}")
    return jobs["TrainingJobSummaries"]
