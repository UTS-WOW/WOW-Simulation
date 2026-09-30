"""Shared checkpoints: published on one machine, resumed on another, never silently overwritten."""

import json
import os
import shutil

import pytest
import torch

from naval_rl import share
from naval_rl.config import Config
from naval_rl.env import init_message, make_workers
from naval_rl.league import League
from naval_rl.mappo import MAPPO

STAGES = [{"name": "mock", "procedural": {"ships": [3, 3]}, "difficulty": 2}]


def _algo(root):
    cfg = Config(workers=2, mock=True, max_team=3, max_allies=2, max_contacts=3, max_zones=3, rollout=16,
                 chunk=8, minibatches=1, d_model=32, hidden=32, layers=1, stages=STAGES,
                 selfplay_from_stage=0, mix_latest=0.0, mix_snapshot=1.0, mix_rule=0.0)
    workers = make_workers(2, init_message(3, 2, 3, 3, "intent", 1.0, 0.02, True), None, 0, mock=True)
    return MAPPO(cfg, workers[0].spec, workers, League(cfg, str(root)), torch.device("cpu"))


def test_published_checkpoint_resumes_on_another_machine(tmp_path):
    # machine A trains, snapshots its actor twice and checkpoints
    run_a = tmp_path / "runs" / "alpha"
    os.makedirs(run_a / "snapshots")
    os.makedirs(run_a / "checkpoints")
    a = _algo(tmp_path)
    for update in (25, 50):
        path = str(run_a / "snapshots" / f"actor_{update:06d}.pt")
        torch.save(a.actor.state_dict(), path)
        a.league.add_snapshot(path, update)
    a.update = 50
    torch.save({**a.state_dict(), "lineage": share.new_line()}, run_a / "checkpoints" / "latest.pt")

    shared = tmp_path / "repo" / "checkpoints" / "alpha"
    info = share.publish(str(run_a / "checkpoints" / "latest.pt"), str(shared), note="first")
    assert info["update"] == 50 and info["use_with"] == "--resume" and info["snapshots"] == 2
    ckpt = torch.load(shared / "latest.pt", weights_only=False)
    assert [s["path"] for s in ckpt["league"]["snapshots"]] == ["snapshots/actor_000025.pt",
                                                               "snapshots/actor_000050.pt"]
    assert "lineage" not in ckpt

    # machine B has only the git checkout: A's runs/ and absolute paths are gone
    shutil.rmtree(tmp_path / "runs")
    clone = tmp_path / "clone" / "checkpoints" / "alpha"
    shutil.copytree(shared, clone)
    b = _algo(tmp_path)
    resume = torch.load(clone / "latest.pt", weights_only=False)
    b.load_state_dict(resume)
    run_b = tmp_path / "runs_b" / "alpha"
    share.localize_snapshots(b.league, str(clone / "latest.pt"), str(run_b))
    assert all(os.path.isfile(s["path"]) and s["path"].startswith(str(run_b)) for s in b.league.snapshots)
    assert b.update == 50
    b._snapshot(b.league.snapshots[0]["id"])       # the self-play opponent actually loads

    line = share.lineage_for_run(resume, str(clone / "latest.pt"))
    assert line["shared"] == "alpha" and line["from_update"] == 50


def _fake(path, update, lineage):
    """A checkpoint with just what publish() reads."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"update": update, "env_steps": update * 100, "config": {"stages": STAGES}, "opt_actor": {},
            "league": {"stage": 0, "window": {0: [1.0, 0.0]}, "snapshots": [], "next_snapshot": 0}}
    if lineage is not None:
        ckpt["lineage"] = lineage
    torch.save(ckpt, path)
    return str(path)


def test_publish_never_discards_a_teammates_training(tmp_path):
    shared = str(tmp_path / "checkpoints" / "main")
    alice = share.new_line()
    share.publish(_fake(tmp_path / "alice.pt", 10, alice), shared)

    # Bob pulls update 10 and starts his own line from it
    bob = share.lineage_for_run({"update": 10}, os.path.join(shared, "latest.pt"))
    assert bob["shared"] == "main" and bob["from_update"] == 10 and bob["line"] != alice["line"]

    # Alice keeps going on her own line: fine
    share.publish(_fake(tmp_path / "alice.pt", 15, alice), shared)
    # Alice publishing something older than what is shared: refused
    with pytest.raises(SystemExit, match="nothing new"):
        share.publish(_fake(tmp_path / "alice_old.pt", 12, alice), shared)
    # Bob, further along but started before Alice's update 15: refused, it would drop her work
    with pytest.raises(SystemExit, match="published update 15"):
        share.publish(_fake(tmp_path / "bob.pt", 20, bob), shared)
    # an unrelated run cannot take the name over
    with pytest.raises(SystemExit, match="did not start from"):
        share.publish(_fake(tmp_path / "other.pt", 99, share.new_line()), shared)
    # under his own name Bob's run is kept alongside
    share.publish(_fake(tmp_path / "bob.pt", 20, bob), str(tmp_path / "checkpoints" / "main_bob"))
    # and --force is the explicit way to replace it
    share.publish(_fake(tmp_path / "bob.pt", 20, bob), shared, force=True)

    with open(os.path.join(shared, share.INFO)) as f:
        assert json.load(f)["update"] == 20
    assert [h["update"] for h in share._history(shared)] == [10, 15, 20]
    assert [r["name"] for r in share.list_shared(str(tmp_path / "checkpoints"))] == ["main", "main_bob"]


def test_old_checkpoint_is_stamped_so_its_run_can_keep_publishing(tmp_path):
    src = _fake(tmp_path / "old.pt", 9, lineage=None)
    shared = str(tmp_path / "checkpoints" / "full_mappo")
    share.publish(src, shared)
    stamped = torch.load(src, weights_only=False)["lineage"]
    assert stamped["shared"] == "full_mappo" and stamped["from_update"] == 9
    # training resumed from that local file stays on the stamped line and may publish again
    later = share.lineage_for_run(torch.load(src, weights_only=False), src)
    share.publish(_fake(tmp_path / "later.pt", 30, later), shared)


def test_incompatible_checkpoint_is_refused_with_a_readable_message(tmp_path):
    a = _algo(tmp_path)
    saved = a.spec.to_json()
    saved["dims"] = {**saved["dims"], "self": saved["dims"]["self"] + 1}
    with pytest.raises(SystemExit, match="self: checkpoint"):
        share.check_compatible(saved, a.spec, "old.pt")
    share.check_compatible(a.spec.to_json(), a.spec, "same.pt")    # same layout: no complaint
