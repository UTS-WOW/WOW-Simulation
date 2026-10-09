"""The C# in-game policy (Assets/Scripts/RL/RLPolicy.cs) must match PyTorch to 1e-3."""

import os
import shutil
import subprocess

import pytest
import torch

from naval_rl.config import Config
from naval_rl.export import export_parity_case, export_policy
from naval_rl.mock_env import mock_spec
from naval_rl.model import Actor, Critic, ValueNorm

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.skipif(shutil.which("dotnet") is None, reason="dotnet SDK not installed")
@pytest.mark.parametrize("counts", [(3, 4, 3, 5), (0, 1, 1, 0), (7, 9, 5, 8)])
def test_csharp_policy_matches_pytorch(tmp_path, counts):
    torch.manual_seed(1)
    spec = mock_spec({"max_team": 8, "max_allies": 7, "max_contacts": 9, "max_zones": 5})
    cfg = Config(d_model=32, heads=4, layers=2, hidden=24)
    actor, critic = Actor(spec, 32, 4, 2, 24), Critic(spec, 32, 4, 2)
    policy = export_policy(actor, critic, spec, cfg, str(tmp_path / "p.bin"), ValueNorm())
    na, nc, nz, no = counts
    case = export_parity_case(actor, critic, spec, str(tmp_path / "c.json"), n_allies=na, n_contacts=nc, n_zones=nz,
                              n_obstacles=no)
    r = subprocess.run(["dotnet", "run", "--project", os.path.join(HERE, "parity"), "--", policy, case],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout, r.stderr)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------------------------------------- simple_mappo (v2)

@pytest.mark.skipif(shutil.which("dotnet") is None, reason="dotnet SDK not installed")
@pytest.mark.parametrize("counts,order", [((2, 3, 2, 2), 0), ((0, 1, 1, 0), 1), ((7, 8, 5, 4), 3), ((3, 2, 3, 1), 4)])
def test_csharp_matches_simple_mappo(tmp_path, counts, order):
    """The captain and commander of simple_mappo, exported for the game, give the same numbers in C#."""
    from simple_mappo.env import NavalEnv
    from simple_mappo.export import export_parity_case, export_policy
    from simple_mappo.mappo import MAPPO

    env = NavalEnv(stages="default", start_stage=4, n_envs=1, mock=True)
    try:
        torch.manual_seed(2)
        model = MAPPO(env, d_model=32, attn_layers=2, hidden_size=24, verbose=0)
        policy = export_policy(model, str(tmp_path / "p.bin"))
        na, nc, nz, no = counts
        case = export_parity_case(model, str(tmp_path / "c.json"), n_allies=na, n_contacts=nc, n_zones=nz,
                                  n_obstacles=no, order=order, n_own=3, n_enemy=4)
    finally:
        env.close()
    r = subprocess.run(["dotnet", "run", "--project", os.path.join(HERE, "parity"), "--", policy, case],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout, r.stderr)
    assert r.returncode == 0, r.stdout + r.stderr
