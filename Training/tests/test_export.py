import json
import struct

from naval_rl.config import Config
from naval_rl.export import export_parity_case, export_policy
from naval_rl.mock_env import mock_spec
from naval_rl.model import Actor, Critic, ValueNorm


def test_export_layout(tmp_path):
    spec = mock_spec({"max_team": 3, "max_allies": 2, "max_contacts": 4, "max_zones": 2})
    cfg = Config(d_model=32, heads=4, layers=1, hidden=32)
    actor, critic = Actor(spec, 32, 4, 1, 32), Critic(spec, 32, 4, 1)
    path = export_policy(actor, critic, spec, cfg, str(tmp_path / "p.bin"), ValueNorm())
    raw = open(path, "rb").read()
    assert raw[:4] == b"NAVP"
    version, hlen = struct.unpack_from("<ii", raw, 4)
    header = json.loads(raw[12:12 + hlen])
    n_floats = (len(raw) - 12 - hlen) // 4
    assert version == 1 and header["has_critic"]
    last = header["tensors"][-1]
    size = 1
    for s in last["shape"]:
        size *= s
    assert last["offset"] + size == n_floats
    names = {t["name"] for t in header["tensors"]}
    assert "actor.gru.weight_ih" in names and "critic.win.weight" in names
    case = json.load(open(export_parity_case(actor, critic, spec, str(tmp_path / "c.json"))))
    assert len(case["logits"]) == 13 + 3 + 5 + 1 + 4 + 2 + 2 + 15
