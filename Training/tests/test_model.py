import torch

from naval_rl.mock_env import mock_spec
from naval_rl.model import Actor, Critic, sample_actions

INIT = {"max_team": 4, "max_allies": 3, "max_contacts": 5, "max_zones": 3, "action_mode": "intent"}


def _obs(spec, B=2, n_allies=3, n_contacts=5, n_zones=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = spec.dims
    return {
        "self": torch.randn(B, d["self"], generator=g),
        "allies": torch.randn(B, n_allies, d["ally"], generator=g), "ally_mask": torch.ones(B, n_allies),
        "contacts": torch.randn(B, n_contacts, d["contact"], generator=g), "contact_mask": torch.ones(B, n_contacts),
        "zones": torch.randn(B, n_zones, d["zone"], generator=g), "zone_mask": torch.ones(B, n_zones),
        "obstacles": torch.randn(B, 4, d["obstacle"], generator=g), "obstacle_mask": torch.ones(B, 4),
    }


def _pad(obs, key, mask_key, extra):
    x = obs[key]
    pad = torch.randn(x.shape[0], extra, x.shape[2]) * 50
    out = dict(obs)
    out[key] = torch.cat([x, pad], 1)
    out[mask_key] = torch.cat([obs[mask_key], torch.zeros(x.shape[0], extra)], 1)
    return out


def test_padding_does_not_change_the_decision():
    spec = mock_spec(INIT)
    actor = Actor(spec, d=32, heads=4, layers=2, hidden=32).eval()
    obs = _obs(spec, n_contacts=3)
    h = torch.zeros(2, 32)
    ref, _, _ = actor(obs, h, torch.zeros(2))
    padded = _pad(obs, "contacts", "contact_mask", 2)
    out, _, _ = actor(padded, h, torch.zeros(2))
    # the target head grows by the two padded slots; every real option scores the same
    sizes = [h.size for h in spec.heads]
    pre = sum(sizes[:2])
    n_target_real = 1 + 3
    torch.testing.assert_close(out[:, :pre], ref[:, :pre], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out[:, pre:pre + n_target_real], ref[:, pre:pre + n_target_real], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out[:, pre + n_target_real + 2:], ref[:, pre + n_target_real:], atol=1e-5, rtol=1e-4)


def test_contact_order_only_permutes_the_pointer_logits():
    spec = mock_spec(INIT)
    actor = Actor(spec, d=32, heads=4, layers=2, hidden=32).eval()
    obs = _obs(spec)
    perm = torch.tensor([3, 0, 4, 1, 2])
    obs2 = dict(obs)
    obs2["contacts"] = obs["contacts"][:, perm]
    h = torch.zeros(2, 32)
    a, _, _ = actor(obs, h, torch.zeros(2))
    b, _, _ = actor(obs2, h, torch.zeros(2))
    sizes = [x.size for x in spec.heads]
    t0 = sum(sizes[:2])
    torch.testing.assert_close(b[:, t0 + 1:t0 + 6], a[:, t0 + 1:t0 + 6][:, perm], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(b[:, :t0 + 1], a[:, :t0 + 1], atol=1e-5, rtol=1e-4)


def test_masked_options_are_never_sampled():
    spec = mock_spec(INIT)
    actor = Actor(spec, d=32, heads=4, layers=1, hidden=32)
    obs = _obs(spec, B=64)
    logits, _, _ = actor(obs, torch.zeros(64, 32), torch.zeros(64))
    sizes = [h.size for h in spec.heads]
    mask = torch.zeros(64, sum(sizes))
    off = 0
    for s in sizes:
        mask[:, off + s - 1] = 1.0          # only the last option of every head is legal
        off += s
    a, lp, _ = sample_actions(logits, mask, sizes)
    for i, s in enumerate(sizes):
        assert (a[:, i] == s - 1).all()
    torch.testing.assert_close(lp, torch.zeros(64), atol=1e-4, rtol=0)


def test_critic_reads_the_right_agent_and_any_team_size():
    spec = mock_spec(INIT)
    critic = Critic(spec, d=32, heads=4, layers=1).eval()
    d = spec.dims
    for n in (2, 7):
        c = {"critic_own": torch.randn(3, n, d["critic_own"]), "own_mask": torch.ones(3, n),
             "critic_enemy": torch.randn(3, n, d["critic_enemy"]), "critic_enemy_mask": torch.ones(3, n),
             "critic_zones": torch.randn(3, 2, d["critic_zone"]), "critic_zone_mask": torch.ones(3, 2),
             "critic_match": torch.randn(3, d["critic_match"])}
        v0, w0 = critic(c, torch.zeros(3, dtype=torch.long))
        v1, w1 = critic(c, torch.ones(3, dtype=torch.long))
        assert v0.shape == (3,) and w0.shape == (3,)
        assert not torch.allclose(v0, v1)
