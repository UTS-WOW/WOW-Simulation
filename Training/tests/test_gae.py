import numpy as np

from naval_rl.mappo import MAPPO


class _Cfg:
    rollout = 4
    gamma = 0.9
    lam = 0.8


def test_gae_stops_at_episode_end_and_bootstraps_truncation():
    algo = MAPPO.__new__(MAPPO)
    algo.cfg = _Cfg()
    b = {
        "rewards": np.array([1.0, 0.0, 2.0, 1.0]).reshape(4, 1, 1, 1),
        "values": np.array([0.5, 0.4, 0.3, 0.2, 10.0]).reshape(5, 1, 1, 1),
        "dones": np.array([0.0, 1.0, 0.0, 0.0]).reshape(4, 1, 1),
    }
    algo._buf = b
    algo.advantages()
    g, l = 0.9, 0.8
    v = [0.5, 0.4, 0.3, 0.2, 10.0]
    r = [1.0, 0.0, 2.0, 1.0]
    d3 = r[3] + g * v[4] - v[3]               # truncated: bootstrap from V(s_T)
    d2 = r[2] + g * v[3] - v[2]
    d1 = r[1] - v[1]                          # terminal: no bootstrap
    d0 = r[0] + g * v[1] - v[0]
    a3 = d3
    a2 = d2 + g * l * a3
    a1 = d1                                   # does not see the next episode
    a0 = d0 + g * l * a1
    np.testing.assert_allclose(b["advantages"].ravel(), [a0, a1, a2, a3], rtol=1e-6)
    np.testing.assert_allclose(b["returns"].ravel(), np.array([a0, a1, a2, a3]) + v[:4], rtol=1e-6)
