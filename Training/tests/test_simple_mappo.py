"""Tests of simple_mappo (the commander + captains MAPPO) on the mock environment - no Unity needed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simple_mappo.env import NavalEnv
from simple_mappo.imitation import behaviour_cloning
from simple_mappo.league import OpponentPool
from simple_mappo.mappo import MAPPO, ppo_objective
from simple_mappo.networks import (CAPTAIN_KEYS, Commander, FleetPolicy, Memory, log_prob_entropy, sample,
                                   to_tensors)
from simple_mappo.rewards import REWARD_WEIGHTS, compute_rewards


@pytest.fixture(scope="module")
def env():
    e = NavalEnv(stages="default", start_stage=4, n_envs=2, mock=True)
    yield e
    e.close()


def small_policy(env, **kw):
    torch.manual_seed(0)
    return FleetPolicy(env.dims, env.heads, d=32, attn_heads=4, layers=1, hidden=32, **kw).eval()


def one_ship(env, obs, e=0, i=0):
    return {k: torch.as_tensor(obs[k][e, i:i + 1]) for k in CAPTAIN_KEYS}


def test_padding_does_not_change_the_decision(env):
    """A masked-out contact slot may hold anything: the other options' scores must not change."""
    policy = small_policy(env)
    obs = env.reset()
    a = one_ship(env, obs)
    a["contact_mask"][0, -1] = 0.0
    b = {k: v.clone() for k, v in a.items()}
    b["contacts"][0, -1] = torch.randn_like(b["contacts"][0, -1]) * 10
    order, h = torch.zeros(1, dtype=torch.long), torch.zeros(1, 32)
    with torch.no_grad():
        la, _ = policy.captain.step(a, order, h)
        lb, _ = policy.captain.step(b, order, h)
    target_off = sum(env.head_sizes[:2])                    # the target head: auto, then one per contact
    last_contact = target_off + env.head_sizes[2] - 1
    keep = [j for j in range(la.shape[1]) if j != last_contact]
    assert torch.allclose(la[0, keep], lb[0, keep], atol=1e-5)


def test_pointer_head_follows_the_contacts(env):
    """Reordering the contacts reorders the target options' scores the same way - nothing else."""
    policy = small_policy(env)
    obs = env.reset()
    a = one_ship(env, obs)
    perm = torch.randperm(a["contacts"].shape[1])
    b = dict(a, contacts=a["contacts"][:, perm], contact_mask=a["contact_mask"][:, perm])
    order, h = torch.zeros(1, dtype=torch.long), torch.zeros(1, 32)
    with torch.no_grad():
        la, _ = policy.captain.step(a, order, h)
        lb, _ = policy.captain.step(b, order, h)
    off = sum(env.head_sizes[:2]) + 1                       # first contact option of the target head
    C = env.head_sizes[2] - 1
    assert torch.allclose(la[0, off:off + C][perm], lb[0, off:off + C], atol=1e-5)
    assert torch.allclose(la[0, :off], lb[0, :off], atol=1e-5)


def test_masked_options_are_never_sampled():
    logits = torch.zeros(500, 6)
    mask = torch.tensor([[1, 0, 1, 0, 1, 1]] * 500, dtype=torch.float32)
    actions, _ = sample(logits, mask, [3, 3])
    assert set(actions[:, 0].tolist()) <= {0, 2}
    assert set(actions[:, 1].tolist()) <= {1, 2}


def test_memory_resets_only_new_battles():
    m = Memory(3, 2, 4, "cpu")
    m.h += 1.0
    m.orders += 3
    m.reset(torch.tensor([0.0, 1.0, 0.0]))
    assert m.h[1].abs().sum() == 0 and m.orders[1].sum() == 0
    assert m.h[0].min() == 1.0 and m.orders[2].min() == 3


def test_training_pass_reproduces_the_rollout(env):
    """Replaying a rollout through Captain.sequence (as training does) gives the same log-probabilities."""
    policy = small_policy(env, commander=True)
    obs = env.reset()
    E, N = obs["alive"].shape
    mem = policy.new_memory(E, N)
    rec = {k: [] for k in CAPTAIN_KEYS + ["action_mask", "starts", "exists"]}
    hs, orders, actions, logps = [], [], [], []
    for _ in range(12):
        out = policy.act(to_tensors(obs, "cpu"), mem)
        for k in rec:
            rec[k].append(torch.as_tensor(obs[k]))
        hs.append(out["h"]), orders.append(out["orders"]), actions.append(out["actions"]), logps.append(out["logp"])
        obs, _, _, _ = env.step(out["actions"].numpy(), out["orders"].numpy())
    T = len(hs)
    seq = {k: torch.stack(v).reshape(T, E * N, *v[0].shape[2:]) for k, v in rec.items() if k not in ("starts", "exists")}
    real = torch.stack(rec["exists"]).reshape(T, E * N) > 0.5         # rows that hold a ship (others are not acted on)
    starts = torch.stack(rec["starts"]).repeat_interleave(N, dim=1)
    with torch.no_grad():
        logits = policy.captain.sequence({k: seq[k] for k in CAPTAIN_KEYS}, torch.stack(orders).reshape(T, E * N),
                                         hs[0].reshape(E * N, -1), starts)
        logp, _ = log_prob_entropy(logits, seq["action_mask"], policy.head_sizes, torch.stack(actions).reshape(T, E * N, -1))
    assert torch.allclose(logp[real], torch.stack(logps).reshape(T, E * N)[real], atol=1e-4)


def test_commander_orders_only_real_circles_and_live_ships(env):
    policy = small_policy(env, commander=True)
    obs = to_tensors(env.reset(), "cpu")
    obs["alive"][0, 1] = 0.0
    mask = Commander.order_mask(obs)
    assert mask[0, 1].tolist() == [1.0] + [0.0] * (mask.shape[-1] - 1)      # sunk: only "free"
    zones = obs["critic_zone_mask"][0]
    assert torch.equal(mask[0, 0, 2:], zones)                                # circle orders: existing circles


def test_dual_clip_bounds_negative_advantages():
    ratio = torch.tensor([10.0, 10.0, 0.5])
    adv = torch.tensor([-1.0, 1.0, -1.0])
    plain = ppo_objective(ratio, adv, 0.2, None)
    dual = ppo_objective(ratio, adv, 0.2, 3.0)
    assert plain[0] == -10.0 and dual[0] == -3.0                          # bounded at 3 x A
    assert dual[1] == plain[1] == 1.2 and dual[2] == plain[2]


def test_pfsp_prefers_opponents_we_lose_to():
    pool = OpponentPool(size=5, seed=1)
    net = torch.nn.Linear(2, 2)
    easy, hard = pool.add(net, 0), pool.add(net, 1)
    for _ in range(30):
        pool.record({"opponent": "past", "opponent_id": easy, "won": True})
        pool.record({"opponent": "past", "opponent_id": hard, "won": False})
    picks = [pool.sample() for _ in range(400)]
    assert picks.count(hard) > 3 * picks.count(easy)
    assert pool.ratings["learner"] < pool.ratings[f"past-{hard}"]


def test_rewards_follow_orders_team_spirit_and_zero_sum():
    N, Z = 3, 2
    before = {"dist_km": np.array([[2.0, 3.0]] * N, np.float32), "inside": np.zeros((N, Z), np.float32),
              "alive": np.ones(N, np.float32)}
    after = {"dist_km": np.array([[1.5, 2.0]] * N, np.float32), "inside": np.zeros((N, Z), np.float32),
             "alive": np.ones(N, np.float32)}
    team = {k: 0.0 for k in ("zones_captured", "zones_lost", "kills", "losses", "win", "loss")}
    zero = np.zeros(N, np.float32)
    ship = {"damage_dealt": np.array([1.0, 0.0, 0.0], np.float32), "damage_taken": zero, "friendly_fire_dealt": zero}
    orders = np.array([0, 1, 3])                    # free (nearest: circle 0), engage, hold circle 1
    r, groups, terms = compute_rewards(before, after, team, ship, None, N, N, REWARD_WEIGHTS, 1.0, orders, 0.0)
    assert terms["approach_circle"].tolist() == pytest.approx([0.5, 0.0, 1.0])
    assert r == pytest.approx(groups.sum(1))
    # full team spirit: every ship gets the fleet's average ship terms
    _, _, shared = compute_rewards(before, after, team, ship, None, N, N, REWARD_WEIGHTS, 1.0, orders, 1.0)
    assert np.allclose(shared["damage_dealt"], shared["damage_dealt"].mean())
    # zero-sum: the enemy's damage dealt is subtracted
    enemy = {"damage_dealt": np.array([2.0, 0.0, 0.0], np.float32), "damage_taken": zero, "friendly_fire_dealt": zero}
    _, _, zs = compute_rewards(before, after, team, ship, enemy, N, 1, REWARD_WEIGHTS, 1.0, orders, 0.0)
    assert zs["zero_sum"][0] == pytest.approx(-REWARD_WEIGHTS["damage_dealt"] * 2.0)


def _mock_expert_dataset(env, battles_steps=200):
    """Labelled decisions from the mock: the right move is compass leg 1 + hint (self features 0-7)."""
    obs = env.reset()
    E, N = obs["alive"].shape
    keys = CAPTAIN_KEYS + ["action_mask"]
    data = {k: [] for k in keys}
    labels, valid = [], []
    for _ in range(battles_steps):
        for k in keys:
            data[k].append(obs[k].reshape(E * N, *obs[k].shape[2:]))
        lab = np.zeros((E * N, len(env.head_sizes)), np.int64)
        lab[:, 0] = 1 + obs["self"][..., :8].reshape(E * N, 8).argmax(-1)
        labels.append(lab)
        valid.append(obs["alive"].reshape(E * N) > 0.5)          # a sunk ship's features are blank
        obs, _, _, _ = env.step(env.sample_random_actions(obs))
    T = len(labels)
    # one sequence per ship slot
    seq_obs = {k: np.stack(v).transpose(1, 0, *range(2, np.stack(v).ndim)).reshape(E * N * T, *v[0].shape[1:])
               for k, v in data.items()}
    lab = np.stack(labels).transpose(1, 0, 2).reshape(E * N * T, -1)
    return {"obs": seq_obs, "labels": lab, "valid": np.stack(valid).T.reshape(-1), "lengths": np.full(E * N, T)}


def test_behaviour_cloning_learns_the_expert(env):
    data = _mock_expert_dataset(env, 120)
    captain, history = behaviour_cloning(data, env.dims, env.heads, d_model=32, attn_layers=1, hidden_size=32,
                                         num_epochs=15, batch_size=16, window=16, learning_rate=3e-3,
                                         held_out=0.25, device="cpu", verbose=0)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["test_accuracy"][-1][0] > 0.6                # the move head copies the expert


def test_learn_save_load_and_predict(env, tmp_path):
    model = MAPPO(env, n_steps=32, chunk_len=8, n_epochs=2, d_model=32, attn_layers=1, hidden_size=32,
                  snapshot_every=1, verbose=0, log_dir=str(tmp_path))
    model.learn(total_timesteps=32 * env.n_envs * 2)
    assert model.n_updates == 2 and (tmp_path / "progress.csv").exists()
    path = model.save(str(tmp_path / "m"))
    again = MAPPO.load(path, env=env)
    obs = env.reset()
    torch.manual_seed(0)
    a1 = model.predict(obs, deterministic=True)
    model._play_memory = None
    a2 = again.predict(obs, deterministic=True)
    assert np.array_equal(a1, a2)


def test_mappo_learns_the_mock_task():
    """End to end: on the mock, the reward pays for the move option 1 + hint; the fleet must learn it."""
    env = NavalEnv(stages=[{"name": "mock", "scenario": "scenarios/stage1_koth_3v3.json", "difficulty": 0}],
                   n_envs=4, mock=True, seed=3)
    try:
        model = MAPPO(env, n_steps=64, chunk_len=16, n_epochs=4, d_model=32, attn_layers=1, hidden_size=64,
                      learning_rate=1e-3, commander=False, verbose=0, seed=3)

        def accuracy():
            obs, hits = env.reset(), []
            for _ in range(20):
                act = model.predict(obs, deterministic=True)
                hits.append(act[..., 0] == 1 + obs["self"][..., :8].argmax(-1))
                obs, _, _, _ = env.step(act)
            return float(np.mean(hits))

        before = accuracy()
        model.learn(total_timesteps=64 * 4 * 40)
        after = accuracy()
        assert after > max(0.5, before + 0.3), (before, after)
    finally:
        env.close()


def test_mirror_battles_train_both_fleets():
    """A mirror battle: our fleet flies both sides, and both sides' experience is training data."""
    stages = [{"name": "mirror", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "league",
               "opponents": {"mirror": 1.0}},
              {"name": "rule", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "rule"}]
    env = NavalEnv(stages=stages, n_envs=2, mock=True, seed=4)
    try:
        assert env.two_sided and env.n_rows == 4
        model = MAPPO(env, n_steps=32, chunk_len=8, n_epochs=1, d_model=32, attn_layers=1, hidden_size=32,
                      commander=True, verbose=0)
        obs = env.reset()
        assert obs["exists"][2:].sum() > 0                       # the enemy fleets are learning rows
        assert model.predict(obs).shape[0] == 4
        model.learn(total_timesteps=32 * 2 * 2)
        b = model.buffer
        assert b["rewards"][:, 2:].any()                         # the enemy rows earned their own rewards
        assert np.array_equal(b["dones"][:, :2], b["dones"][:, 2:])
        done = b["dones"][:, :2] > 0.5
        assert np.all(b["won"][:, :2][done] + b["won"][:, 2:][done] == 1.0)   # one side wins, the other loses
        assert model.num_timesteps == 32 * 2 * 2                 # timesteps count battles, not rows

        env.set_stage(1)                                         # a rule-AI battle: the enemy rows sit idle
        obs = env.reset()
        assert obs["exists"][2:].sum() == 0 and obs["commander_on"][2:].sum() == 0
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------- circles

def test_free_ships_spread_over_the_circles_that_need_them():
    """9 free ships, 5 neutral circles: shared out nearest first, at most 2 per circle (9 / 5 rounded up)."""
    from simple_mappo.rewards import circle_duty
    rng = np.random.default_rng(0)
    dist = rng.uniform(0.5, 6.0, size=(9, 5)).astype(np.float32)
    dist[:, 1] = 0.2                                   # every ship is closest to the centre circle
    facts = {"dist_km": dist, "inside": np.zeros((9, 5), np.float32), "alive": np.ones(9, np.float32),
             "mine": np.zeros(5, np.float32)}
    circle = circle_duty(facts, np.zeros(9, np.int64))
    assert sorted(np.bincount(circle, minlength=5).tolist()) == [1, 2, 2, 2, 2]
    # a circle we own already needs nobody; an order still wins over the sharing out
    facts["mine"][1] = 1.0
    orders = np.zeros(9, np.int64)
    orders[0] = 2 + 1                                  # ship 0: hold circle 1
    circle = circle_duty(facts, orders)
    assert circle[0] == 1 and (circle[1:] != 1).all()
    # every circle ours: each free ship guards its nearest
    facts["mine"][:] = 1.0
    assert (circle_duty(facts, None) == dist.argmin(1)).all()


def test_circles_held_pays_the_fleet():
    from simple_mappo.rewards import REWARD_WEIGHTS, compute_rewards
    N = 2
    facts = {"dist_km": np.full((N, 3), np.inf, np.float32), "inside": np.zeros((N, 3), np.float32),
             "alive": np.ones(N, np.float32), "mine": np.zeros(3, np.float32)}
    team = {k: 0.0 for k in ("zones_captured", "zones_lost", "kills", "losses", "win", "loss")}
    team["circles_held"] = (2 - 1) / 3                  # we hold 2 of 3 circles, they hold 1
    zero = np.zeros(N, np.float32)
    ship = {"damage_dealt": zero, "damage_taken": zero, "friendly_fire_dealt": zero}
    _, _, terms = compute_rewards(facts, facts, team, ship, None, N, N, REWARD_WEIGHTS, 2.0)
    assert terms["circles_held"] == pytest.approx(np.full(N, REWARD_WEIGHTS["circles_held"] * 2.0 / 3))


# ---------------------------------------------------------------------------------------------- PQN

def test_checkpoints_from_fewer_ship_slots_still_fit():
    from simple_mappo.networks import same_heads
    eight = [{"name": "target", "fixed": 1, "pointer": "contacts", "size": 9}]
    nine = [{"name": "target", "fixed": 1, "pointer": "contacts", "size": 10}]
    assert same_heads(eight, nine)
    assert not same_heads(eight, [{"name": "target", "fixed": 2, "pointer": "contacts", "size": 11}])


def test_epsilon_greedy_takes_the_best_legal_option_or_a_random_legal_one():
    from simple_mappo.networks import epsilon_greedy
    scores = torch.tensor([[5.0, 1.0, 0.0, 9.0, 2.0]]).repeat(500, 1)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]]).repeat(500, 1)   # option 3 scores best but is illegal
    greedy = epsilon_greedy(scores, mask, [3, 2], 0.0)
    assert (greedy[:, 0] == 0).all() and (greedy[:, 1] == 1).all()
    explore = epsilon_greedy(scores, mask, [3, 2], 1.0)
    assert set(explore[:, 0].tolist()) == {0, 1} and set(explore[:, 1].tolist()) == {1}


def test_a_rollout_keeps_only_used_ship_slots_and_pads_minibatches():
    from simple_mappo.pqn import RolloutChunks
    rollout = RolloutChunks(chunk_len=4)
    for ships in (1, 3):
        exists = np.zeros((5, 9), np.float32)
        exists[:, :ships] = 1
        rollout.add({"exists": exists, "self": np.ones((5, 9, 7), np.float32), "critic_enemy_mask": exists.copy(),
                     "critic_enemy": np.ones((5, 9, 2), np.float32), "rewards": np.ones((5, 9), np.float32),
                     "dones": np.zeros(5, np.float32), "h0": np.ones((9, 4), np.float32)})
    assert rollout.chunks[0]["self"].shape == (5, 1, 7) and rollout.chunks[0]["self"].dtype == np.float16
    assert len(rollout) == 2 and rollout.decisions == 8
    batch = rollout.stack([1, 0])
    assert batch["self"].shape == (5, 2, 3, 7) and batch["h0"].shape == (2, 3, 4)
    assert batch["exists"][:, 1, 1:].sum() == 0                  # the 1-ship chunk is padded with empty slots


def test_pqn_learn_save_load_and_predict(env, tmp_path):
    from simple_mappo.pqn import FleetPQN
    model = FleetPQN(env, d_model=32, attn_layers=1, hidden_size=32, chunk_len=8, num_steps=16, mixer="none",
                     verbose=0, log_dir=str(tmp_path))
    model.learn(total_timesteps=16 * env.n_envs * 3, log_interval=1)
    assert model.n_updates == 3 and model.n_gradient_steps > 0 and (tmp_path / "progress.csv").exists()
    path = model.save(str(tmp_path / "q"))
    again = FleetPQN.load(path, env=env)
    obs = env.reset()
    a1, a2 = model.predict(obs, deterministic=True), again.predict(obs, deterministic=True)
    assert np.array_equal(a1, a2)
    mask = obs["action_mask"]
    off = 0
    for h, size in enumerate(env.head_sizes):        # greedy actions are legal ones
        chosen = np.take_along_axis(mask[..., off:off + size], a1[..., h:h + 1], -1)
        assert (chosen[obs["exists"] > 0.5] > 0.5).all()
        off += size
    with pytest.raises(ValueError):
        FleetPQN(env, mixer="qmix")


# ---------------------------------------------------------------------------------------------- the league

def test_champion_is_replaced_only_when_beaten():
    """AlphaGo Zero's gating: the learner takes the champion's place after winning 55 % of 10 battles."""
    pool = OpponentPool(champion_games=10, champion_win_rate=0.55)
    net = torch.nn.Linear(2, 2)
    first = pool.crown(net, 0)
    for _ in range(10):
        pool.record({"opponent": "champion", "opponent_id": first, "won": False})
    assert not pool.champion_beaten()
    for _ in range(6):
        pool.record({"opponent": "champion", "opponent_id": first, "won": True})
    assert pool.champion_beaten()                    # 6 of the last 10
    second = pool.crown(net, 1)
    assert pool.champion == second and pool.sample("champion") == second
    assert pool._get(first)["kind"] == "past" and not pool.champion_beaten()


def test_cross_play_puts_the_other_methods_fleet_in_the_league(env):
    from simple_mappo.pqn import FleetPQN
    mappo = MAPPO(env, d_model=32, attn_layers=1, hidden_size=32, verbose=0)
    pqn = FleetPQN(env, d_model=32, attn_layers=1, hidden_size=32, verbose=0)
    assert mappo.set_rival(pqn) and pqn.set_rival(mappo)
    rival = mappo.league.sample("rival")
    assert mappo.league.greedy(rival)                # a PQN fleet chooses by its Q-values
    assert not pqn.league.greedy(pqn.league.sample("rival"))
    assert not mappo.set_rival(FleetPQN(env, d_model=16, attn_layers=1, hidden_size=32, verbose=0))  # other network
    obs = env.reset()
    enemy = {k: v[:1] for k, v in obs.items()}
    assert mappo._opponent_act(enemy, [0], [rival]).shape == (1, env.n_agents, len(env.head_sizes))


def test_a_missing_champion_or_rival_plays_a_mirror_battle():
    stages = [{"name": "x", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "league",
               "opponents": {"rival": 1.0, "mirror": 0.0}}]
    env = NavalEnv(stages=stages, n_envs=1, mock=True)
    try:
        env.set_opponent_policy(lambda *a: None, lambda kind: None)        # nothing in the league yet
        assert env._pick_opponent(env.stages[0]) == ("mirror", -1)
        env.set_opponent_policy(lambda *a: None, lambda kind: 7 if kind == "rival" else None)
        assert env._pick_opponent(env.stages[0]) == ("rival", 7)
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------- logs, versus

def test_a_log_with_an_older_layout_is_read_and_rewritten(tmp_path):
    """A battle log started before a reward term existed, with rows of the new layout appended (what a
    session that pulled an old log from the team folder produced) - read by name, then migrated."""
    import pandas as pd
    from simple_mappo.rewards import REWARD_WEIGHTS
    from simple_mappo.utils import REASON_KEYS, load_results, migrate_monitor
    base = ["r", "l", "t", "won", "passed", "stage", "opponent", "battle_time", "survived"] + REASON_KEYS
    old_terms = [k for k in REWARD_WEIGHTS if k != "circles_held"]
    old = base + ["term_" + k for k in old_terms] + ["reason"]
    new = base + ["term_" + k for k in REWARD_WEIGHTS] + ["reason"]
    path = tmp_path / "monitor.csv"
    with open(path, "w") as f:
        f.write("#{}\n" + ",".join(old) + "\n")
        f.write(",".join(["1", "300", "1", "1", "1", "7", "rule", "300", "1"] + ["5"] * len(REASON_KEYS)
                         + ["0.1"] * len(old_terms) + ['"old, row"']) + "\n")
        f.write(",".join(["2", "400", "2", "0", "0", "7", "rival", "400", "0"] + ["6"] * len(REASON_KEYS)
                         + ["0.2"] * len(REWARD_WEIGHTS) + ['"new row"']) + "\n")
    df = load_results(str(tmp_path))
    assert df.opponent.tolist() == ["rule", "rival"] and np.isnan(df.term_circles_held[0])
    assert df.term_circles_held[1] == pytest.approx(0.2) and df.term_win.tolist() == pytest.approx([0.1, 0.2])
    migrate_monitor(str(path), new)
    assert list(pd.read_csv(path, skiprows=1).columns) == new


def test_head_to_head_puts_the_other_fleet_in_the_enemy_seat(tmp_path):
    from simple_mappo.pqn import FleetPQN
    from simple_mappo.utils import Monitor, load_results
    from simple_mappo.versus import both_sides, versus_stages
    battles = NavalEnv(stages=versus_stages(4), n_envs=2, mock=True)
    try:
        env_m, env_p = Monitor(battles, str(tmp_path / "m")), Monitor(battles, str(tmp_path / "p"))
        mappo = MAPPO(env_m, d_model=32, attn_layers=1, hidden_size=32, verbose=0)
        pqn = FleetPQN(env_p, d_model=32, attn_layers=1, hidden_size=32, verbose=0)
        r = both_sides(mappo, pqn, env_m, env_p, n_battles=2)
        assert r["battles"] == 4 and 0.0 <= r["mappo_win_rate"] <= 1.0
        assert set(load_results(str(tmp_path / "m")).opponent) == {"rival"}
        assert mappo.league.greedy(mappo.league.sample("rival"))           # PQN plays by its Q-values
    finally:
        battles.close()


@pytest.mark.parametrize("met", [True, False])
def test_a_goal_stage_pays_the_result_for_the_goal(met):
    # Gunnery and Capture: the passive enemy never scores, so the game's tiebreak would call every
    # battle won - the win / loss reward must follow the stage's goal instead (env.py)
    stages = [{"name": "goal", "scenario": "scenarios/stage0_gunnery.json", "opponent": "passive",
               "pass_if": "sunk", "within": 150}]
    env = NavalEnv(stages=stages, n_envs=1, mock=True)
    env._passed = lambda e, o, captures=None: met
    obs, info = env.reset(), None
    for _ in range(500):
        obs, _, dones, infos = env.step(env.sample_random_actions(obs))
        if dones[0]:
            info = infos[0]
            break
    env.close()
    terms = info["reward_terms"]
    assert (terms["win"] > 0 and terms["loss"] == 0) if met else (terms["win"] == 0 and terms["loss"] < 0)


def test_a_missed_deadline_ends_the_battle():
    stages = [{"name": "goal", "scenario": "scenarios/stage0_gunnery.json", "opponent": "passive",
               "pass_if": "sunk", "within": 20}]
    env = NavalEnv(stages=stages, n_envs=1, mock=True)            # a mock battle lasts 60 s
    obs, info = env.reset(), None
    for _ in range(100):
        obs, _, dones, infos = env.step(env.sample_random_actions(obs))
        if dones[0]:
            info = infos[0]
            break
    env.close()
    assert info["battle_time"] == 20 and not info["passed"] and not info["won"] and "missed" in info["reason"]
    assert info["reward_terms"]["loss"] < 0


def test_pqn_mirror_battles_train_both_fleets():
    """Self-play for PQN too: in a mirror battle both fleets' chunks go into the rollout."""
    from simple_mappo.pqn import FleetPQN
    stages = [{"name": "mirror", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "league",
               "opponents": {"mirror": 1.0}},
              {"name": "rule", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "rule"}]
    env = NavalEnv(stages=stages, n_envs=2, mock=True, seed=4)
    try:
        pqn = FleetPQN(env, chunk_len=8, num_steps=8, d_model=32, attn_layers=1, hidden_size=32, verbose=0)
        pqn.learn(total_timesteps=8 * 2)                         # one rollout: a chunk of 8 decisions per battle
        assert len(pqn.rollout) == 4                             # 2 battles x both sides
        env.set_stage(1)                                         # against the rule AI: our side only
        pqn.learn(total_timesteps=pqn.num_timesteps + 8 * 2)
        assert len(pqn.rollout) == 2
    finally:
        env.close()


def test_a_stage_that_is_not_passed_moves_on_after_its_cap(tmp_path):
    from simple_mappo.curriculum import CurriculumCallback
    from simple_mappo.utils import Monitor, compare_stages
    stages = [{"name": "0 hard", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "rule",
               "promote_win_rate": 1.1, "window": 1000, "max_timesteps": 200},       # a bar no one can reach
              {"name": "1 next", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "rule"}]
    battles = NavalEnv(stages=stages, n_envs=2, mock=True)
    env = Monitor(battles, str(tmp_path))
    try:
        model = MAPPO(env, n_steps=32, chunk_len=8, n_epochs=1, d_model=32, attn_layers=1, hidden_size=32,
                      commander=False, verbose=0)
        first = CurriculumCallback(stop_on_promotion=True, verbose=0)
        model.learn(total_timesteps=128, callback=[first])                  # one session: not there yet
        assert env.stage == 0 and not first.moved_on_unpassed
        second = CurriculumCallback(stop_on_promotion=True, verbose=0)      # the next session continues the count
        model.learn(total_timesteps=model.num_timesteps + 1000, callback=[second])
        assert env.stage == 1 and second.moved_on_unpassed and model.num_timesteps < 128 + 1000
        table = compare_stages({"MAPPO": str(tmp_path)}, stages)
        assert table.loc["0 hard", "MAPPO: passed"] == "no - moved on"

        # FORCE_STAGE: back to the capped stage - it trains again, with its cap counted afresh
        env.set_stage(0)
        again = CurriculumCallback(stop_on_promotion=True, fresh=True, verbose=0)
        model.learn(total_timesteps=model.num_timesteps + 128, callback=[again])
        assert env.stage == 0 and not again.moved_on_unpassed                # without fresh: moved on at once
        assert compare_stages({"MAPPO": str(tmp_path)}, stages).loc["0 hard", "MAPPO: passed"] == "training again"
        # the next session continues the retrain's count (from the "again" mark), not the stage's whole log
        later = CurriculumCallback(stop_on_promotion=True, verbose=0)
        start = model.num_timesteps
        model.learn(total_timesteps=start + 1000, callback=[later])
        assert env.stage == 1 and 40 <= model.num_timesteps - start < 200     # the rest of the new 200-step cap
    finally:
        battles.close()


def test_the_final_stage_ends_at_its_cap(tmp_path):
    from simple_mappo.curriculum import CurriculumCallback
    from simple_mappo.utils import Monitor, compare_stages
    stages = [{"name": "0 first", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "rule"},
              {"name": "1 last", "scenario": "scenarios/stage1_koth_3v3.json", "opponent": "rule",
               "stop_win_rate": 1.1, "stop_window": 1000, "eval_every": 10**9, "max_timesteps": 200}]
    battles = NavalEnv(stages=stages, start_stage=1, n_envs=2, mock=True)
    env = Monitor(battles, str(tmp_path))
    try:
        model = MAPPO(env, n_steps=32, chunk_len=8, n_epochs=1, d_model=32, attn_layers=1, hidden_size=32,
                      commander=False, verbose=0)
        cb = CurriculumCallback(stop_on_promotion=True, verbose=0)
        model.learn(total_timesteps=5000, callback=[cb])
        assert cb.finished and cb.capped_out and model.num_timesteps < 300      # over at the cap, not at 5000
        assert compare_stages({"MAPPO": str(tmp_path)}, stages).loc["1 last", "MAPPO: passed"] == "no - cap reached"
    finally:
        battles.close()


def test_running_aground_and_ramming_cost_points():
    N, Z = 2, 1
    facts = {"dist_km": np.full((N, Z), np.inf, np.float32), "inside": np.zeros((N, Z), np.float32),
             "alive": np.ones(N, np.float32)}
    team = {k: 0.0 for k in ("zones_captured", "zones_lost", "kills", "losses", "win", "loss")}
    zero = np.zeros(N, np.float32)
    ship = {"damage_dealt": zero, "damage_taken": zero, "friendly_fire_dealt": zero,
            "aground": np.array([1.0, 0.0], np.float32), "collisions": np.array([0.0, 2.0], np.float32)}
    r, _, terms = compute_rewards(facts, facts, team, ship, None, N, 1, REWARD_WEIGHTS, 1.0)
    assert terms["aground"].tolist() == pytest.approx([REWARD_WEIGHTS["aground"], 0.0])
    assert terms["collision"].tolist() == pytest.approx([0.0, 2 * REWARD_WEIGHTS["collision"]])
    assert REWARD_WEIGHTS["aground"] < 0 and REWARD_WEIGHTS["collision"] < 0 and (r < 0).all()
    # a player built before these terms existed reports neither: nothing is charged
    old = {k: v for k, v in ship.items() if k not in ("aground", "collisions")}
    _, _, terms = compute_rewards(facts, facts, team, old, None, N, 1, REWARD_WEIGHTS, 1.0)
    assert not terms["aground"].any() and not terms["collision"].any()


def test_pqn_softmax_exploration_samples_and_trains(env, tmp_path):
    from simple_mappo.pqn import FleetPQN
    greedy = FleetPQN(env, d_model=32, attn_layers=1, hidden_size=32, exploration="epsilon", verbose=0)
    soft = FleetPQN(env, d_model=32, attn_layers=1, hidden_size=32, exploration="softmax", temperature=10.0,
                    chunk_len=8, num_steps=16, verbose=0)
    soft.policy.load_state_dict(greedy.policy.state_dict())
    assert greedy.act_temperature is None and soft.act_temperature == 10.0
    obs = env.reset()
    picks_g = {tuple(greedy.predict(obs, deterministic=True)[0, 0]) for _ in range(20)}
    picks_s = {tuple(soft.predict(obs)[0, 0]) for _ in range(20)}
    assert len(picks_g) == 1 and len(picks_s) > 1             # greedy repeats its best option, softmax samples
    assert {tuple(soft.predict(obs, deterministic=True)[0, 0]) for _ in range(5)} == picks_g   # deterministic = greedy
    soft.learn(total_timesteps=64)                              # plays (and trains) with it, saves and reloads it
    path = soft.save(str(tmp_path / "soft"))
    assert FleetPQN.load(path, env=env).hp["exploration"] == "softmax"
    with pytest.raises(ValueError):
        FleetPQN(env, exploration="boltzmann")


def test_pqn_lambda_targets(env):
    from simple_mappo.pqn import FleetPQN
    m = FleetPQN(env, d_model=32, attn_layers=1, hidden_size=32, gamma=0.5, num_steps=32, verbose=0)
    r = torch.tensor([[1.0], [2.0], [4.0], [0.0]])           # [T1 = 4, B = 1]: 3 decisions + the bootstrap state
    q = torch.tensor([[10.0], [20.0], [30.0], [40.0]])       # max_a Q(s_t)
    no_end = torch.zeros(4, 1)
    m.hp["lam"] = 1.0                                        # the discounted return, bootstrapped at the end
    assert m._lambda_targets(r, no_end, q)[:, 0].tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 4 + 0.125 * 40,
                                                                            2 + 0.5 * 4 + 0.25 * 40, 4 + 0.5 * 40])
    m.hp["lam"] = 0.0                                        # one-step Q-learning targets
    assert m._lambda_targets(r, no_end, q)[:, 0].tolist() == pytest.approx([1 + 0.5 * 20, 2 + 0.5 * 30, 4 + 0.5 * 40])
    end = torch.tensor([[0.0], [1.0], [0.0], [0.0]])         # the battle ends after decision 1
    m.hp["lam"] = 0.5
    y = m._lambda_targets(r, end, q)[:, 0].tolist()
    assert y[1] == pytest.approx(2.0)                        # nothing after the end
    assert y[0] == pytest.approx(1 + 0.5 * (0.5 * 2.0 + 0.5 * 20))
    assert y[2] == pytest.approx(4 + 0.5 * 40)               # a new battle: its own bootstrap


def test_pqn_learns_saves_and_loads(env, tmp_path):
    from simple_mappo.pqn import FleetPQN
    m = FleetPQN(env, d_model=32, attn_layers=1, hidden_size=32, num_steps=64, verbose=0)
    m.learn(total_timesteps=64 * env.n_envs * 2)
    assert m.n_updates == 2 and m.n_gradient_steps == 2 * 4 * 4        # epochs x minibatches per update
    path = m.save(str(tmp_path / "pqn"))
    loaded = FleetPQN.load(path, env=env)
    assert loaded.hp["lam"] == 0.85 and loaded.num_timesteps == m.num_timesteps
    assert torch.load(path, weights_only=False)["method"] == "fleet-pqn"
    mappo = MAPPO(env, d_model=32, attn_layers=1, hidden_size=32, verbose=0).save(str(tmp_path / "mappo"))
    with pytest.raises(ValueError):
        FleetPQN.load(mappo, env=env)                        # a MAPPO checkpoint is not a PQN one


def test_a_kernel_whose_folder_was_renamed_says_restart(tmp_path, monkeypatch):
    from simple_mappo import env as env_module
    env_module.check_code_location()                              # the tests run from Training/: all is well
    monkeypatch.setattr(env_module, "TRAINING_DIR", str(tmp_path / "A3_DDQN" / "Training"))
    with pytest.raises(RuntimeError, match="no longer exists"):
        env_module.find_unity_binary()                            # renamed after the kernel started
    old, new = tmp_path / "A3_DDQN" / "Training", tmp_path / "A3_PQN" / "Training"
    for folder in (old, new):
        (folder / "simple_mappo").mkdir(parents=True)
    monkeypatch.chdir(new)
    with pytest.raises(RuntimeError, match="Restart the kernel"):
        env_module.check_code_location()                          # the code still comes from the old copy
