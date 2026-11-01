import helpers
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from delivery_lab import planner as P  # noqa: E402
from delivery_lab import sim as S  # noqa: E402
from delivery_lab.learn import eta, jev, prep  # noqa: E402
from delivery_lab.learn.fit import Data, fit  # noqa: E402
from delivery_lab.learn.nn import Net  # noqa: E402
from delivery_lab.sim import DROPOFF, PICKUP  # noqa: E402

F = jev.N_FEATURES


def linear_model(weights: dict[str, float], bias=0.0):
    w = np.zeros((F, 1), dtype=np.float32)
    for name, value in weights.items():
        w[jev.FEATURES.index(name), 0] = value
    state = {"W0": w, "b0": np.array([bias], dtype=np.float32)}
    return jev.JevModel("linear", state, np.zeros(F), np.ones(F))


def test_decision_contract_probabilities_confidence_and_exact_contributions():
    model = linear_model({"travel_min": -1.0, "slack_min": -0.1})
    feats = np.zeros((3, F), dtype=np.float32)
    feats[:, jev.FEATURES.index("travel_min")] = [2.0, 5.0, 9.0]
    feats[:, jev.FEATURES.index("slack_min")] = [30.0, 10.0, 0.0]
    cands = [(PICKUP, 1), (DROPOFF, 2), (PICKUP, 3)]
    d = model.decide(feats, cands)
    scores = -feats[:, 0] - 0.1 * feats[:, 3]
    want = np.exp(scores - scores.max())
    want /= want.sum()
    assert np.allclose(d.probabilities, want) and d.probabilities.sum() == pytest.approx(1.0)
    assert d.chosen == int(want.argmax()) and d.stop == cands[d.chosen]
    assert d.confidence == pytest.approx(want.max()) and d.margin == pytest.approx(np.sort(want)[-1] - np.sort(want)[-2])
    # a soma das contribuições de uma candidata é a pontuação dela menos a média das candidatas
    assert np.allclose(d.contributions.sum(axis=1), scores - scores.mean(), atol=1e-5)
    # sem peso, o fator não explica nada
    assert np.allclose(d.contributions[:, jev.FEATURES.index("age_min")], 0.0)


def test_mlp_models_decide_but_do_not_pretend_to_explain():
    net = Net(F, (4,), 1, seed=0)
    model = jev.JevModel("mlp", net.state(), np.zeros(F), np.ones(F))
    d = model.decide(np.random.default_rng(0).normal(size=(3, F)).astype(np.float32), [(PICKUP, i) for i in range(3)])
    assert d.contributions is None and d.probabilities.sum() == pytest.approx(1.0)


def test_batch_probabilities_match_one_by_one_and_ignore_padding():
    rng = np.random.default_rng(1)
    model = jev.JevModel("mlp", Net(F, (5,), 1, seed=1).state(), rng.normal(size=F), rng.uniform(0.5, 2, F))
    n_cand = rng.integers(2, jev.MAX_CANDIDATES + 1, 40)
    x = rng.normal(size=(40, jev.MAX_CANDIDATES, F)).astype(np.float32)
    mask = np.arange(jev.MAX_CANDIDATES)[None, :] < n_cand[:, None]
    batch = model.probabilities_batch(x, mask)
    for i in range(40):
        one = model.decide(x[i, : n_cand[i]], [(PICKUP, k) for k in range(n_cand[i])]).probabilities
        assert np.allclose(batch[i, : n_cand[i]], one, atol=1e-5)
        assert (batch[i, n_cand[i]:] == 0).all()


def test_set_scorer_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    net = Net(F, (4,), 1, seed=3, dtype=torch.float64)
    scorer = jev.SetScorer(net, jev.MAX_CANDIDATES, F)
    n = 6
    n_cand = rng.integers(2, jev.MAX_CANDIDATES + 1, n)
    mask = (np.arange(jev.MAX_CANDIDATES)[None, :] < n_cand[:, None]).astype(float)
    x = torch.tensor(np.concatenate([rng.normal(size=(n, jev.MAX_CANDIDATES * F)), mask], axis=1))
    y = torch.tensor(np.array([[rng.integers(0, k)] for k in n_cand], dtype=float))
    loss = jev.set_nll(scorer.forward(x), y)
    for p in net.parameters():
        p.grad = None
    loss.backward()
    eps, checked = 1e-6, 0
    for p in net.parameters():
        flat, grad = p.detach().view(-1), p.grad.view(-1)
        for i in rng.choice(len(flat), size=min(4, len(flat)), replace=False):
            old = float(flat[i])
            flat[i] = old + eps
            up = float(jev.set_nll(scorer.forward(x), y).detach())
            flat[i] = old - eps
            down = float(jev.set_nll(scorer.forward(x), y).detach())
            flat[i] = old
            assert float(grad[i]) == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-8)
            checked += 1
    assert checked >= 8


def test_training_recovers_a_known_rule():
    rng = np.random.default_rng(4)
    n = 6000
    n_cand = rng.integers(2, jev.MAX_CANDIDATES + 1, n)
    mask = np.arange(jev.MAX_CANDIDATES)[None, :] < n_cand[:, None]
    x = rng.normal(size=(n, jev.MAX_CANDIDATES, F)).astype(np.float32) * mask[..., None]
    travel = np.where(mask, x[:, :, 0], np.inf)
    y = travel.argmin(axis=1)  # a regra verdadeira: ir à mais perto
    make = lambda sl: jev.JevSet(x[sl], mask[sl], y[sl], np.full(len(y[sl]), -1))  # noqa: E731
    model = jev.train_jev(make(slice(0, 4500)), make(slice(4500, 5200)), kind="linear", epochs=30, batch=100, log=lambda s: None)
    m = jev.evaluate_jev(model, make(slice(5200, n)), latency_sample=200)
    assert m["top1"] > 0.9 and m["top1"] > 1.4 * m["chance"]
    assert int(np.argmax(np.abs(model.weights))) == 0 and model.weights[0] < 0  # aprendeu "menor tempo de viagem"
    assert m["nll"] < 0.7  # um sorteio uniforme entre as candidatas daria ~1,3
    assert 0.0 <= m["ece"] <= 1.0 and m["decide_us"] > 0
    assert m["baselines_top1"]["parada_mais_proxima"] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def tiny_models(grid_city, grid_world, grid_paths):
    ptrain = prep.build_prep_set(grid_city, grid_world, range(21, 36))
    pval = prep.build_prep_set(grid_city, grid_world, range(36, 40))
    prep_model, _ = prep.train_prep(grid_world, ptrain, pval, epochs=10, log=lambda s: None)
    fleet = S.generate_fleet(grid_world, grid_city, 60, seed=7)
    ltrain = eta.collect_legs(grid_city, grid_world, grid_paths, fleet, range(21, 27))
    lval = eta.collect_legs(grid_city, grid_world, grid_paths, fleet, range(27, 29))
    eta_model, _ = eta.train_eta(grid_city, ltrain, lval, epochs=8, log=lambda s: None)
    train = jev.build_jev_set(grid_city, grid_world, grid_paths, fleet, range(21, 27), prep_model, eta_model)
    val = jev.build_jev_set(grid_city, grid_world, grid_paths, fleet, range(27, 29), prep_model, eta_model)
    test = jev.build_jev_set(grid_city, grid_world, grid_paths, fleet, range(101, 103), prep_model, eta_model, planner_view=True)
    model = jev.train_jev(train, val, kind="linear", epochs=12, log=lambda s: None)
    return prep_model, eta_model, model, fleet, train, test


def test_collected_decisions_are_well_formed(tiny_models):
    *_, train, test = tiny_models
    assert len(train) > 500 and len(test) > 200
    assert (train.mask.sum(axis=1) >= 2).all()
    assert (train.y >= 0).all() and (train.y < train.mask.sum(axis=1)).all()
    assert np.isfinite(train.x).all()
    assert (test.planner_view >= 0).all() and (test.planner_view < test.mask.sum(axis=1)).all()


def test_features_only_use_what_the_platform_sees(grid_city, grid_world, grid_paths, tiny_models):
    prep_model, eta_model, *_ = tiny_models
    orders = helpers.tiny_orders(grid_city, [0, 1], [300, 900], [43000.0, 43050.0])

    def features(shift):
        view = P.Planner(prep_model=prep_model, eta_model=eta_model)
        sim = helpers.make_sim(grid_city, grid_world, grid_paths, helpers.always_on_fleet([10]), orders, view)
        sim.t_ready = sim.t_ready + shift  # o preparo verdadeiro muda; a visão da plataforma não pode notar
        view.bind(sim)
        sim.now = 43100.0
        sim.shift_ok[:] = True
        return jev.candidate_features(view, sim, 0, [(PICKUP, 0), (PICKUP, 1)])

    a, b = features(0.0), features(5000.0)
    assert np.array_equal(a, b)
    assert a.shape == (2, F) and np.isfinite(a).all()
    assert (a[:, jev.FEATURES.index("is_pickup")] == 1).all()
    assert (a[:, jev.FEATURES.index("bag_after")] == 1).all()  # nada na mochila + a coleta


def test_jev_policy_runs_a_full_day_and_records_explanations(grid_city, grid_world, grid_paths, tiny_models):
    prep_model, eta_model, model, fleet, *_ = tiny_models
    policy = jev.JevPlanner(prep_model, eta_model, model, record=True)
    result = S.run_day(grid_city, grid_world, grid_paths, fleet, policy, day=105)
    ok = ~result.rejected
    assert not np.isnan(result.t_delivered[ok]).any()
    assert (result.t_picked[ok] >= result.t_ready[ok] + result.config.handover_s - 1e-6).all()
    assert (result.t_picked[ok] < result.t_delivered[ok]).all()
    assert policy.name == "Jev" and len(policy.decisions) > 100 and len(policy.decide_s) >= len(policy.decisions)
    d = policy.decisions[0]["decision"]
    assert d.contributions is not None and d.probabilities.sum() == pytest.approx(1.0)
    assert all(len(x["decision"].candidates) >= 2 for x in policy.decisions)


def test_evaluation_matches_a_hand_count(tiny_models):
    *_, model, _fleet, _train, test = tiny_models
    m = jev.evaluate_jev(model, test, latency_sample=100)
    p = model.probabilities_batch(test.x, test.mask)
    assert m["top1"] == pytest.approx(float((p.argmax(axis=1) == test.y).mean()))
    assert m["decisions"] == len(test) and sum(b["n"] for b in m["reliability"]) == len(test)
    assert m["top1"] > m["chance"] and m["baselines_top1"]["planejador_reotimizando"] > m["chance"]


def test_model_roundtrip(tmp_path, tiny_models):
    *_, model, _fleet, _train, test = tiny_models
    model.save(tmp_path / "j.npz")
    again = jev.JevModel.load(tmp_path / "j.npz")
    assert again.kind == "linear" and np.allclose(again.probabilities_batch(test.x[:50], test.mask[:50]), model.probabilities_batch(test.x[:50], test.mask[:50]))


def test_replay_carries_the_decisions_and_their_explanations(grid_city, grid_world, grid_paths, tiny_models):
    from delivery_lab.replay import render_replay

    prep_model, eta_model, model, fleet, *_ = tiny_models
    policy = jev.JevPlanner(prep_model, eta_model, model, record=True)
    result = S.run_day(grid_city, grid_world, grid_paths, fleet, policy, day=106, record=True)
    payload = render_replay(result, grid_city, grid_world, policy.name, 106, policy.decisions)
    d = payload["decisions"]
    n = len(d["leg"])
    assert n > 100 and len(d["chosen"]) == n and len(d["ptr"]) == n + 1 and d["ptr"][-1] == len(d["kind"]) == len(d["prob"])
    assert len(d["contrib"]) == len(d["prob"]) and all(len(row) == F for row in d["contrib"])
    assert d["factors"] == list(jev.FEATURES) and len(d["labels"]) == F
    # cada decisão aponta para o trecho que nasceu dela: a parada escolhida é o destino do trecho
    legs = payload["legs"]
    for i in range(0, n, max(1, n // 25)):
        k = d["ptr"][i] + d["chosen"][i]
        assert legs["kind"][d["leg"][i]] == d["kind"][k] and legs["order"][d["leg"][i]] == d["order"][k]
        assert sum(d["prob"][d["ptr"][i] : d["ptr"][i + 1]]) == pytest.approx(1.0, abs=0.01)
    import json

    json.dumps(payload)
