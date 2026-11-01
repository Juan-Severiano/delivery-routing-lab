import helpers
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from delivery_lab import planner as P  # noqa: E402
from delivery_lab import sim as S  # noqa: E402
from delivery_lab.learn import eta, prep  # noqa: E402
from delivery_lab.traffic import rain_of_day  # noqa: E402
from delivery_lab.world import sample_day  # noqa: E402


@pytest.fixture(scope="module")
def prep_sets(grid_city, grid_world):
    train = prep.build_prep_set(grid_city, grid_world, range(21, 41))
    val = prep.build_prep_set(grid_city, grid_world, range(41, 46))
    test = prep.build_prep_set(grid_city, grid_world, range(101, 106))
    return train, val, test


@pytest.fixture(scope="module")
def prep_model(grid_world, prep_sets):
    model, _ = prep.train_prep(grid_world, prep_sets[0], prep_sets[1], epochs=20, seed=0, log=lambda s: None)
    return model


@pytest.fixture(scope="module")
def eta_model(grid_city, grid_world, grid_paths):
    fleet = S.generate_fleet(grid_world, grid_city, 70, seed=7)
    train = eta.collect_legs(grid_city, grid_world, grid_paths, fleet, range(21, 29))
    val = eta.collect_legs(grid_city, grid_world, grid_paths, fleet, range(29, 31))
    model, _ = eta.train_eta(grid_city, train, val, epochs=12, seed=0, log=lambda s: None)
    return model, train


def test_prep_set_is_exactly_the_hidden_prep_the_simulator_draws(grid_city, grid_world, grid_paths):
    day = 5
    data = prep.build_prep_set(grid_city, grid_world, range(day, day + 1))
    orders = sample_day(grid_world, grid_city, day)
    sim = helpers.make_sim(grid_city, grid_world, grid_paths, helpers.always_on_fleet([0]), orders, S.NearestPolicy(), day)
    assert np.allclose(data.prep_s, sim.t_ready - sim.t_created)
    assert (data.rain == rain_of_day(grid_world.seed, day)).all()


def test_prep_model_beats_the_rule_of_thumb_and_is_calibrated(grid_world, prep_sets, prep_model):
    train, _, test = prep_sets
    m = prep.evaluate_prep(prep_model, test, train)
    mae = m["mae_min"]
    assert mae["modelo_mediana"] < 0.6 * mae["regra_de_bolso_700s"]
    assert mae["modelo_mediana"] < mae["media_global"]
    assert mae["modelo_mediana"] <= 1.15 * mae["verdade_do_simulador"]  # perto do piso de ruído
    assert m["calibration"]["mean_abs_error"] < 0.04
    assert m["sigma"]["modelo_medio"] == pytest.approx(m["sigma"]["verdade_media"], rel=0.15)


def test_prep_quantiles_are_ordered_and_saved_models_predict_the_same(tmp_path, grid_world, prep_sets, prep_model):
    _, _, test = prep_sets
    r, t, rain = test.restaurant[:200], test.t_s[:200], test.rain[:200]
    q10, q50, q90 = (prep_model.quantile_s(r, t, rain, q) for q in (0.1, 0.5, 0.9))
    assert (q10 < q50).all() and (q50 < q90).all()
    prep_model.save(tmp_path / "p.npz")
    again = prep.PrepModel.load(tmp_path / "p.npz")
    assert np.allclose(again.quantile_s(r, t, rain, 0.5), q50)
    assert again.meta["train_orders"] == prep_model.meta["train_orders"]


def test_prep_model_learns_that_the_peak_is_slower_than_the_valley(grid_world, prep_model):
    both = np.nonzero(grid_world.is_open(19.8) & grid_world.is_open(16.0))[0]  # restaurantes com dados nas duas horas
    assert len(both) >= 5
    none = np.zeros(len(both), bool)
    peak = prep_model.quantile_s(both, np.full(len(both), 19.8 * 3600), none).mean()
    valley = prep_model.quantile_s(both, np.full(len(both), 16.0 * 3600), none).mean()
    truth = (grid_world.prep_minutes(19.8)[both] / grid_world.prep_minutes(16.0)[both]).mean()
    assert peak / valley == pytest.approx(truth, rel=0.12)
    assert peak / valley > 1.05


def test_eta_model_is_closer_to_the_observed_leg_time_than_the_free_flow_time(grid_city, grid_world, grid_paths, eta_model):
    model, train = eta_model
    fleet = S.generate_fleet(grid_world, grid_city, 70, seed=7)
    test = eta.collect_legs(grid_city, grid_world, grid_paths, fleet, range(101, 104))
    m = eta.evaluate_eta(grid_city, model, test, train)
    assert m["modelo"]["todos"]["mae_s"] < m["tempo_livre"]["todos"]["mae_s"]
    assert abs(m["modelo"]["todos"]["bias_pct"]) < abs(m["tempo_livre"]["todos"]["bias_pct"])
    assert m["verdade_sem_ruido_de_viagem"]["todos"]["mae_s"] <= m["modelo"]["todos"]["mae_s"] + 1e-6


def test_eta_features_are_finite_and_have_the_expected_width(grid_city):
    u, v = np.array([0, 10]), np.array([300, 900])
    x = eta.eta_features(grid_city.node_xy[u], grid_city.node_xy[v], np.array([120.0, 400.0]), np.array([43000.0, 70000.0]),
                         np.array([False, True]), np.zeros(2))
    assert x.shape == (2, eta.N_FEATURES) and np.isfinite(x).all()


def test_b1_plus_uses_the_model_and_never_the_hidden_prep(grid_city, grid_world, grid_paths, prep_model, eta_model):
    orders = helpers.tiny_orders(grid_city, [0, 1], [200, 900], [43000.0, 43100.0])
    fleet = helpers.always_on_fleet([10, 400])

    def ready_after(shift):
        pl = P.Planner(prep_model=prep_model, eta_model=eta_model[0])
        sim = helpers.make_sim(grid_city, grid_world, grid_paths, fleet, orders, pl)
        sim.t_ready = sim.t_ready + shift  # o preparo verdadeiro muda; o B1+ não pode notar
        pl.bind(sim)
        return np.array(pl.ready), pl, sim

    a, pl, sim = ready_after(0.0)
    b, _, _ = ready_after(5000.0)
    assert np.array_equal(a, b)
    want = sim.t_created + prep_model.quantile_s(orders.restaurant, sim.t_created, np.full(2, sim.traffic.rain), pl.prep_quantile)
    assert np.allclose(a, want)
    assert pl.name == "B1+"

    sim.now = 43200.0
    u, v = np.array([10, 400]), np.array([500, 20])
    free = grid_paths.dist[u, v]
    got = pl.leg_times(sim, u, v)
    assert not np.allclose(got, free)  # o ETA corrige o tempo livre
    assert (got > 0.5 * free).all() and (got < 3.0 * free).all()


def test_planner_names_and_default_margins(prep_model, eta_model):
    assert P.Planner().name == "B1" and P.Planner(oracle=True).name == "B2"
    assert P.Planner(prep_model=prep_model).name == "B1+preparo"
    assert P.Planner(eta_model=eta_model[0]).name == "B1+ETA"
    assert P.Planner(prep_model=prep_model).depart_margin_s == P.DEPART_MARGIN_LEARNED_S
    assert P.Planner(eta_model=eta_model[0]).depart_margin_s == P.DEPART_MARGIN_B1_S


def test_b1_plus_beats_b1_on_a_busy_grid_day(grid_city, grid_world, grid_paths, prep_model, eta_model):
    fleet = S.generate_fleet(grid_world, grid_city, 60, seed=7)
    ontime = {}
    for name, make in (("b1", P.Planner), ("b1+", lambda: P.Planner(prep_model=prep_model, eta_model=eta_model[0]))):
        ontime[name] = np.mean([S.summarize(S.run_day(grid_city, grid_world, grid_paths, fleet, make(), d))["on_time_pct"] for d in (1, 2, 3)])
    assert ontime["b1+"] > ontime["b1"]
