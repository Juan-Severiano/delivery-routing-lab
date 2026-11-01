from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from delivery_lab import world as w
from delivery_lab.city import City, urban_focus

DEV = dict(restaurants=40, orders_per_day=900, active_share=1.0)


@pytest.fixture(scope="module")
def world(grid_city):
    return w.generate_world(grid_city, seed=7, **DEV)


def test_same_seed_gives_the_same_world_and_the_same_day(grid_city, world):
    again = w.generate_world(grid_city, seed=7, **DEV)
    assert np.array_equal(world.restaurant_node, again.restaurant_node)
    assert np.array_equal(world.prep_mean_min, again.prep_mean_min)
    a, b = w.sample_day(world, grid_city, 3), w.sample_day(again, grid_city, 3)
    assert np.array_equal(a.t_s, b.t_s) and np.array_equal(a.restaurant, b.restaurant)
    assert np.array_equal(a.customer_node, b.customer_node)


def test_other_seeds_and_other_days_differ(grid_city, world):
    other = w.generate_world(grid_city, seed=8, **DEV)
    assert not np.array_equal(world.restaurant_node, other.restaurant_node)
    d1, d2 = w.sample_day(world, grid_city, 1), w.sample_day(world, grid_city, 2)
    assert len(d1) != len(d2) or not np.array_equal(d1.t_s, d2.t_s)


def test_restaurants_sit_on_distinct_graph_nodes(grid_city, world):
    assert world.n_restaurants == 40
    assert len(set(world.restaurant_node.tolist())) == 40
    assert np.allclose(world.restaurant_xy, grid_city.node_xy[world.restaurant_node])


def test_restaurant_kinds_follow_the_configured_shares(grid_city):
    big = w.generate_world(grid_city, seed=1, restaurants=400, orders_per_day=100, active_share=1.0)
    observed = np.bincount(big.restaurant_kind, minlength=len(w.KINDS)) / big.n_restaurants
    expected = np.array([k.share for k in w.KINDS])
    assert np.abs(observed - expected).max() < 0.07


def test_orders_are_sorted_and_inside_the_day(grid_city, world):
    day = w.sample_day(world, grid_city, 1)
    assert len(day) > 500
    assert (np.diff(day.t_s) >= 0).all()
    assert day.t_s.min() >= 0 and day.t_s.max() < 86_400


def test_every_order_has_an_open_restaurant_within_the_radius(grid_city, world):
    day = w.sample_day(world, grid_city, 1)
    for i in range(len(day)):
        r = day.restaurant[i]
        assert world.is_open(day.t_s[i] / 3600.0)[r]
        assert np.hypot(*(day.customer_xy[i] - world.restaurant_xy[r])) <= world.max_radius_m
    assert (day.customer_xy == grid_city.node_xy[day.customer_node]).all()


def test_window_that_passes_midnight_stays_open_after_it(world):
    late = replace(
        world,
        open_from=np.full(world.n_restaurants, 18.0),
        open_until=np.full(world.n_restaurants, 25.0),
        active=np.ones(world.n_restaurants, bool),
    )
    assert late.is_open(23.0).all() and late.is_open(0.5).all()
    assert not late.is_open(1.5).any() and not late.is_open(12.0).any()


def test_inactive_restaurants_never_open(world):
    half = replace(world, active=np.arange(world.n_restaurants) % 2 == 0)
    for hour in (12.5, 19.5, 22.0):
        assert not (half.is_open(hour) & ~half.active).any()


def test_demand_curve_is_a_distribution_with_lunch_and_dinner_peaks():
    curve = w.demand_curve()
    assert curve.sum() == pytest.approx(1.0)
    by_hour = curve.reshape(24, 60).sum(axis=1)
    assert by_hour.argmax() == 19  # o jantar é o maior pico
    assert by_hour[12] > 2 * by_hour[15] and by_hour[12] > 10 * by_hour[3]


def test_hourly_orders_follow_the_demand_curve_when_restaurants_are_open(grid_city):
    busy = w.generate_world(grid_city, seed=7, restaurants=300, orders_per_day=3000, active_share=1.0)
    counts = np.zeros(24)
    total = 0
    for day in range(1, 21):
        orders = w.sample_day(busy, grid_city, day)
        counts += np.bincount((orders.t_s // 3600).astype(int), minlength=24)
        total += len(orders)
    share = counts / total
    expected = busy.minute_curve.reshape(24, 60).sum(axis=1)
    for hour in (12, 19):  # horas em que há restaurante aberto de sobra
        assert share[hour] == pytest.approx(expected[hour], rel=0.12)


def test_preparation_is_slower_at_the_peak_than_at_night(world):
    assert (world.prep_minutes(19.8) > world.prep_minutes(4.0)).all()
    assert (world.prep_minutes(4.0) == pytest.approx(world.prep_mean_min * (1 + (world.rush_gain - 1) * world.minute_curve[240] / world.minute_curve.max())))


def test_heat_adds_up_to_the_average_orders_per_day(grid_city, world):
    heat = w.heat_by_hour(world, grid_city, days=3)
    in_heat = sum(sum(values) for _, _, values in heat["cells"])
    served = np.mean([len(w.sample_day(world, grid_city, d)) for d in (1, 2, 3)])
    assert in_heat == pytest.approx(served, rel=0.01)
    assert all(len(values) == 24 for _, _, values in heat["cells"])


def test_payload_is_json_and_has_everything_the_map_needs(grid_city, world):
    import json

    payload = w.render_payload(world, grid_city, days=2)
    json.dumps(payload)
    r = payload["restaurants"]
    assert len(r["x"]) == len(r["kind"]) == len(r["active"]) == world.n_restaurants
    assert len(payload["online_by_hour"]) == 24 and len(payload["curve_by_hour"]) == 24
    assert payload["day"]["count"] == len(payload["day"]["x"]) == len(payload["day"]["restaurant"])


REAL = Path(__file__).resolve().parents[1] / "data" / "out" / "city.npz"


@pytest.mark.skipif(not REAL.exists(), reason="rode `delivery-lab city build` antes")
def test_platform_scale_on_the_real_city():
    city = City.load(REAL)
    big = w.generate_world(city, seed=7, **{**w.PRESETS["plataforma"]})
    center, radius = urban_focus(city.node_xy)
    near = np.hypot(*(big.restaurant_xy - center).T) < 4 * radius
    assert near.mean() > 0.9  # restaurantes no miolo urbano, não na estrada rural
    online = big.open_matrix(np.arange(24) + 0.5).sum(axis=1)
    assert 150 <= online.max() <= 260  # a ordem de grandeza combinada: ~200 online por vez
    day = w.sample_day(big, city, 1)
    assert 5000 <= len(day) + day.dropped <= 6200
    assert day.dropped / (len(day) + day.dropped) < 0.08
