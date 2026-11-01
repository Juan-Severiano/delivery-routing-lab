import json

import numpy as np
import pytest

from delivery_lab import sim as S
from delivery_lab.replay import render_replay


@pytest.fixture(scope="module")
def replay(grid_city, grid_world, grid_paths):
    fleet = S.generate_fleet(grid_world, grid_city, 70, seed=7)
    result = S.run_day(grid_city, grid_world, grid_paths, fleet, S.NearestPolicy(), day=1, record=True)
    return render_replay(result, grid_city, grid_world, "B0", 1), result


def test_columns_are_consistent(replay):
    payload, result = replay
    legs, n_legs = payload["legs"], len(payload["legs"]["t0"])
    assert len(legs["kind"]) == len(legs["order"]) == n_legs and len(legs["ptr"]) == n_legs + 1
    assert legs["ptr"][0] == 0 and legs["ptr"][-1] == len(payload["leg_edges"]) == len(payload["leg_deciseconds"])
    assert all(a <= b for a, b in zip(legs["ptr"], legs["ptr"][1:]))
    first = payload["couriers"]["first_leg"]
    assert len(first) == len(result.fleet) + 1 and first[0] == 0 and first[-1] == n_legs
    assert all(len(v) == len(payload["orders"]["x"]) for v in payload["orders"].values())


def test_legs_of_a_courier_are_in_time_order(replay):
    payload, _ = replay
    first, t0 = payload["couriers"]["first_leg"], payload["legs"]["t0"]
    for a, b in zip(first, first[1:]):
        assert t0[a:b] == sorted(t0[a:b])


def test_edge_table_covers_every_edge_and_the_payload_is_json(replay, grid_city):
    payload, _ = replay
    assert len(payload["edges"]["geom"]) == len(payload["edges"]["rev"]) == grid_city.n_edges
    assert max(payload["leg_edges"]) < grid_city.n_edges
    json.dumps(payload)


def test_replay_leg_durations_match_the_simulation(replay):
    payload, result = replay
    legs = [leg for courier in result.legs for leg in courier]
    ptr, ds = payload["legs"]["ptr"], payload["leg_deciseconds"]
    for i in (0, len(legs) // 2, len(legs) - 1):
        assert sum(ds[ptr[i] : ptr[i + 1]]) / 10.0 == pytest.approx(float(legs[i].seconds.sum()), abs=0.1 * max(1, len(legs[i].seconds)))


def test_summary_is_in_the_meta(replay):
    payload, _ = replay
    summary = payload["meta"]["summary"]
    assert payload["meta"]["policy"] == "B0"
    assert summary["delivered"] + summary["unserved"] == payload["meta"]["orders"]
    assert sum(payload["orders"]["rejected"]) == summary["unserved"]
    assert 0 < payload["meta"]["summary"]["on_time_pct"] <= 100


def test_undelivered_orders_become_null():
    from delivery_lab.replay import _nullable

    assert _nullable(np.array([1.234, np.nan])) == [1.2, None]


def test_replay_requires_recorded_legs(grid_city, grid_world, grid_paths):
    fleet = S.generate_fleet(grid_world, grid_city, 10, seed=7)
    result = S.run_day(grid_city, grid_world, grid_paths, fleet, S.NearestPolicy(), day=1, record=False)
    with pytest.raises(ValueError):
        render_replay(result, grid_city, grid_world, "B0", 1)
