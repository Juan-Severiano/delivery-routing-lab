from dataclasses import replace

import helpers
import numpy as np
import pytest

from delivery_lab import sim as S
from delivery_lab import world as W
from delivery_lab.traffic import build_traffic


def tiny_orders(world, city, restaurants, customers, times):
    return helpers.tiny_orders(city, restaurants, customers, times)


def always_on_fleet(city, nodes):
    return helpers.always_on_fleet(nodes)


def run(city, world, paths, fleet, orders, policy=None, day=1, record=True, config=None):
    return helpers.make_sim(city, world, paths, fleet, orders, policy or S.NearestPolicy(), day, record, config).run()


@pytest.fixture(scope="module")
def day_result(grid_city, grid_world, grid_paths):
    fleet = S.generate_fleet(grid_world, grid_city, 70, seed=7)
    return S.run_day(grid_city, grid_world, grid_paths, fleet, S.NearestPolicy(), day=1, record=True)


def test_every_accepted_order_is_delivered_with_a_consistent_timeline(day_result):
    r = day_result
    ok = ~r.rejected
    assert ok.sum() > 500
    assert not np.isnan(r.t_delivered[ok]).any()
    assert (r.t_created[ok] <= r.t_assigned[ok]).all()
    assert (r.t_assigned[ok] <= r.t_picked[ok]).all()
    assert (r.t_picked[ok] < r.t_delivered[ok]).all()
    assert (r.courier[ok] >= 0).all()


def test_orders_created_while_nobody_is_on_shift_are_rejected(day_result):
    r = day_result
    first_shift = r.fleet.shift_start_s.min()
    last_shift_end = r.fleet.shift_end_s.max()
    assert r.rejected.any()
    assert (r.rejected == ((r.t_created < first_shift) | (r.t_created >= last_shift_end))).all()
    assert np.isnan(r.t_delivered[r.rejected]).all() and (r.courier[r.rejected] == -1).all()


def test_nobody_collects_before_the_order_is_ready(day_result):
    r = day_result
    ok = ~r.rejected
    assert (r.t_picked[ok] >= r.t_ready[ok] + r.config.handover_s - 1e-6).all()


def test_orders_are_only_assigned_to_couriers_on_shift(day_result):
    r = day_result
    ok = ~r.rejected
    start, end = r.fleet.shift_start_s[r.courier[ok]], r.fleet.shift_end_s[r.courier[ok]]
    assert (r.t_assigned[ok] >= start - 1e-6).all() and (r.t_assigned[ok] < end).all()


def test_the_bag_never_exceeds_the_capacity(grid_city, grid_world, grid_paths):
    class Watching(S.NearestPolicy):
        peak = 0

        def next_stop(self, sim, courier):
            Watching.peak = max(Watching.peak, int(sim.load.max()), max(len(b) for b in sim.bag))
            return super().next_stop(sim, courier)

    fleet = S.generate_fleet(grid_world, grid_city, 25, seed=7)  # pouca gente: força lotes
    S.run_day(grid_city, grid_world, grid_paths, fleet, Watching(), day=2)
    assert Watching.peak == S.SimConfig().capacity


def test_legs_chain_through_the_graph_and_match_the_distance_driven(day_result, grid_city):
    r = day_result
    for c, legs in enumerate(r.legs):
        total = 0.0
        position = int(r.fleet.start_node[c])
        for leg in legs:
            if len(leg.edges):
                assert grid_city.edge_src[leg.edges[0]] == position
                assert all(grid_city.edge_dst[a] == grid_city.edge_src[b] for a, b in zip(leg.edges, leg.edges[1:]))
                position = int(grid_city.edge_dst[leg.edges[-1]])
                total += float(grid_city.edge_len[leg.edges].sum())
            assert (leg.seconds > 0).all()
        assert total == pytest.approx(r.courier_distance_m[c], rel=1e-6)


def test_same_inputs_give_the_same_day(grid_city, grid_world, grid_paths):
    fleet = S.generate_fleet(grid_world, grid_city, 70, seed=7)
    a = S.run_day(grid_city, grid_world, grid_paths, fleet, S.NearestPolicy(), day=3)
    b = S.run_day(grid_city, grid_world, grid_paths, fleet, S.NearestPolicy(), day=3)
    assert S.summarize(a) == S.summarize(b)
    assert np.array_equal(a.t_delivered, b.t_delivered, equal_nan=True)


def test_one_order_one_courier_follows_the_hand_computed_timeline(grid_city, grid_world, grid_paths):
    r = 0
    shop = int(grid_world.restaurant_node[r])
    courier_node = int(grid_world.restaurant_node[1])
    customer = int(np.argmax(np.hypot(*(grid_city.node_xy - grid_city.node_xy[shop]).T)))
    orders = tiny_orders(grid_world, grid_city, [r], [customer], [12 * 3600.0])
    result = run(grid_city, grid_world, grid_paths, always_on_fleet(grid_city, [courier_node]), orders)
    leg_to_shop, leg_to_customer = result.legs[0]
    cfg = result.config
    arrive_shop = leg_to_shop.t0 + leg_to_shop.seconds.sum()
    wait = max(0.0, result.t_ready[0] - arrive_shop)
    assert leg_to_shop.t0 == pytest.approx(orders.t_s[0])  # sai assim que o pedido nasce (B0)
    assert result.wait_at_restaurant_s[0] == pytest.approx(wait)
    assert result.t_picked[0] == pytest.approx(arrive_shop + wait + cfg.handover_s)
    assert leg_to_customer.t0 == pytest.approx(result.t_picked[0])
    arrive_customer = leg_to_customer.t0 + leg_to_customer.seconds.sum()
    assert result.t_delivered[0] == pytest.approx(arrive_customer + cfg.delivered_after_arrival_s)
    assert S.summarize(result)["unserved"] == 0


def test_b0_prefers_a_free_courier_over_a_nearer_one_that_is_already_loaded(grid_city, grid_world, grid_paths):
    shop = int(grid_world.restaurant_node[0])
    near = shop
    far = int(np.argmax(np.hypot(*(grid_city.node_xy - grid_city.node_xy[shop]).T)))
    orders = tiny_orders(grid_world, grid_city, [0], [near], [100.0])
    traffic = build_traffic(grid_city, grid_world.seed, 1)
    sim = S.Sim(grid_city, grid_world, grid_paths, traffic, always_on_fleet(grid_city, [near, far]), orders, S.NearestPolicy())
    sim.shift_ok[:] = True
    sim.load[0] = 1  # o de perto já carrega um pedido
    assert S.NearestPolicy().assign(sim, 0) == 1
    sim.load[0] = 0  # com os dois livres, vale o mais próximo
    assert S.NearestPolicy().assign(sim, 0) == 0
    sim.load[:] = sim.config.capacity  # ninguém com espaço
    assert S.NearestPolicy().assign(sim, 0) == -1


def test_orders_wait_in_the_queue_until_a_courier_starts_the_shift(grid_city, grid_world, grid_paths):
    # há um entregador de turno (o primeiro), mas ele está ocupado: o pedido espera na fila
    orders = tiny_orders(grid_world, grid_city, [0, 0], [5, 6], [1000.0, 1001.0])
    one = S.Fleet(np.array([0.0]), np.array([30000.0]), np.array([3], dtype=np.int32))
    cfg = S.SimConfig(capacity=1)
    result = run(grid_city, grid_world, grid_paths, one, orders, config=cfg)
    assert not result.rejected.any()
    assert result.t_assigned[1] >= result.t_delivered[0] - 1e-6  # só entra quando o primeiro sai da mochila


def test_an_order_with_nobody_on_shift_is_rejected_and_left_out_of_the_metrics(grid_city, grid_world, grid_paths):
    orders = tiny_orders(grid_world, grid_city, [0, 0], [5, 6], [1000.0, 5000.0])
    shift = S.Fleet(np.array([3000.0]), np.array([30000.0]), np.array([3], dtype=np.int32))
    result = run(grid_city, grid_world, grid_paths, shift, orders)
    assert result.rejected.tolist() == [True, False]
    assert np.isnan(result.t_delivered[0]) and not np.isnan(result.t_delivered[1])
    m = S.summarize(result)
    assert m["delivered"] == 1 and m["unserved"] == 1 and m["rejected_pct"] == pytest.approx(50.0)


def test_more_couriers_means_faster_deliveries(grid_city, grid_world, grid_paths):
    means = []
    for n in (30, 70, 140):
        fleet = S.generate_fleet(grid_world, grid_city, n, seed=7)
        means.append(S.summarize(S.run_day(grid_city, grid_world, grid_paths, fleet, S.NearestPolicy(), day=1))["delivery_mean_min"])
    assert means[0] > means[1] > means[2]


def test_summary_of_a_hand_made_result(day_result):
    r = replace(
        day_result,
        t_created=np.array([0.0, 0.0, 0.0, 0.0]),
        t_assigned=np.array([60.0, 60.0, 120.0, 0.0]),
        t_delivered=np.array([1800.0, 3000.0, 2400.0, np.nan]),
        promised=np.array([2000.0, 2400.0, 2400.0, 2000.0]),
        wait_at_restaurant_s=np.zeros(4),
    )
    m = S.summarize(r)
    assert m["delivered"] == 3 and m["unserved"] == 1
    assert m["on_time_pct"] == pytest.approx(100 * 2 / 3)  # o 1 e o 3 chegam a tempo; o 2 atrasa 10 min
    assert m["late_mean_min"] == pytest.approx(10 / 3)
    assert m["delivery_mean_min"] == pytest.approx((30 + 50 + 40) / 3)
