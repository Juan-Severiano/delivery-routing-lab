import itertools
import os

import helpers
import numpy as np
import pytest

from delivery_lab import planner as P
from delivery_lab import sim as S
from delivery_lab.planner import Planner, _valid
from delivery_lab.sim import DROPOFF, PICKUP


def bound(city, world, paths, planner, orders, fleet, day=1):
    sim = helpers.make_sim(city, world, paths, fleet, orders, planner, day)
    planner.bind(sim)
    sim.shift_ok[:] = True
    return sim


def all_pairs_travel(planner, sim, nodes):
    nodes = sorted(set(nodes))
    pairs = [(a, b) for a in nodes for b in nodes if a != b]
    times = planner.leg_times(sim, np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
    return dict(zip(pairs, times.tolist()))


def brute_force(planner, orders, node, t0, travel):
    """Melhor sequência entre todas as ordens válidas de coletas e entregas (até 3 pedidos = 90)."""
    stops = [(k, o) for o in orders for k in (PICKUP, DROPOFF)]
    best = min(
        (list(p) for p in itertools.permutations(stops) if _valid(list(p))),
        key=lambda seq: planner.cost(seq, node, t0, travel),
    )
    return best, planner.cost(best, node, t0, travel)


@pytest.mark.parametrize("oracle", [False, True])
def test_plans_respect_precedence_and_the_bag_all_day(grid_city, grid_world, grid_paths, oracle):
    checked = {"n": 0}

    class Watching(Planner):
        def assign(self, sim, order):
            c = super().assign(sim, order)
            if c >= 0:
                for plan, leg in zip(self.plan, sim.leg_stop):
                    stops = plan + ([leg] if leg else [])
                    assert _valid(plan)
                    assert len({o for _, o in stops}) <= sim.config.capacity
                checked["n"] += 1
            return c

    fleet = S.generate_fleet(grid_world, grid_city, 25, seed=7)  # pouca gente: planos cheios
    result = S.run_day(grid_city, grid_world, grid_paths, fleet, Watching(oracle=oracle), day=2)
    assert checked["n"] > 500
    ok = ~result.rejected
    assert not np.isnan(result.t_delivered[ok]).any()  # todo pedido aceito sai
    assert (result.t_picked[ok] >= result.t_ready[ok] + result.config.handover_s - 1e-6).all()
    assert (result.t_assigned[ok] <= result.t_picked[ok]).all() and (result.t_picked[ok] < result.t_delivered[ok]).all()


def test_two_opt_never_makes_a_plan_worse_and_keeps_it_valid(grid_city, grid_world, grid_paths):
    rng = np.random.default_rng(3)
    orders = helpers.tiny_orders(grid_city, [0, 1, 2], rng.integers(0, grid_city.n_nodes, 3), [100.0, 130.0, 160.0])
    for oracle in (False, True):
        pl = Planner(oracle=oracle)
        sim = bound(grid_city, grid_world, grid_paths, pl, orders, helpers.always_on_fleet([5]))
        sim.now = 200.0
        nodes = [5] + [pl.shop_node[o] for o in range(3)] + [pl.home_node[o] for o in range(3)]
        travel = all_pairs_travel(pl, sim, nodes)
        stops = [(k, o) for o in range(3) for k in (PICKUP, DROPOFF)]
        for _ in range(60):
            seq = list(rng.permutation(len(stops)))
            seq = [stops[i] for i in seq]
            if not _valid(seq):
                continue
            after = pl.two_opt(seq, [], 5, 200.0, travel)
            assert _valid(after) and sorted(after) == sorted(seq)
            assert pl.cost(after, 5, 200.0, travel) <= pl.cost(seq, 5, 200.0, travel) + 1e-9


def test_insertion_and_two_opt_are_close_to_the_optimum_on_three_orders(grid_city, grid_world, grid_paths, capsys):
    rng = np.random.default_rng(11)
    optimal, excess = 0, []
    n = int(os.environ.get("OPTIMALITY_INSTANCES", "120"))
    for trial in range(n):
        customers = rng.integers(0, grid_city.n_nodes, 3)
        shops = rng.integers(0, grid_world.n_restaurants, 3)
        t_created = 1000.0 + np.sort(rng.uniform(0, 600, 3))
        orders = helpers.tiny_orders(grid_city, shops, customers, t_created)
        start = int(rng.integers(0, grid_city.n_nodes))
        pl = Planner(oracle=trial % 2 == 1)
        sim = bound(grid_city, grid_world, grid_paths, pl, orders, helpers.always_on_fleet([start]))
        sim.now = float(t_created[-1])
        nodes = [start] + [pl.shop_node[o] for o in range(3)] + [pl.home_node[o] for o in range(3)]
        travel = all_pairs_travel(pl, sim, nodes)
        plan: list = []
        for o in range(3):  # os três pedidos entram um a um, como na simulação
            _, plan = pl.best_insertion(o, [], plan, start, sim.now, travel)
        plan = pl.two_opt(plan, [], start, sim.now, travel)
        _, best_cost = brute_force(pl, range(3), start, sim.now, travel)
        got = pl.cost(plan, start, sim.now, travel)
        assert got >= best_cost - 1e-6
        optimal += got <= best_cost + 1e-6
        excess.append(100.0 * (got - best_cost) / best_cost)
    with capsys.disabled():
        print(f"\n[inserção + 2-opt contra força bruta, 3 pedidos, {n} instâncias] "
              f"ótimo em {100 * optimal / n:.0f}%, custo a mais: média {np.mean(excess):.2f}%, máx {np.max(excess):.1f}%")
    assert optimal / n >= 0.75 and np.mean(excess) < 3.0


def test_b1_cannot_see_the_real_ready_time_but_b2_can(grid_city, grid_world, grid_paths):
    orders = helpers.tiny_orders(grid_city, [0], [grid_city.n_nodes - 1], [1000.0])
    fleet = helpers.always_on_fleet([10, 400, 800, 1200])

    def decide(oracle, ready_shift):
        pl = Planner(oracle=oracle)
        sim = helpers.make_sim(grid_city, grid_world, grid_paths, fleet, orders, pl)
        sim.t_ready = sim.t_ready + ready_shift  # o preparo "de verdade" muda; o B1 não pode notar
        pl.bind(sim)
        sim.shift_ok[:] = True
        sim.now = 1000.0
        c = pl.assign(sim, 0)
        return c, list(pl.plan[c]), pl.ready[0]

    b1_a, b1_b = decide(False, 0.0), decide(False, 3000.0)
    assert b1_a == b1_b
    b2_a, b2_b = decide(True, 0.0), decide(True, 3000.0)
    assert b2_b[2] == pytest.approx(b2_a[2] + 3000.0)  # o B2 lê a hora real


def test_b2_leg_times_equal_the_traffic_sum_over_the_path(grid_city, grid_world, grid_paths):
    orders = helpers.tiny_orders(grid_city, [0], [5], [1000.0])
    pl = Planner(oracle=True)
    sim = bound(grid_city, grid_world, grid_paths, pl, orders, helpers.always_on_fleet([0]))
    sim.now = 18 * 3600.0 + 120.0
    rng = np.random.default_rng(5)
    u, v = rng.integers(0, grid_city.n_nodes, 40), rng.integers(0, grid_city.n_nodes, 40)
    got = pl.leg_times(sim, u, v)
    for k in range(40):
        want = sim.traffic.leg_seconds(grid_paths.edges(int(u[k]), int(v[k])), 18) if u[k] != v[k] else 0.0
        assert got[k] == pytest.approx(want, rel=1e-5)


def test_the_planner_holds_the_courier_until_the_order_is_ready(grid_city, grid_world, grid_paths):
    shop = int(grid_world.restaurant_node[0])
    courier_node = int(grid_world.restaurant_node[1])
    customer = int(np.argmax(np.hypot(*(grid_city.node_xy - grid_city.node_xy[shop]).T)))
    orders = helpers.tiny_orders(grid_city, [0], [customer], [12 * 3600.0])
    fleet = helpers.always_on_fleet([courier_node])
    waits = {}
    for name, policy in (("b0", S.NearestPolicy()), ("b2", Planner(oracle=True))):
        r = helpers.make_sim(grid_city, grid_world, grid_paths, fleet, orders, policy).run()
        waits[name] = r.wait_at_restaurant_s[0]
        if name == "b2":
            assert r.legs[0][0].t0 > orders.t_s[0]  # só sai quando vale a pena
            assert r.t_picked[0] - r.t_ready[0] < 120.0 + r.config.handover_s  # chega quase junto do preparo
    assert waits["b0"] > 300.0 and waits["b2"] < 60.0


def test_a_new_close_order_can_jump_ahead_of_a_pickup_that_is_still_cooking(grid_city, grid_world, grid_paths):
    # restaurante 0 demora; um pedido de restaurante 1, vizinho do entregador, entra antes e é entregue primeiro
    near = int(grid_world.restaurant_node[1])
    orders = helpers.tiny_orders(grid_city, [0, 1], [grid_city.n_nodes - 2, near + 1], [43200.0, 43230.0])
    pl = Planner(oracle=True)
    sim = helpers.make_sim(grid_city, grid_world, grid_paths, helpers.always_on_fleet([near]), orders, pl)
    sim.t_ready[0] = 43200.0 + 1500.0  # 25 min de preparo
    sim.t_ready[1] = 43230.0 + 120.0  # o outro fica pronto em 2 min
    result = sim.run()
    assert result.t_delivered[1] < result.t_picked[0]  # o pedido rápido sai antes da coleta do lento
    assert not np.isnan(result.t_delivered).any()


def test_b2_beats_the_floor_on_a_busy_day(grid_city, grid_world, grid_paths):
    fleet = S.generate_fleet(grid_world, grid_city, 60, seed=7)
    ontime = {}
    for name, policy in (("b0", S.NearestPolicy()), ("b2", Planner(oracle=True))):
        ontime[name] = np.mean([S.summarize(S.run_day(grid_city, grid_world, grid_paths, fleet, policy, d))["on_time_pct"] for d in (1, 2)])
    assert ontime["b2"] > ontime["b0"]
