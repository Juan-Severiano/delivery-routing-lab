import numpy as np
import pytest

from delivery_lab import traffic as T
from delivery_lab.paths import Paths, adjacency_lists, apsp_python, dijkstra_tree, free_flow_seconds


@pytest.fixture(scope="module")
def paths(grid_city):
    dist, pred = apsp_python(grid_city)
    return Paths(grid_city, dist, pred)


def test_every_pair_is_reachable_and_the_diagonal_is_zero(grid_city, paths):
    assert np.isfinite(paths.dist).all()
    assert (np.diag(paths.dist) == 0).all()


def test_path_edges_chain_from_origin_to_destination_and_add_up_to_the_distance(grid_city, paths):
    rng = np.random.default_rng(0)
    weight = free_flow_seconds(grid_city)
    for u, v in rng.integers(0, grid_city.n_nodes, size=(60, 2)):
        if u == v:
            continue
        edges = paths.edges(int(u), int(v))
        assert grid_city.edge_src[edges[0]] == u and grid_city.edge_dst[edges[-1]] == v
        assert all(grid_city.edge_dst[a] == grid_city.edge_src[b] for a, b in zip(edges, edges[1:]))
        assert weight[edges].sum() == pytest.approx(paths.seconds(int(u), int(v)), rel=1e-4)


def test_triangle_inequality_holds(grid_city, paths):
    rng = np.random.default_rng(1)
    for u, v, w in rng.integers(0, grid_city.n_nodes, size=(200, 3)):
        assert paths.dist[u, w] <= paths.dist[u, v] + paths.dist[v, w] + 1e-2


def test_the_avenue_is_faster_than_a_residential_street(grid_city, paths):
    # a linha do meio é `primary` (50 km/h) e as outras são residenciais (25 km/h): ao longo da avenida
    # o tempo é mais ou menos a metade do de uma rua paralela do mesmo comprimento
    size = 40
    node = lambda r, c: r * size + c  # noqa: E731  (índices de nó seguem a ordem de criação do fixture)
    xy = grid_city.node_xy
    left = int(np.argmin(np.hypot(*(xy - [xy[:, 0].min(), xy[:, 1].mean()]).T)))
    right = int(np.argmin(np.hypot(*(xy - [xy[:, 0].max(), xy[:, 1].mean()]).T)))
    edge_cls = grid_city.edge_cls[paths.edges(left, right)]
    assert (edge_cls == 1).mean() > 0.9  # o caminho mínimo escolhe a avenida


def test_unreachable_destination_raises(grid_city, paths):
    broken = Paths(grid_city, paths.dist, np.full_like(paths.pred, -1))
    with pytest.raises(ValueError):
        broken.edges(0, 5)


def test_save_and_load_roundtrip(grid_city, paths, tmp_path):
    paths.save(tmp_path)
    again = Paths.load(grid_city, tmp_path)
    assert np.array_equal(again.dist, paths.dist) and np.array_equal(again.pred, paths.pred)
    with pytest.raises(FileNotFoundError):
        Paths.load(grid_city, tmp_path / "nada")


def test_dijkstra_matches_a_brute_force_on_a_small_graph(grid_city):
    adj = adjacency_lists(grid_city, free_flow_seconds(grid_city))
    dist, _ = dijkstra_tree(adj, 0)
    # Bellman-Ford de conferência, vetorizado
    best = np.full(grid_city.n_nodes, np.inf)
    best[0] = 0.0
    w = free_flow_seconds(grid_city)
    for _ in range(grid_city.n_nodes):
        relaxed = best.copy()
        np.minimum.at(relaxed, grid_city.edge_dst, best[grid_city.edge_src] + w)
        if np.array_equal(relaxed, best):
            break
        best = relaxed
    assert np.allclose(dist, best, rtol=1e-4)


def test_traffic_is_deterministic_and_changes_with_the_day(grid_city):
    a, b = T.build_traffic(grid_city, 7, 1), T.build_traffic(grid_city, 7, 1)
    assert np.array_equal(a.edge_seconds, b.edge_seconds)
    assert not np.array_equal(a.edge_seconds, T.build_traffic(grid_city, 7, 2).edge_seconds)
    assert a.edge_seconds.shape == (24, grid_city.n_edges)


def test_rush_hour_is_slower_than_the_night_in_the_center(grid_city):
    traffic = T.build_traffic(grid_city, 7, 1)
    ratio = traffic.edge_seconds / free_flow_seconds(grid_city)[None, :]
    assert ratio[18].mean() > ratio[4].mean() * 1.05
    assert ratio[12].mean() > ratio[4].mean() * 1.05
    assert (traffic.edge_seconds > 0).all()


def test_rush_intensity_peaks_in_the_evening_and_vanishes_at_night():
    assert T.rush(np.array([18.5]))[0] > 0.9 and T.rush(np.array([4.0]))[0] < 0.01


def test_rain_makes_the_peak_slower(grid_city):
    days = [T.build_traffic(grid_city, 7, d) for d in range(1, 41)]
    rainy = [t for t in days if t.rain]
    dry = [t for t in days if not t.rain]
    assert rainy and dry  # com 20% de chuva, 40 dias têm dos dois

    def peak_ratio(t):
        return (t.edge_seconds[18] / free_flow_seconds(grid_city)).mean() / t.day_factor

    assert np.mean([peak_ratio(t) for t in rainy]) > np.mean([peak_ratio(t) for t in dry]) * 1.02


def test_center_is_more_congested_than_the_edge(grid_city):
    traffic = T.build_traffic(grid_city, 7, 3)
    center = T.edge_centrality(grid_city)
    ratio = (traffic.edge_seconds[18] / free_flow_seconds(grid_city)) / traffic.day_factor
    near = ratio[center > np.quantile(center, 0.8)].mean()
    far = ratio[center < np.quantile(center, 0.2)].mean()
    assert near > far
