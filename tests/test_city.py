import json
import math
from pathlib import Path

import numpy as np
import pytest

from delivery_lab.city import City, build_city, free_flow_kmh, largest_scc, render_payload, urban_focus
from delivery_lab.osm import parse_maxspeed, parse_oneway, read_osm

DEG = 1e-3  # 0,001 grau perto da linha do equador = ~111,3 m
METERS_PER_DEG = math.radians(1) * 6_371_008.8


def write_osm(tmp_path: Path, nodes: dict[int, tuple[float, float]], ways: list[dict]) -> Path:
    elements = [{"type": "node", "id": i, "lat": lat, "lon": lon} for i, (lat, lon) in nodes.items()]
    elements += [{"type": "way", "id": 1000 + k, **w} for k, w in enumerate(ways)]
    path = tmp_path / "osm.json"
    path.write_text(json.dumps({"elements": elements}))
    return path


def build(tmp_path, nodes, ways):
    return build_city(read_osm(write_osm(tmp_path, nodes, ways)))


def test_chain_between_junctions_becomes_one_edge(tmp_path):
    # 1 - 2 - 3 - 4 em linha: só as pontas são cruzamentos
    nodes = {i: (0.0, (i - 1) * DEG) for i in (1, 2, 3, 4)}
    city, report = build(tmp_path, nodes, [{"nodes": [1, 2, 3, 4], "tags": {"highway": "residential"}}])
    assert city.n_nodes == 2
    assert city.n_edges == 2  # mão dupla: ida e volta
    assert city.n_geoms == 1
    assert city.edge_len[0] == pytest.approx(3 * DEG * METERS_PER_DEG, rel=1e-3)
    assert len(city.geometry(0)) == 4  # a geometria guarda os nós do meio
    assert report.chains == 1


def test_crossing_node_splits_both_ways(tmp_path):
    #      3
    #      |
    #  1 - 5 - 2     o nó 5 é usado pelas duas vias
    #      |
    #      4
    nodes = {1: (0, -DEG), 2: (0, DEG), 3: (DEG, 0), 4: (-DEG, 0), 5: (0, 0)}
    ways = [
        {"nodes": [1, 5, 2], "tags": {"highway": "residential"}},
        {"nodes": [3, 5, 4], "tags": {"highway": "residential"}},
    ]
    city, _ = build(tmp_path, nodes, ways)
    assert city.n_nodes == 5
    assert city.n_edges == 8
    assert city.n_geoms == 4


def test_oneway_creates_a_single_direction(tmp_path):
    nodes = {1: (0, 0), 2: (0, DEG), 3: (DEG, 0)}
    ways = [
        {"nodes": [1, 2], "tags": {"highway": "residential", "oneway": "yes"}},
        {"nodes": [3, 2], "tags": {"highway": "residential", "oneway": "-1"}},  # 2 -> 3
        {"nodes": [3, 1], "tags": {"highway": "residential"}},
    ]
    city, _ = build(tmp_path, nodes, ways)

    def node_at(lat, lon):
        return int(np.abs(city.to_latlon(city.node_xy) - [lat, lon]).sum(axis=1).argmin())

    n1, n2, n3 = node_at(0, 0), node_at(0, DEG), node_at(DEG, 0)
    edges = set(zip(city.edge_src.tolist(), city.edge_dst.tolist()))
    assert edges == {(n1, n2), (n2, n3), (n3, n1), (n1, n3)}


def test_dead_end_behind_a_one_way_street_is_dropped(tmp_path):
    # triângulo de mão dupla 1-2-3 e uma rua de mão única 3 -> 4 sem saída
    nodes = {1: (0, 0), 2: (0, DEG), 3: (DEG, 0), 4: (2 * DEG, 0)}
    ways = [
        {"nodes": [1, 2], "tags": {"highway": "residential"}},
        {"nodes": [2, 3], "tags": {"highway": "residential"}},
        {"nodes": [3, 1], "tags": {"highway": "residential"}},
        {"nodes": [3, 4], "tags": {"highway": "residential", "oneway": "yes"}},
    ]
    city, report = build(tmp_path, nodes, ways)
    assert city.n_nodes == 3
    assert report.nodes_before == 4 and report.nodes_after == 3
    assert city.n_edges == 6


def test_every_node_reaches_every_other_in_the_kept_graph():
    # anel de mão única com um atalho de ida só: uma componente só
    src = np.array([0, 1, 2, 2])
    dst = np.array([1, 2, 0, 3])
    mask = largest_scc(4, src, dst)
    assert mask.tolist() == [True, True, True, False]


def test_not_drivable_and_foreign_ways_are_ignored(tmp_path):
    nodes = {1: (0, 0), 2: (0, DEG), 3: (DEG, 0)}
    ways = [
        {"nodes": [1, 2], "tags": {"highway": "residential"}},
        {"nodes": [2, 1], "tags": {"highway": "footway"}},
        {"nodes": [1, 3], "tags": {"highway": "raceway"}},
    ]
    raw = read_osm(write_osm(tmp_path, nodes, ways))
    assert [w.highway for w in raw.ways] == ["residential"]


def test_roundabout_is_one_way_unless_tagged():
    assert parse_oneway({"junction": "roundabout"}) == 1
    assert parse_oneway({"junction": "roundabout", "oneway": "no"}) == 0
    assert parse_oneway({}) == 0


def test_maxspeed_parsing():
    assert parse_maxspeed("40") == 40
    assert parse_maxspeed("40 km/h") == 40
    assert parse_maxspeed("30 mph") == pytest.approx(48.28, abs=0.01)
    assert parse_maxspeed("BR:urban") is None
    assert parse_maxspeed("40;50") is None
    assert parse_maxspeed(None) is None


def test_speed_comes_from_maxspeed_then_class_and_links_are_slower():
    assert free_flow_kmh("residential", False, None) == 25
    assert free_flow_kmh("primary", False, None) == 50
    assert free_flow_kmh("primary", True, None) == pytest.approx(35)
    assert free_flow_kmh("residential", False, 60) == 60
    assert free_flow_kmh("residential", False, 5) == 10  # limites de sanidade
    assert free_flow_kmh("residential", False, 200) == 80


def test_signals_are_marked_on_nodes(tmp_path):
    nodes = {1: (0, 0), 2: (0, DEG), 3: (0, 2 * DEG)}
    path = write_osm(tmp_path, nodes, [{"nodes": [1, 2], "tags": {"highway": "residential"}},
                                       {"nodes": [2, 3], "tags": {"highway": "residential"}}])
    data = json.loads(path.read_text())
    for element in data["elements"]:
        if element["type"] == "node" and element["id"] == 2:
            element["tags"] = {"highway": "traffic_signals"}
    path.write_text(json.dumps(data))
    city, _ = build_city(read_osm(path))
    assert city.node_signal.sum() == 1


def test_adjacency_lists_outgoing_edges(tmp_path):
    nodes = {1: (0, 0), 2: (0, DEG), 3: (DEG, 0)}
    ways = [{"nodes": [1, 2], "tags": {"highway": "residential"}},
            {"nodes": [2, 3], "tags": {"highway": "residential"}},
            {"nodes": [3, 1], "tags": {"highway": "residential"}}]
    city, _ = build(tmp_path, nodes, ways)
    ptr, edges = city.adjacency
    for v in range(city.n_nodes):
        out = edges[ptr[v] : ptr[v + 1]]
        assert len(out) == 2
        assert all(city.edge_src[e] == v for e in out)


def test_edge_points_follow_the_travel_direction(tmp_path):
    nodes = {1: (0, 0), 2: (0, DEG), 3: (0, 2 * DEG)}
    city, _ = build(tmp_path, nodes, [{"nodes": [1, 2, 3], "tags": {"highway": "residential"}}])
    for e in range(city.n_edges):
        pts = city.edge_points(e)
        assert np.allclose(pts[0], city.node_xy[city.edge_src[e]])
        assert np.allclose(pts[-1], city.node_xy[city.edge_dst[e]])


def test_save_and_load_roundtrip(tmp_path):
    nodes = {1: (0, 0), 2: (0, DEG), 3: (DEG, 0)}
    ways = [{"nodes": [1, 2], "tags": {"highway": "residential"}},
            {"nodes": [2, 3], "tags": {"highway": "primary"}},
            {"nodes": [3, 1], "tags": {"highway": "residential"}}]
    city, _ = build(tmp_path, nodes, ways)
    city.save(tmp_path / "city.npz")
    again = City.load(tmp_path / "city.npz")
    assert again.origin == city.origin
    for name in ("node_xy", "edge_src", "edge_dst", "edge_len", "edge_speed", "edge_geom", "geom_pts"):
        assert np.array_equal(getattr(again, name), getattr(city, name)), name


def test_render_payload_lists_each_street_once(tmp_path):
    nodes = {1: (0, 0), 2: (0, DEG), 3: (DEG, 0)}
    ways = [{"nodes": [1, 2], "tags": {"highway": "residential"}},
            {"nodes": [2, 3], "tags": {"highway": "primary"}},
            {"nodes": [3, 1], "tags": {"highway": "residential"}}]
    city, _ = build(tmp_path, nodes, ways)
    payload = render_payload(city)
    assert len(payload["geoms"]) == 3  # 6 arestas, 3 ruas
    assert sorted(g["c"] for g in payload["geoms"]) == [1, 5, 5]
    assert all(len(g["p"]) % 2 == 0 for g in payload["geoms"])
    json.dumps(payload)  # serializável


def test_urban_focus_finds_the_dense_cluster_not_the_median():
    rng = np.random.default_rng(0)
    cluster = rng.normal([10_000, -3_000], 300, size=(400, 2))
    sparse = rng.uniform([-30_000, -30_000], [30_000, 30_000], size=(700, 2))
    center, radius = urban_focus(np.vstack([cluster, sparse]))
    assert np.hypot(*(center - [10_000, -3_000])) < 400
    assert 300 < radius < 20_000


REAL = Path(__file__).resolve().parents[1] / "data" / "raw" / "itapipoca-roads.json"


@pytest.mark.skipif(not REAL.exists(), reason="rode `delivery-lab city fetch` para baixar as ruas")
def test_real_city_is_connected_and_sane():
    city, report = build_city(read_osm(REAL))
    assert city.n_nodes > 5_000
    assert report.nodes_after / report.nodes_before > 0.95
    assert (city.edge_len > 0).all()
    # toda aresta sai e chega em nós que existem, e todo nó tem saída
    assert city.edge_src.max() < city.n_nodes and city.edge_dst.max() < city.n_nodes
    ptr, _ = city.adjacency
    assert (np.diff(ptr) > 0).all()
    # velocidades dentro do intervalo de sanidade
    kmh = city.edge_speed * 3.6
    assert kmh.min() >= 7 - 1e-3 and kmh.max() <= 80 + 1e-3
