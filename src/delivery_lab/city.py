"""Grafo de ruas: cruzamentos viram nós; trechos viram arestas com geometria e velocidade livre.

Tudo em metros num plano local (projeção equiretangular em torno do centro do recorte; para uma
cidade de dezenas de km o erro fica abaixo de 0,01%). O grafo só mantém a maior componente
fortemente conectada, para que qualquer par de nós tenha caminho nos dois sentidos.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .osm import DRIVABLE, RawOsm

EARTH_RADIUS_M = 6_371_008.8
CLASSES = DRIVABLE  # o id da classe é a posição nesta tupla

# Velocidade livre (km/h) quando a via não tem `maxspeed`. É uma suposição do laboratório para moto
# numa cidade do interior, não uma medição.
DEFAULT_SPEED_KMH = {
    "trunk": 60.0,
    "primary": 50.0,
    "secondary": 40.0,
    "tertiary": 35.0,
    "unclassified": 30.0,
    "residential": 25.0,
}
LINK_FACTOR = 0.7  # rampas e alças (`*_link`) são mais lentas que a via principal
MIN_SPEED_KMH, MAX_SPEED_KMH = 10.0, 80.0


@dataclass(frozen=True)
class BuildReport:
    ways: int
    chains: int
    self_loops_dropped: int
    nodes_before: int
    edges_before: int
    nodes_after: int
    edges_after: int
    km_before: float  # extensão contando cada rua uma vez, mesmo de mão dupla
    km_after: float

    def lines(self) -> list[str]:
        return [
            f"vias lidas: {self.ways}, trechos entre cruzamentos: {self.chains}"
            f" (laços descartados: {self.self_loops_dropped})",
            f"antes da componente conectada: {self.nodes_before} nós, {self.edges_before} arestas,"
            f" {self.km_before:.1f} km",
            f"depois: {self.nodes_after} nós, {self.edges_after} arestas, {self.km_after:.1f} km",
        ]


@dataclass(frozen=True)
class City:
    origin: tuple[float, float]  # (lat, lon) do ponto (0, 0) do plano
    node_xy: np.ndarray  # (n, 2) metros; x para leste, y para norte
    node_signal: np.ndarray  # (n,) bool: cruzamento com semáforo
    edge_src: np.ndarray  # (m,) int32
    edge_dst: np.ndarray  # (m,) int32
    edge_len: np.ndarray  # (m,) float32 metros
    edge_speed: np.ndarray  # (m,) float32 m/s, velocidade livre
    edge_cls: np.ndarray  # (m,) uint8, posição em CLASSES
    edge_geom: np.ndarray  # (m,) int32, índice da geometria (mão dupla divide a mesma)
    edge_rev: np.ndarray  # (m,) bool, True se a aresta percorre a geometria de trás para frente
    geom_ptr: np.ndarray  # (g + 1,) int64
    geom_pts: np.ndarray  # (p, 2) metros

    @property
    def n_nodes(self) -> int:
        return len(self.node_xy)

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    @property
    def n_geoms(self) -> int:
        return len(self.geom_ptr) - 1

    def geometry(self, geom: int) -> np.ndarray:
        return self.geom_pts[self.geom_ptr[geom] : self.geom_ptr[geom + 1]]

    def edge_points(self, edge: int) -> np.ndarray:
        """Pontos da aresta no sentido em que ela é percorrida."""
        pts = self.geometry(int(self.edge_geom[edge]))
        return pts[::-1] if self.edge_rev[edge] else pts

    @cached_property
    def adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        """Lista de adjacência em CSR: `edges[ptr[v]:ptr[v + 1]]` são as arestas que saem de v."""
        order = np.argsort(self.edge_src, kind="stable")
        counts = np.bincount(self.edge_src, minlength=self.n_nodes)
        ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return ptr, order.astype(np.int32)

    @cached_property
    def geom_class(self) -> np.ndarray:
        """Classe de cada geometria (as duas direções de uma rua têm a mesma)."""
        out = np.zeros(self.n_geoms, dtype=np.uint8)
        out[self.edge_geom] = self.edge_cls
        return out

    def km_by_class(self) -> dict[str, float]:
        """Extensão por classe, contando cada rua uma vez."""
        seg = np.diff(self.geom_pts, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        lengths = np.zeros(self.n_geoms)
        for g in range(self.n_geoms):
            lengths[g] = seg_len[self.geom_ptr[g] : self.geom_ptr[g + 1] - 1].sum()
        return {
            name: float(lengths[self.geom_class == i].sum() / 1000.0)
            for i, name in enumerate(CLASSES)
        }

    def to_latlon(self, xy: np.ndarray) -> np.ndarray:
        lat0, lon0 = self.origin
        lat = lat0 + np.degrees(xy[..., 1] / EARTH_RADIUS_M)
        lon = lon0 + np.degrees(xy[..., 0] / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
        return np.stack([lat, lon], axis=-1)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            origin=np.array(self.origin),
            node_xy=self.node_xy,
            node_signal=self.node_signal,
            edge_src=self.edge_src,
            edge_dst=self.edge_dst,
            edge_len=self.edge_len,
            edge_speed=self.edge_speed,
            edge_cls=self.edge_cls,
            edge_geom=self.edge_geom,
            edge_rev=self.edge_rev,
            geom_ptr=self.geom_ptr,
            geom_pts=self.geom_pts,
        )

    @classmethod
    def load(cls, path: Path) -> City:
        with np.load(Path(path)) as data:
            fields = {k: data[k] for k in data.files}
        origin = fields.pop("origin")
        return cls(origin=(float(origin[0]), float(origin[1])), **fields)


def project(lat: np.ndarray, lon: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    lat0, lon0 = origin
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.stack([x, y], axis=-1)


def free_flow_kmh(highway: str, link: bool, maxspeed_kmh: float | None) -> float:
    if maxspeed_kmh is not None:
        return min(max(maxspeed_kmh, MIN_SPEED_KMH), MAX_SPEED_KMH)
    speed = DEFAULT_SPEED_KMH[highway]
    return speed * LINK_FACTOR if link else speed


def largest_scc(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Máscara dos nós da maior componente fortemente conectada (Tarjan iterativo)."""
    order = np.argsort(src, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))]).tolist()
    adj = dst[order].tolist()
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    comp = [-1] * n
    stack: list[int] = []
    counter = 0
    n_comp = 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, ptr[root])]
        while work:
            v, i = work[-1]
            if i < ptr[v + 1]:
                work[-1] = (v, i + 1)
                w = adj[i]
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, ptr[w]))
                elif on_stack[w]:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = n_comp
                    if w == v:
                        break
                n_comp += 1
    comp_arr = np.array(comp)
    return comp_arr == np.bincount(comp_arr).argmax()


def build_city(raw: RawOsm) -> tuple[City, BuildReport]:
    uses = Counter(n for way in raw.ways for n in way.nodes)
    osm_ids = np.fromiter(uses.keys(), dtype=np.int64)
    lat = np.array([raw.coords[i][0] for i in osm_ids])
    lon = np.array([raw.coords[i][1] for i in osm_ids])
    origin = ((lat.min() + lat.max()) / 2.0, (lon.min() + lon.max()) / 2.0)
    xy = project(lat, lon, origin)
    slot = {int(osm): k for k, osm in enumerate(osm_ids)}

    src: list[int] = []
    dst: list[int] = []
    cls: list[int] = []
    speed: list[float] = []
    geom_of: list[int] = []
    rev: list[bool] = []
    geoms: list[np.ndarray] = []
    n_chains = 0
    self_loops = 0

    for way in raw.ways:
        slots = [slot[n] for n in way.nodes]
        start = 0
        for j in range(1, len(slots)):
            # cruzamento = nó usado por mais de uma via, ou ponta da via
            if uses[way.nodes[j]] < 2 and j != len(slots) - 1:
                continue
            chunk = slots[start : j + 1]
            start = j
            n_chains += 1
            if chunk[0] == chunk[-1]:
                self_loops += 1
                continue
            pts = xy[chunk]
            geoms.append(pts)
            g = len(geoms) - 1
            mps = free_flow_kmh(way.highway, way.link, way.maxspeed_kmh) / 3.6
            for forward in (True, False):
                if (forward and way.oneway == -1) or (not forward and way.oneway == 1):
                    continue
                src.append(chunk[0] if forward else chunk[-1])
                dst.append(chunk[-1] if forward else chunk[0])
                cls.append(CLASSES.index(way.highway))
                speed.append(mps)
                geom_of.append(g)
                rev.append(not forward)

    return _finish(
        xy=xy,
        osm_ids=osm_ids,
        signals=raw.signals,
        origin=origin,
        src=np.array(src, dtype=np.int64),
        dst=np.array(dst, dtype=np.int64),
        cls=np.array(cls, dtype=np.uint8),
        speed=np.array(speed, dtype=np.float32),
        geom_of=np.array(geom_of, dtype=np.int64),
        rev=np.array(rev, dtype=bool),
        geoms=geoms,
        report_head=(len(raw.ways), n_chains, self_loops),
    )


def _polyline_length(pts: np.ndarray) -> float:
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _finish(
    *, xy, osm_ids, signals, origin, src, dst, cls, speed, geom_of, rev, geoms, report_head
) -> tuple[City, BuildReport]:
    geom_len = np.array([_polyline_length(p) for p in geoms])
    nodes_used = np.unique(np.concatenate([src, dst]))
    remap = np.full(len(xy), -1, dtype=np.int64)
    remap[nodes_used] = np.arange(len(nodes_used))
    src_c, dst_c = remap[src], remap[dst]
    keep_node = largest_scc(len(nodes_used), src_c, dst_c)
    keep_edge = keep_node[src_c] & keep_node[dst_c]

    # reindexa nós e geometrias que sobraram
    node_new = np.cumsum(keep_node) - 1
    kept_geoms = np.unique(geom_of[keep_edge])
    geom_new = np.full(len(geoms), -1, dtype=np.int64)
    geom_new[kept_geoms] = np.arange(len(kept_geoms))
    node_slots = nodes_used[keep_node]

    pieces = [geoms[g] for g in kept_geoms]
    ptr = np.concatenate([[0], np.cumsum([len(p) for p in pieces])]).astype(np.int64)
    pts = np.concatenate(pieces) if pieces else np.zeros((0, 2))
    edge_geom = geom_new[geom_of[keep_edge]]
    edge_len = geom_len[geom_of[keep_edge]].astype(np.float32)

    city = City(
        origin=(float(origin[0]), float(origin[1])),
        node_xy=xy[node_slots],
        node_signal=np.isin(osm_ids[node_slots], np.fromiter(signals, dtype=np.int64)),
        edge_src=node_new[src_c[keep_edge]].astype(np.int32),
        edge_dst=node_new[dst_c[keep_edge]].astype(np.int32),
        edge_len=edge_len,
        edge_speed=speed[keep_edge],
        edge_cls=cls[keep_edge],
        edge_geom=edge_geom.astype(np.int32),
        edge_rev=rev[keep_edge],
        geom_ptr=ptr,
        geom_pts=pts,
    )
    used_before = np.unique(geom_of)
    report = BuildReport(
        ways=report_head[0],
        chains=report_head[1],
        self_loops_dropped=report_head[2],
        nodes_before=len(nodes_used),
        edges_before=len(src),
        nodes_after=city.n_nodes,
        edges_after=city.n_edges,
        km_before=float(geom_len[used_before].sum() / 1000.0),
        km_after=float(geom_len[kept_geoms].sum() / 1000.0),
    )
    return city, report


def urban_focus(node_xy: np.ndarray, cell: float = 500.0, share: float = 0.25) -> tuple[np.ndarray, float]:
    """Centro e raio do miolo urbano: o pico de densidade de cruzamentos, não a mediana.

    A mediana é puxada pelas estradas rurais (o município é bem maior que a área urbana). O raio é
    o que contém a fração `share` dos cruzamentos mais próximos do centro.
    """
    cells, counts = np.unique(np.floor(node_xy / cell).astype(int), axis=0, return_counts=True)
    center = cells[counts.argmax()] * cell + cell / 2
    for _ in range(5):  # refina pela média dos cruzamentos num raio de 3 células
        near = np.hypot(*(node_xy - center).T) < 3 * cell
        center = node_xy[near].mean(axis=0)
    radius = float(np.quantile(np.hypot(*(node_xy - center).T), share))
    return center, radius


def render_payload(city: City) -> dict:
    """JSON para o mapa: cada rua uma vez, polilinhas em metros com uma casa decimal."""
    center, radius = urban_focus(city.node_xy)
    lo, hi = city.geom_pts.min(axis=0), city.geom_pts.max(axis=0)
    return {
        "origin": list(city.origin),
        "classes": list(CLASSES),
        "bounds": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        "focus": {"x": float(center[0]), "y": float(center[1]), "radius": radius},
        "stats": {
            "nodes": city.n_nodes,
            "edges": city.n_edges,
            "signals": int(city.node_signal.sum()),
            "km_by_class": {k: round(v, 1) for k, v in city.km_by_class().items()},
        },
        "geoms": [
            {"c": int(city.geom_class[g]), "p": np.round(city.geometry(g), 1).ravel().tolist()}
            for g in range(city.n_geoms)
        ],
    }
