"""Caminhos mínimos entre todos os pares de cruzamentos, pelo tempo livre das ruas.

O caminho que o entregador segue é o de menor tempo livre (fixo, portanto cacheável). Quanto ele
realmente demora depende do trânsito escondido (`traffic.py`) sobre as arestas desse caminho.

Formato em disco (uma linha por origem): `dist.f32` (segundos, float32) e `pred.i16` (id da aresta que
entra em cada destino no caminho mínimo, int16; -1 na própria origem). Quem escreve pode ser este
módulo (referência em Python) ou o binário em Go (`go/apsp`): os dois geram os mesmos arquivos.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np

from .city import City

INF = float("inf")


def free_flow_seconds(city: City) -> np.ndarray:
    """Tempo de cada aresta com a rua livre: comprimento / velocidade livre."""
    return city.edge_len.astype(np.float64) / city.edge_speed.astype(np.float64)


def adjacency_lists(city: City, weight: np.ndarray) -> list[list[tuple[int, float, int]]]:
    adj: list[list[tuple[int, float, int]]] = [[] for _ in range(city.n_nodes)]
    for e, (u, v, w) in enumerate(zip(city.edge_src.tolist(), city.edge_dst.tolist(), weight.tolist())):
        adj[u].append((v, w, e))
    return adj


def dijkstra_tree(
    adj: list[list[tuple[int, float, int]]], source: int
) -> tuple[np.ndarray, np.ndarray]:
    """Árvore de caminhos mínimos a partir de `source`: (distâncias, aresta que entra em cada nó)."""
    n = len(adj)
    dist = [INF] * n
    pred = [-1] * n
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w, e in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = e
                heapq.heappush(heap, (nd, v))
    return np.array(dist, dtype=np.float32), np.array(pred, dtype=np.int16)


def apsp_python(city: City, sources: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Referência em Python puro: uma árvore por origem. Devolve (distâncias, predecessoras) por linha."""
    if city.n_edges > np.iinfo(np.int16).max:
        raise ValueError("mais de 32767 arestas: pred.i16 não comporta os ids")
    adj = adjacency_lists(city, free_flow_seconds(city))
    rows = np.arange(city.n_nodes) if sources is None else np.asarray(sources)
    dist = np.empty((len(rows), city.n_nodes), dtype=np.float32)
    pred = np.empty((len(rows), city.n_nodes), dtype=np.int16)
    for i, s in enumerate(rows):
        dist[i], pred[i] = dijkstra_tree(adj, int(s))
    return dist, pred


class Paths:
    """Consulta de tempo livre e de caminho entre dois cruzamentos, sobre os arquivos do APSP."""

    def __init__(self, city: City, dist: np.ndarray, pred: np.ndarray):
        self.city = city
        self.dist = dist
        self.pred = pred
        self._src = city.edge_src

    @staticmethod
    def files(directory: Path) -> tuple[Path, Path]:
        return Path(directory) / "dist.f32", Path(directory) / "pred.i16"

    @classmethod
    def load(cls, city: City, directory: Path) -> Paths:
        dist_file, pred_file = cls.files(directory)
        n = city.n_nodes
        for f, item in ((dist_file, 4), (pred_file, 2)):
            if not f.exists() or f.stat().st_size != n * n * item:
                raise FileNotFoundError(f"{f} ausente ou do tamanho errado; rode `delivery-lab paths build`")
        dist = np.memmap(dist_file, dtype=np.float32, mode="r", shape=(n, n))
        pred = np.memmap(pred_file, dtype=np.int16, mode="r", shape=(n, n))
        return cls(city, dist, pred)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dist_file, pred_file = self.files(directory)
        np.asarray(self.dist, dtype=np.float32).tofile(dist_file)
        np.asarray(self.pred, dtype=np.int16).tofile(pred_file)

    def seconds(self, u: int, v: int) -> float:
        """Tempo livre de u até v."""
        return float(self.dist[u, v])

    def edges(self, u: int, v: int) -> list[int]:
        """Ids das arestas do caminho de u até v, na ordem em que são percorridas."""
        row = self.pred[u]
        out: list[int] = []
        node = v
        while node != u:
            e = int(row[node])
            if e < 0:
                raise ValueError(f"sem caminho de {u} até {v}")
            out.append(e)
            node = int(self._src[e])
        out.reverse()
        return out
