"""Replay de um dia simulado para o mapa: tudo em colunas, para o JSON ficar pequeno.

Trechos (`legs`) ficam um depois do outro, ordenados por entregador e por hora de saída; as arestas e
os tempos de todos os trechos vão em dois vetores planos, indexados por `leg_ptr`. O mapa reconstrói a
posição de cada entregador em qualquer instante andando pelas arestas.
"""

from __future__ import annotations

import numpy as np

from .city import City
from .sim import SimResult, summarize
from .world import KINDS, World


def render_replay(result: SimResult, city: City, world: World, policy: str, day: int) -> dict:
    if result.legs is None:
        raise ValueError("rode a simulação com record=True para ter os trechos")
    n_couriers = len(result.fleet)

    leg_t0, leg_kind, leg_order, leg_ptr = [], [], [], [0]
    flat_edges: list[int] = []
    flat_ds: list[int] = []
    courier_first = [0]
    for legs in result.legs:
        for leg in legs:
            leg_t0.append(round(leg.t0, 1))
            leg_kind.append(leg.kind)
            leg_order.append(leg.order)
            flat_edges.extend(leg.edges.tolist())
            flat_ds.extend(np.round(leg.seconds * 10.0).astype(int).tolist())
            leg_ptr.append(len(flat_edges))
        courier_first.append(len(leg_t0))

    start_xy = np.round(city.node_xy[result.fleet.start_node], 1)
    o = result.orders
    r = o.restaurant
    return {
        "meta": {
            "policy": policy,
            "day": day,
            "seed": world.seed,
            "rain": result.traffic.rain,
            "day_factor": round(result.traffic.day_factor, 3),
            "couriers": n_couriers,
            "orders": len(o),
            "capacity": result.config.capacity,
            "summary": {k: round(v, 2) for k, v in summarize(result).items()},
        },
        "edges": {"geom": city.edge_geom.tolist(), "rev": city.edge_rev.astype(int).tolist()},
        "couriers": {
            "shift_start": np.round(result.fleet.shift_start_s, 1).tolist(),
            "shift_end": np.round(result.fleet.shift_end_s, 1).tolist(),
            "x": start_xy[:, 0].tolist(),
            "y": start_xy[:, 1].tolist(),
            "first_leg": courier_first,
        },
        "legs": {"t0": leg_t0, "kind": leg_kind, "order": leg_order, "ptr": leg_ptr},
        "leg_edges": flat_edges,
        "leg_deciseconds": flat_ds,
        "orders": {
            "x": np.round(o.customer_xy[:, 0], 1).tolist(),
            "y": np.round(o.customer_xy[:, 1], 1).tolist(),
            "restaurant": r.tolist(),
            "kind": world.restaurant_kind[r].tolist(),
            "created": np.round(result.t_created, 1).tolist(),
            "assigned": _nullable(result.t_assigned),
            "ready": np.round(result.t_ready, 1).tolist(),
            "picked": _nullable(result.t_picked),
            "delivered": _nullable(result.t_delivered),
            "promised": np.round(result.promised, 1).tolist(),
            "courier": result.courier.tolist(),
            "rejected": result.rejected.astype(int).tolist(),
        },
        "kinds": [k.name for k in KINDS],
    }


def _nullable(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else round(float(v), 1) for v in values]
