"""Restaurantes e demanda sintéticos sobre o grafo da cidade. Tudo sai de uma semente.

O que é escondido do modelo (e só o simulador e o mapa "visão de deus" enxergam): o perfil de preparo
de cada restaurante. O que o modelo poderá observar: os horários de criado e pronto dos pedidos.

Distâncias aqui são em linha reta; o caminho pelas ruas entra no simulador (Passo 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .city import City

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class Kind:
    name: str
    share: float  # fração dos restaurantes cadastrados
    prep_mean_min: float  # preparo médio fora do pico
    prep_sigma: float  # dispersão (lognormal) do preparo de um pedido
    rush_gain: float  # quanto o preparo cresce no pico de demanda (1,5 = +50%)
    window: tuple[float, float]  # janela online, em horas; passar de 24 vira madrugada


# Tipos genéricos de estabelecimento; nenhum restaurante real.
KINDS: tuple[Kind, ...] = (
    Kind("marmita", 0.18, 10.0, 0.25, 1.3, (10.5, 14.5)),
    Kind("restaurante", 0.20, 18.0, 0.25, 1.4, (11.0, 15.0)),
    Kind("lanchonete", 0.22, 14.0, 0.30, 1.5, (16.0, 23.0)),
    Kind("pizzaria", 0.15, 28.0, 0.25, 1.6, (18.0, 23.5)),
    Kind("açaí e sorvete", 0.10, 5.0, 0.30, 1.2, (14.0, 22.5)),
    Kind("espetinho", 0.10, 20.0, 0.30, 1.5, (18.0, 25.0)),
    Kind("dia todo", 0.05, 16.0, 0.30, 1.4, (10.0, 22.0)),
)

# `restaurants` = cadastrados; `active_share` = fração que de fato atua; só os ativos dentro da janela
# ficam online. O pico de online é medido e vai no README, não é um alvo.
PRESETS = {
    "dev": {"restaurants": 40, "orders_per_day": 900, "active_share": 1.0},
    "plataforma": {"restaurants": 600, "orders_per_day": 5500, "active_share": 0.6},
}

# Peso do cruzamento para abrir restaurante, conforme a melhor via que passa por ele.
ROAD_BONUS = {0: 1.0, 1: 3.0, 2: 3.0, 3: 2.0, 4: 0.3, 5: 1.0}  # índice = classe em city.CLASSES
DECAY_M = 2000.0  # o cliente prefere restaurante perto: peso exp(-distância / DECAY_M)
DENSITY_CELL_M = 250.0


@dataclass(frozen=True)
class World:
    restaurant_node: np.ndarray  # (R,) int32, cruzamento do restaurante
    restaurant_xy: np.ndarray  # (R, 2) metros
    restaurant_kind: np.ndarray  # (R,) uint8, posição em KINDS
    active: np.ndarray  # (R,) bool, cadastrado e atuando (os outros nunca ficam online)
    popularity: np.ndarray  # (R,) float, peso relativo de ser escolhido
    open_from: np.ndarray  # (R,) horas
    open_until: np.ndarray  # (R,) horas (pode passar de 24)
    prep_mean_min: np.ndarray  # (R,) preparo médio fora do pico
    prep_sigma: np.ndarray  # (R,) dispersão lognormal
    rush_gain: np.ndarray  # (R,)
    node_weight: np.ndarray  # (N,) probabilidade de o cliente morar neste cruzamento (soma 1)
    minute_curve: np.ndarray  # (1440,) probabilidade de um pedido cair neste minuto (soma 1)
    orders_per_day: float
    max_radius_m: float
    seed: int

    @property
    def n_restaurants(self) -> int:
        return len(self.restaurant_node)

    def open_matrix(self, hours: np.ndarray) -> np.ndarray:
        """(n, R): o restaurante r está online na hora `hours[i]` (0 a 24)?"""
        h = np.asarray(hours, dtype=float)[:, None]
        today = (self.open_from[None, :] <= h) & (h < self.open_until[None, :])
        after_midnight = (self.open_until[None, :] > 24.0) & (h + 24.0 < self.open_until[None, :])
        return (today | after_midnight) & self.active[None, :]

    def is_open(self, hour: float) -> np.ndarray:
        """Quais restaurantes estão online na hora `hour` (0 a 24)."""
        return self.open_matrix(np.array([hour]))[0]

    def prep_minutes(self, hour: float) -> np.ndarray:
        """Preparo médio de cada restaurante na hora `hour`: cresce com a demanda do momento."""
        minute = int(hour * 60) % MINUTES_PER_DAY
        load = self.minute_curve[minute] / self.minute_curve.max()
        return self.prep_mean_min * (1.0 + (self.rush_gain - 1.0) * load)


@dataclass(frozen=True)
class Orders:
    """Os pedidos de um dia, em ordem de criação."""

    t_s: np.ndarray  # (n,) segundos desde a meia-noite
    restaurant: np.ndarray  # (n,) int32
    customer_node: np.ndarray  # (n,) int32
    customer_xy: np.ndarray  # (n, 2)
    dropped: int  # pedidos que não acharam restaurante aberto dentro do raio

    def __len__(self) -> int:
        return len(self.t_s)


def _bump(hour: np.ndarray, center: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((hour - center) / sigma) ** 2)


def demand_curve() -> np.ndarray:
    """Probabilidade de um pedido cair em cada minuto do dia: almoço, lanche, jantar e noite."""
    hour = np.arange(MINUTES_PER_DAY) / 60.0
    curve = 0.04 + _bump(hour, 12.5, 1.0, 1.0) + _bump(hour, 16.5, 1.2, 0.35)
    curve += _bump(hour, 19.8, 1.4, 1.3) + _bump(hour, 22.0, 1.0, 0.25)
    curve += _bump(hour + 24.0, 22.0, 1.0, 0.25)  # a cauda da noite passa da meia-noite
    return curve / curve.sum()


def node_density(city: City) -> np.ndarray:
    """Cruzamentos por vizinhança (grade de 250 m, janela 5x5), um valor por cruzamento."""
    cell = DENSITY_CELL_M
    ij = np.floor((city.node_xy - city.node_xy.min(axis=0)) / cell).astype(int)
    grid = np.zeros(ij.max(axis=0) + 1)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1.0)
    padded = np.pad(grid, 2)
    smooth = sum(
        padded[2 + dx : 2 + dx + grid.shape[0], 2 + dy : 2 + dy + grid.shape[1]]
        for dx in range(-2, 3)
        for dy in range(-2, 3)
    )
    return smooth[ij[:, 0], ij[:, 1]]


def _best_road_class(city: City) -> np.ndarray:
    best = np.full(city.n_nodes, 5, dtype=np.int64)
    np.minimum.at(best, city.edge_src, city.edge_cls)
    return best


def generate_world(
    city: City,
    seed: int = 7,
    restaurants: int = 600,
    orders_per_day: float = 5500.0,
    max_radius_m: float = 4000.0,
    active_share: float = 0.6,
) -> World:
    rng = np.random.default_rng([seed, 0])
    density = node_density(city)

    # restaurantes: cruzamentos do miolo urbano, mais perto do centro e das avenidas
    bonus = np.vectorize(ROAD_BONUS.get)(_best_road_class(city))
    eligible = density >= np.quantile(density, 0.6)
    weight = np.where(eligible, density**1.5 * bonus, 0.0)
    restaurants = min(restaurants, int(eligible.sum()))
    nodes = rng.choice(city.n_nodes, size=restaurants, replace=False, p=weight / weight.sum())

    kind = rng.choice(len(KINDS), size=restaurants, p=[k.share for k in KINDS])
    start = np.array([KINDS[k].window[0] for k in kind]) + rng.uniform(-0.5, 0.5, restaurants)
    end = np.array([KINDS[k].window[1] for k in kind]) + rng.uniform(-0.5, 0.5, restaurants)
    prep = np.array([KINDS[k].prep_mean_min for k in kind]) * rng.lognormal(0.0, 0.15, restaurants)

    return World(
        restaurant_node=nodes.astype(np.int32),
        restaurant_xy=city.node_xy[nodes],
        restaurant_kind=kind.astype(np.uint8),
        active=rng.random(restaurants) < active_share,
        popularity=rng.lognormal(0.0, 0.8, restaurants),
        open_from=start,
        open_until=end,
        prep_mean_min=prep,
        prep_sigma=np.array([KINDS[k].prep_sigma for k in kind]),
        rush_gain=np.array([KINDS[k].rush_gain for k in kind]),
        node_weight=density / density.sum(),
        minute_curve=demand_curve(),
        orders_per_day=float(orders_per_day),
        max_radius_m=float(max_radius_m),
        seed=seed,
    )


def sample_day(world: World, city: City, day: int) -> Orders:
    """Os pedidos do dia `day`. O mesmo (mundo, dia) devolve sempre o mesmo dia."""
    rng = np.random.default_rng([world.seed, 1, day])
    n = int(rng.poisson(world.orders_per_day))
    minute = rng.choice(MINUTES_PER_DAY, size=n, p=world.minute_curve)
    t_s = np.sort(minute * 60.0 + rng.uniform(0.0, 60.0, n))
    hour = t_s / 3600.0

    open_now = world.open_matrix(hour)  # open_now[i, r]: r está online quando o pedido i é feito

    customer = rng.choice(city.n_nodes, size=n, p=world.node_weight)
    chosen = np.full(n, -1, dtype=np.int64)
    pending = np.arange(n)
    for _ in range(8):  # quem não acha restaurante perto sorteia outro endereço
        if len(pending) == 0:
            break
        diff = city.node_xy[customer[pending]][:, None, :] - world.restaurant_xy[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        w = world.popularity[None, :] * np.exp(-dist / DECAY_M)
        w = np.where(open_now[pending] & (dist <= world.max_radius_m), w, 0.0)
        total = w.sum(axis=1)
        ok = total > 0
        pick = (np.cumsum(w[ok], axis=1) < (rng.random(ok.sum()) * total[ok])[:, None]).sum(axis=1)
        chosen[pending[ok]] = np.minimum(pick, world.n_restaurants - 1)
        pending = pending[~ok]
        customer[pending] = rng.choice(city.n_nodes, size=len(pending), p=world.node_weight)

    keep = chosen >= 0
    return Orders(
        t_s=t_s[keep],
        restaurant=chosen[keep].astype(np.int32),
        customer_node=customer[keep].astype(np.int32),
        customer_xy=city.node_xy[customer[keep]],
        dropped=int((~keep).sum()),
    )


def heat_by_hour(world: World, city: City, days: int, cell_m: float = 250.0) -> dict:
    """Pedidos por célula e por hora, em média de `days` dias simulados (dias 1 a `days`)."""
    origin = city.node_xy.min(axis=0)
    counts: dict[tuple[int, int], np.ndarray] = {}
    for day in range(1, days + 1):
        orders = sample_day(world, city, day)
        cell = np.floor((orders.customer_xy - origin) / cell_m).astype(int)
        hour = (orders.t_s // 3600).astype(int)
        for (ix, iy), h in zip(map(tuple, cell), hour):
            counts.setdefault((ix, iy), np.zeros(24))[h] += 1.0
    return {
        "cell_m": cell_m,
        "origin": [float(origin[0]), float(origin[1])],
        "days": days,
        "cells": [[int(ix), int(iy), np.round(v / days, 2).tolist()] for (ix, iy), v in sorted(counts.items())],
    }


def render_payload(world: World, city: City, days: int = 30, show_day: int = 1) -> dict:
    """JSON do mapa: restaurantes, calor médio por hora e os pedidos de um dia."""
    orders = sample_day(world, city, show_day)
    hours = np.arange(24) + 0.5
    online = world.open_matrix(hours).sum(axis=1)
    xy = np.round(world.restaurant_xy, 1)
    return {
        "scale": {
            "restaurants": world.n_restaurants,
            "active": int(world.active.sum()),
            "orders_per_day": world.orders_per_day,
            "max_radius_m": world.max_radius_m,
            "seed": world.seed,
        },
        "kinds": [k.name for k in KINDS],
        "restaurants": {
            "x": xy[:, 0].tolist(),
            "y": xy[:, 1].tolist(),
            "kind": world.restaurant_kind.tolist(),
            "active": world.active.astype(int).tolist(),
            "popularity": np.round(world.popularity, 2).tolist(),
            "open_from": np.round(world.open_from, 2).tolist(),
            "open_until": np.round(world.open_until, 2).tolist(),
            "prep_mean_min": np.round(world.prep_mean_min, 1).tolist(),
        },
        "curve_by_hour": np.round(world.minute_curve.reshape(24, 60).sum(axis=1), 5).tolist(),
        "online_by_hour": online.tolist(),
        "heat": heat_by_hour(world, city, days),
        "day": {
            "seed_day": show_day,
            "count": len(orders),
            "dropped": orders.dropped,
            "t_min": np.round(orders.t_s / 60.0, 1).tolist(),
            "x": np.round(orders.customer_xy[:, 0], 1).tolist(),
            "y": np.round(orders.customer_xy[:, 1], 1).tolist(),
            "restaurant": orders.restaurant.tolist(),
        },
    }
