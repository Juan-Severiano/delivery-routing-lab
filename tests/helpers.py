"""Funções de apoio para montar mundos pequenos à mão nos testes do simulador e dos planejadores."""

import numpy as np

from delivery_lab import sim as S
from delivery_lab import world as W
from delivery_lab.traffic import build_traffic


def tiny_orders(city, restaurants, customers, times):
    """Pedidos montados à mão: restaurante, cruzamento do cliente e hora de criação (s)."""
    customers = np.asarray(customers)
    return W.Orders(
        t_s=np.asarray(times, dtype=float),
        restaurant=np.asarray(restaurants, dtype=np.int32),
        customer_node=customers.astype(np.int32),
        customer_xy=city.node_xy[customers],
        dropped=0,
    )


def always_on_fleet(nodes):
    n = len(nodes)
    return S.Fleet(np.zeros(n), np.full(n, 40 * 3600.0), np.asarray(nodes, dtype=np.int32))


def make_sim(city, world, paths, fleet, orders, policy, day=1, record=True, config=None):
    traffic = build_traffic(city, world.seed, day)
    return S.Sim(city, world, paths, traffic, fleet, orders, policy, day, config, record)
