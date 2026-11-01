"""Trânsito escondido: quanto cada aresta realmente demora, hora a hora.

É a verdade do simulador. Os baselines clássicos planejam com o tempo livre; só o teto (B2) e, depois
de aprender, os modelos chegam perto disto. Toda a forma abaixo é suposição do laboratório.

    tempo_real(aresta, hora) = tempo_livre x fator_dia x (1 + A x pico(hora) x centralidade x classe x chuva) x ruído_de_zona
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .city import City, urban_focus
from .paths import free_flow_seconds

AMPLITUDE = 1.0  # no pico, no centro, numa avenida: o tempo dobra
CENTRALITY_M = 2500.0  # a lentidão cai com exp(-distância ao centro / 2,5 km)
CLASS_SENSITIVITY = np.array([0.9, 1.0, 1.0, 0.7, 0.1, 0.4])  # trunk..residential (ordem de city.CLASSES)
RAIN_PROBABILITY = 0.2
RAIN_SLOWDOWN = 0.25  # com chuva, o excesso de lentidão cresce 25%
ZONE_M = 1000.0
ZONE_NOISE_SIGMA = 0.06
TRIP_NOISE_SIGMA = 0.08  # ruído de cada viagem, aplicado pelo simulador


def _bump(hour: np.ndarray, center: float, sigma: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((hour - center) / sigma) ** 2)


def rush(hour: np.ndarray) -> np.ndarray:
    """Intensidade do pico de trânsito (0 a ~1): manhã fraca, almoço e fim de tarde fortes."""
    return np.clip(
        _bump(hour, 7.5, 0.7, 0.4) + _bump(hour, 12.6, 0.9, 0.8) + _bump(hour, 18.3, 1.2, 1.0), 0.0, 1.0
    )


@dataclass(frozen=True)
class Traffic:
    edge_seconds: np.ndarray  # (24, m) float32: tempo real de cada aresta em cada hora, sem ruído de viagem
    rain: bool
    day_factor: float
    seed: int
    day: int

    def leg_seconds(self, edges: list[int], hour: int) -> float:
        return float(self.edge_seconds[hour % 24, edges].sum())


def edge_centrality(city: City) -> np.ndarray:
    center, _ = urban_focus(city.node_xy)
    mid = (city.node_xy[city.edge_src] + city.node_xy[city.edge_dst]) / 2.0
    return np.exp(-np.hypot(*(mid - center).T) / CENTRALITY_M)


def build_traffic(city: City, seed: int, day: int) -> Traffic:
    """O trânsito do dia `day`. O mesmo (cidade, semente, dia) devolve sempre o mesmo trânsito."""
    rng = np.random.default_rng([seed, 2, day])
    rain = bool(rng.random() < RAIN_PROBABILITY)
    day_factor = float(rng.lognormal(0.0, 0.04))

    hours = np.arange(24) + 0.5
    slow = AMPLITUDE * rush(hours)[:, None] * edge_centrality(city)[None, :]
    slow = slow * CLASS_SENSITIVITY[city.edge_cls][None, :] * (1.0 + (RAIN_SLOWDOWN if rain else 0.0))

    # ruído por zona de 1 km e por hora, correlacionado no tempo (AR(1))
    mid = (city.node_xy[city.edge_src] + city.node_xy[city.edge_dst]) / 2.0
    cell = np.floor((mid - mid.min(axis=0)) / ZONE_M).astype(int)
    zone = cell[:, 0] * (cell[:, 1].max() + 1) + cell[:, 1]
    _, zone = np.unique(zone, return_inverse=True)
    eps = np.zeros((24, zone.max() + 1))
    eps[0] = rng.normal(0.0, ZONE_NOISE_SIGMA, eps.shape[1])
    for h in range(1, 24):
        eps[h] = 0.7 * eps[h - 1] + rng.normal(0.0, ZONE_NOISE_SIGMA * np.sqrt(1 - 0.49), eps.shape[1])
    zone_factor = np.exp(eps)[:, zone]

    factor = (1.0 + slow) * zone_factor * day_factor
    seconds = free_flow_seconds(city)[None, :] * factor
    return Traffic(seconds.astype(np.float32), rain, day_factor, seed, day)
