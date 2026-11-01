"""Modelo de preparo: quanto um restaurante demora, por hora, como distribuição.

A rede recebe o restaurante (embedding), o tipo, a hora do dia e se chove, e devolve a média e o desvio
do **log** do preparo em segundos, treinada com a perda gaussiana (o preparo do simulador é lognormal,
mas o modelo não sabe disso). Com a distribuição o planejador pode escolher um quantil, e dá para medir
se as probabilidades estão calibradas.

O que o modelo vê é o que uma plataforma tem na hora em que o pedido nasce: o restaurante, a hora e a
previsão do tempo. O rótulo é o preparo observado depois. Suposição: o rótulo é limpo (a plataforma
saberia exatamente quando o pedido ficou pronto); na vida real ele seria censurado pela chegada do
entregador.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ..city import City
from ..sim import sample_prep
from ..traffic import rain_of_day
from ..world import KINDS, World, sample_day
from .fit import Data, fit, gaussian_nll
from .nn import Net, NumpyNet

HARMONICS = 3
N_KINDS = len(KINDS)
N_FEATURES = 2 * HARMONICS + N_KINDS + 1
ID_DIM = 8
HIDDEN = (48, 48)
RULE_OF_THUMB_S = 700.0  # a regra de bolso do B1


def time_features(t_s: np.ndarray) -> np.ndarray:
    angle = 2.0 * math.pi * (np.asarray(t_s, dtype=float) % 86400.0) / 86400.0
    cols = [f(k * angle) for k in range(1, HARMONICS + 1) for f in (np.sin, np.cos)]
    return np.stack(cols, axis=1)


def features(kind: np.ndarray, t_s: np.ndarray, rain: np.ndarray) -> np.ndarray:
    onehot = np.eye(N_KINDS)[np.asarray(kind, dtype=int)]
    return np.concatenate([time_features(t_s), onehot, np.asarray(rain, dtype=float)[:, None]], axis=1).astype(np.float32)


@dataclass
class PrepSet:
    restaurant: np.ndarray  # (n,) int
    kind: np.ndarray
    t_s: np.ndarray
    rain: np.ndarray  # (n,) bool
    prep_s: np.ndarray  # preparo observado, segundos
    oracle_median_s: np.ndarray  # a verdade do simulador: mediana esperada
    oracle_sigma: np.ndarray  # a verdade do simulador: dispersão do log

    def __len__(self) -> int:
        return len(self.restaurant)


def build_prep_set(city: City, world: World, days: range) -> PrepSet:
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("restaurant", "t_s", "rain", "prep", "median")}
    for day in days:
        orders = sample_day(world, city, day)
        rain = rain_of_day(world.seed, day)
        rng = np.random.default_rng([world.seed, 3, day])  # o mesmo sorteio que o simulador faz
        mean_prep, prep = sample_prep(world, orders, rain, rng)
        cols["restaurant"].append(orders.restaurant)
        cols["t_s"].append(orders.t_s)
        cols["rain"].append(np.full(len(orders), rain))
        cols["prep"].append(prep)
        cols["median"].append(mean_prep * (1.1 if rain else 1.0))  # a mediana do lognormal(0, s) é 1
    restaurant = np.concatenate(cols["restaurant"])
    return PrepSet(
        restaurant=restaurant,
        kind=world.restaurant_kind[restaurant],
        t_s=np.concatenate(cols["t_s"]),
        rain=np.concatenate(cols["rain"]),
        prep_s=np.concatenate(cols["prep"]),
        oracle_median_s=np.concatenate(cols["median"]),
        oracle_sigma=world.prep_sigma[restaurant],
    )


class PrepModel:
    """Preparo aprendido, só com numpy na hora de usar."""

    def __init__(self, state: dict[str, np.ndarray], restaurant_kind: np.ndarray, y_mean: float, meta: dict | None = None):
        self.net = NumpyNet(state)
        self.state = state
        self.restaurant_kind = np.asarray(restaurant_kind, dtype=int)
        self.y_mean = float(y_mean)
        self.meta = meta or {}

    def predict(self, restaurant: np.ndarray, t_s: np.ndarray, rain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(média, desvio) do log do preparo em segundos."""
        restaurant = np.asarray(restaurant, dtype=np.int64)
        x = features(self.restaurant_kind[restaurant], t_s, rain)
        out = self.net(x, restaurant)
        return out[:, 0].astype(float) + self.y_mean, np.exp(np.clip(out[:, 1], -5.0, 2.0)).astype(float)

    def quantile_s(self, restaurant: np.ndarray, t_s: np.ndarray, rain: np.ndarray, q: float = 0.5) -> np.ndarray:
        mu, sigma = self.predict(restaurant, t_s, rain)
        return np.exp(mu + NormalDist().inv_cdf(q) * sigma)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            restaurant_kind=self.restaurant_kind,
            y_mean=np.array(self.y_mean),
            meta=np.array(json.dumps(self.meta)),
            **self.state,
        )

    @classmethod
    def load(cls, path: Path) -> PrepModel:
        with np.load(Path(path)) as data:
            files = {k: data[k] for k in data.files}
        meta = json.loads(str(files.pop("meta")))
        return cls(files, files.pop("restaurant_kind"), float(files.pop("y_mean")), meta)


def train_prep(
    world: World, train: PrepSet, val: PrepSet, epochs: int = 40, seed: int = 0, log=print
) -> tuple[PrepModel, object]:
    y_mean = float(np.log(train.prep_s).mean())

    def data(s: PrepSet) -> Data:
        return Data(
            x=features(s.kind, s.t_s, s.rain),
            y=(np.log(s.prep_s) - y_mean).astype(np.float32)[:, None],
            ids=s.restaurant.astype(np.int64),
        )

    net = Net(N_FEATURES, HIDDEN, 2, n_ids=world.n_restaurants, id_dim=ID_DIM, seed=seed)
    history = fit(net, gaussian_nll, data(train), data(val), epochs=epochs, batch=1024, lr=4e-3, lr_end=2e-4,
                  patience=8, seed=seed, log=log)
    meta = {"train_orders": len(train), "val_orders": len(val), "best_epoch": history.best_epoch,
            "hidden": list(HIDDEN), "id_dim": ID_DIM, "seed": seed}
    return PrepModel(net.state(), world.restaurant_kind, y_mean, meta), history


def evaluate_prep(model: PrepModel, test: PrepSet, train: PrepSet) -> dict:
    """Métricas na hora de teste: erro em minutos contra baselines e calibração das probabilidades."""
    y = test.prep_s
    mu, sigma = model.predict(test.restaurant, test.t_s, test.rain)
    mae = lambda pred: float(np.mean(np.abs(y - pred)) / 60.0)  # noqa: E731

    # tabela tipo x hora: a média do treino em cada célula (o baseline "sem aprendizado de rede")
    hour_tr = (train.t_s // 3600).astype(int) % 24
    hour_te = (test.t_s // 3600).astype(int) % 24
    table = np.full((N_KINDS, 24), float(train.prep_s.mean()))
    for k in range(N_KINDS):
        for h in range(24):
            m = (train.kind == k) & (hour_tr == h)
            if m.any():
                table[k, h] = np.median(train.prep_s[m])

    nd = NormalDist()
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    coverage = [float(np.mean(np.log(y) <= mu + nd.inv_cdf(q) * sigma)) for q in quantiles]
    # a verdade: o que um modelo perfeito saberia (mediana e dispersão do simulador)
    oracle_mu, oracle_sigma = np.log(test.oracle_median_s), test.oracle_sigma
    nll = lambda m, s: float(np.mean(np.log(s) + 0.5 * ((np.log(y) - m) / s) ** 2))  # noqa: E731
    return {
        "orders": len(test),
        "mae_min": {
            "regra_de_bolso_700s": mae(RULE_OF_THUMB_S),
            "media_global": mae(float(train.prep_s.mean())),
            "tabela_tipo_x_hora": mae(table[test.kind, hour_te]),
            "modelo_mediana": mae(np.exp(mu)),
            "verdade_do_simulador": mae(test.oracle_median_s),
        },
        "nll": {"modelo": nll(mu, sigma), "verdade_do_simulador": nll(oracle_mu, oracle_sigma)},
        "calibration": {"quantiles": quantiles, "coverage": coverage,
                        "mean_abs_error": float(np.mean(np.abs(np.array(coverage) - quantiles)))},
        "sigma": {"modelo_medio": float(sigma.mean()), "verdade_media": float(oracle_sigma.mean())},
    }
