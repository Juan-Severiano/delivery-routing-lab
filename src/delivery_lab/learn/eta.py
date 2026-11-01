"""Modelo de ETA: quanto um trecho realmente demora em relação ao tempo livre das ruas.

A rede prevê o log da razão `tempo real / tempo livre` de um trecho a partir de features baratas, que o
planejador consegue montar para milhares de pares por decisão sem reconstruir o caminho: hora do dia,
chuva, onde ficam a origem e o destino, o tempo livre e a distância em linha reta. O rótulo é o tempo
observado nos trechos que os entregadores de fato percorreram (com o ruído de 8% de cada viagem).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..city import City, urban_focus
from ..paths import Paths
from ..sim import Fleet, NearestPolicy, Policy, run_day
from ..traffic import rain_of_day
from ..world import World
from .fit import Data, fit, mse
from .nn import Net, NumpyNet

HARMONICS = 3
N_FEATURES = 2 * HARMONICS + 1 + 4 + 3
HIDDEN = (64, 64)
POSITION_SCALE_M = 5000.0


def eta_features(
    src_xy: np.ndarray, dst_xy: np.ndarray, free_s: np.ndarray, t_s: np.ndarray, rain: np.ndarray, center: np.ndarray
) -> np.ndarray:
    angle = 2.0 * math.pi * (np.asarray(t_s, dtype=float) % 86400.0) / 86400.0
    time_cols = [f(k * angle) for k in range(1, HARMONICS + 1) for f in (np.sin, np.cos)]
    straight = np.hypot(*(np.asarray(dst_xy) - np.asarray(src_xy)).T)
    free_s = np.maximum(np.asarray(free_s, dtype=float), 1.0)
    cols = [
        *time_cols,
        np.asarray(rain, dtype=float),
        *((np.asarray(src_xy) - center).T / POSITION_SCALE_M),
        *((np.asarray(dst_xy) - center).T / POSITION_SCALE_M),
        np.log(free_s) - 5.0,
        np.log(straight + 1.0) - 6.0,
        straight / free_s / 10.0,
    ]
    return np.stack(cols, axis=1).astype(np.float32)


@dataclass
class LegSet:
    src: np.ndarray
    dst: np.ndarray
    t_s: np.ndarray
    rain: np.ndarray
    seconds: np.ndarray  # tempo observado
    free_s: np.ndarray  # tempo livre do mesmo trecho
    expected_s: np.ndarray  # a verdade do simulador sem o ruído da viagem (só para medir o piso)

    def __len__(self) -> int:
        return len(self.src)


def collect_legs(
    city: City, world: World, paths: Paths, fleet: Fleet, days: range, policy: Policy | None = None
) -> LegSet:
    """Trechos percorridos em dias simulados (por padrão com o B0, que é o mais rápido de rodar)."""
    from ..traffic import build_traffic

    cols: dict[str, list] = {k: [] for k in ("src", "dst", "t", "rain", "sec", "free", "exp")}
    for day in days:
        result = run_day(city, world, paths, fleet, policy or NearestPolicy(), day, record=True)
        traffic = build_traffic(city, world.seed, day)
        rain = rain_of_day(world.seed, day)
        for legs in result.legs:
            for leg in legs:
                if len(leg.edges) == 0:
                    continue
                u, v = int(city.edge_src[leg.edges[0]]), int(city.edge_dst[leg.edges[-1]])
                cols["src"].append(u)
                cols["dst"].append(v)
                cols["t"].append(leg.t0)
                cols["rain"].append(rain)
                cols["sec"].append(float(leg.seconds.sum()))
                cols["free"].append(float(paths.dist[u, v]))
                cols["exp"].append(float(traffic.edge_seconds[int(leg.t0 // 3600) % 24, leg.edges].sum()))
    return LegSet(
        src=np.array(cols["src"]), dst=np.array(cols["dst"]), t_s=np.array(cols["t"]), rain=np.array(cols["rain"]),
        seconds=np.array(cols["sec"]), free_s=np.array(cols["free"]), expected_s=np.array(cols["exp"]),
    )


class EtaModel:
    def __init__(self, state: dict[str, np.ndarray], center: np.ndarray, y_mean: float, meta: dict | None = None):
        self.net = NumpyNet(state)
        self.state = state
        self.center = np.asarray(center, dtype=float)
        self.y_mean = float(y_mean)
        self.meta = meta or {}

    def ratio(self, src_xy, dst_xy, free_s, t_s, rain) -> np.ndarray:
        """Quanto o trecho demora em relação ao tempo livre (1 = livre)."""
        x = eta_features(src_xy, dst_xy, free_s, t_s, rain, self.center)
        return np.exp(self.net(x)[:, 0].astype(float) + self.y_mean)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, center=self.center, y_mean=np.array(self.y_mean), meta=np.array(json.dumps(self.meta)), **self.state)

    @classmethod
    def load(cls, path: Path) -> EtaModel:
        with np.load(Path(path)) as data:
            files = {k: data[k] for k in data.files}
        meta = json.loads(str(files.pop("meta")))
        return cls(files, files.pop("center"), float(files.pop("y_mean")), meta)


def _arrays(city: City, s: LegSet, center: np.ndarray):
    x = eta_features(city.node_xy[s.src], city.node_xy[s.dst], s.free_s, s.t_s, s.rain, center)
    return x, np.log(np.maximum(s.seconds, 1.0) / np.maximum(s.free_s, 1.0))


def train_eta(city: City, train: LegSet, val: LegSet, epochs: int = 30, seed: int = 0, log=print):
    center, _ = urban_focus(city.node_xy)
    x_tr, y_tr = _arrays(city, train, center)
    x_va, y_va = _arrays(city, val, center)
    y_mean = float(y_tr.mean())
    net = Net(N_FEATURES, HIDDEN, 1, seed=seed)
    history = fit(
        net, mse,
        Data(x_tr, (y_tr - y_mean).astype(np.float32)[:, None]),
        Data(x_va, (y_va - y_mean).astype(np.float32)[:, None]),
        epochs=epochs, batch=2048, lr=4e-3, lr_end=2e-4, patience=6, seed=seed, log=log,
    )
    meta = {"train_legs": len(train), "val_legs": len(val), "best_epoch": history.best_epoch,
            "hidden": list(HIDDEN), "seed": seed}
    return EtaModel(net.state(), center, y_mean, meta), history


def evaluate_eta(city: City, model: EtaModel, test: LegSet, train: LegSet) -> dict:
    """Erro do tempo previsto de um trecho, em segundos e em %, contra o tempo livre e uma tabela por hora."""
    pred = test.free_s * model.ratio(city.node_xy[test.src], city.node_xy[test.dst], test.free_s, test.t_s, test.rain)
    hour_tr = (train.t_s // 3600).astype(int) % 24
    hour_te = (test.t_s // 3600).astype(int) % 24
    mean_ratio = np.array([np.mean(train.seconds[hour_tr == h] / np.maximum(train.free_s[hour_tr == h], 1.0)) if (hour_tr == h).any() else 1.0 for h in range(24)])
    table = test.free_s * mean_ratio[hour_te]
    peak = np.isin(hour_te, (12, 13, 18, 19, 20))

    def row(p, mask=None):
        m = np.ones(len(p), bool) if mask is None else mask
        err = np.abs(test.seconds[m] - p[m])
        return {"mae_s": float(err.mean()), "mape_pct": float(100 * (err / test.seconds[m]).mean()),
                "bias_pct": float(100 * ((p[m] - test.seconds[m]) / test.seconds[m]).mean())}

    out = {"legs": len(test), "peak_share": float(peak.mean())}
    for name, p in (("tempo_livre", test.free_s), ("tabela_por_hora", table), ("modelo", pred), ("verdade_sem_ruido_de_viagem", test.expected_s)):
        out[name] = {"todos": row(p), "pico": row(p, peak)}
    return out
