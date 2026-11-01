"""O Jev: decide a próxima parada de um entregador e diz por quê.

`decide(features, candidatas)` recebe as paradas que o entregador pode fazer agora (coletas pendentes e
entregas de pedidos já coletados) e devolve a **probabilidade de cada uma**, a confiança e quanto cada
fator pesou. A pontuação de cada candidata é um modelo próprio (linear ou uma rede pequena) e a
probabilidade é um softmax entre as candidatas, treinado para imitar a escolha do B2, o planejador que vê
o preparo e o trânsito verdadeiros. O Jev só usa o que a plataforma enxerga: o ETA e o preparo aprendidos
(Passo 5), o prazo prometido e a idade do pedido.

No modelo linear a explicação é exata: a contribuição do fator j para a candidata i é
`w_j * (z_ij - média_das_candidatas z_j)`, e a soma delas é a diferença de pontuação da candidata para a
candidata "média". Na rede pequena não há explicação assim, só a comparação de acerto.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..city import City
from ..paths import Paths
from ..planner import PREP_QUANTILE, Planner, Stop
from ..sim import PICKUP, Fleet, Sim, run_day
from ..world import World
from .eta import EtaModel
from .fit import Data, fit
from .nn import Net, NumpyNet
from .prep import PrepModel

MAX_CANDIDATES = 6  # mochila de 3 pedidos: no máximo 3 coletas e 3 entregas
FEATURES = (
    "travel_min",
    "is_pickup",
    "wait_min",
    "slack_min",
    "age_min",
    "late",
    "spread_min",
    "bag_after",
    "ready_gap_min",
)
FEATURE_LABELS = {
    "travel_min": "Tempo até a parada",
    "is_pickup": "É uma coleta",
    "wait_min": "Espera prevista no restaurante",
    "slack_min": "Folga até o prazo",
    "age_min": "Idade do pedido",
    "late": "Pedido já atrasado",
    "spread_min": "Distância às outras paradas",
    "bag_after": "Pedidos na mochila depois",
    "ready_gap_min": "Falta para o pedido ficar pronto",
}
N_FEATURES = len(FEATURES)


def candidate_features(view: Planner, sim: Sim, courier: int, cands: list[Stop]) -> np.ndarray:
    """Features de cada candidata, só com o que a plataforma enxerga (`view` é um planejador B1+)."""
    now, node, n = sim.now, sim.node[courier], len(cands)
    dest = [view.shop_node[o] if kind == PICKUP else view.home_node[o] for kind, o in cands]
    pick = [i for i, (kind, _) in enumerate(cands) if kind == PICKUP]
    u = [node] * n + [view.shop_node[cands[i][1]] for i in pick]
    v = dest + [view.home_node[cands[i][1]] for i in pick]
    t = view.leg_times(sim, np.array(u), np.array(v))
    travel = t[:n]
    shop_to_home = dict(zip(pick, t[n:]))
    if n > 1:
        d = np.asarray(sim.paths.dist[np.ix_(dest, dest)], dtype=float)
        spread = d.sum(axis=1) / (n - 1)
    else:
        spread = np.zeros(1)
    carried = sum(1 for o in sim.bag[courier] if sim.picked[o])
    handover, after = sim.config.handover_s, sim.config.delivered_after_arrival_s

    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i, (kind, o) in enumerate(cands):
        arrival = now + travel[i]
        if kind == PICKUP:
            ready = view.ready[o]
            wait = max(0.0, ready - arrival)
            delivery = max(arrival, ready) + handover + shop_to_home[i] + after
            gap, bag_after = (ready - now) / 60.0, carried + 1
        else:
            wait, delivery, gap, bag_after = 0.0, arrival + after, 0.0, carried - 1
        slack = (view.promised[o] - delivery) / 60.0
        out[i] = (
            travel[i] / 60.0,
            kind == PICKUP,
            min(wait / 60.0, 60.0),
            min(max(slack, -60.0), 90.0),
            min((now - view.created[o]) / 60.0, 120.0),
            slack < 0.0,
            spread[i] / 60.0,
            bag_after,
            min(max(gap, -30.0), 60.0),
        )
    return out


@dataclass(frozen=True)
class Decision:
    """O contrato do Jev: a escolha, a probabilidade de cada candidata e o porquê."""

    candidates: tuple[Stop, ...]
    chosen: int  # índice da candidata escolhida
    probabilities: np.ndarray  # (n,) soma 1
    confidence: float  # probabilidade da escolhida
    margin: float  # probabilidade da escolhida menos a da segunda
    contributions: np.ndarray | None  # (n, F) por fator, só no modelo linear
    factors: tuple[str, ...] = FEATURES

    @property
    def stop(self) -> Stop:
        return self.candidates[self.chosen]


def _softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max())
    return e / e.sum()


class JevModel:
    def __init__(self, kind: str, state: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray, meta: dict | None = None):
        self.kind, self.state, self.meta = kind, state, meta or {}
        self.mean, self.std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
        self.net = NumpyNet(state)
        self.weights = state["W0"][:, 0].astype(np.float32) if kind == "linear" else None

    def _scores(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = (np.asarray(feats, dtype=np.float32) - self.mean) / self.std
        return self.net(z)[:, 0], z

    def decide(self, feats: np.ndarray, cands: list[Stop]) -> Decision:
        scores, z = self._scores(feats)
        p = _softmax(scores.astype(float))
        i = int(p.argmax())
        second = np.sort(p)[-2] if len(p) > 1 else 0.0
        contrib = None
        if self.weights is not None:
            contrib = (self.weights[None, :] * (z - z.mean(axis=0, keepdims=True))).astype(float)
        return Decision(tuple(cands), i, p, float(p[i]), float(p[i] - second), contrib)

    def probabilities_batch(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Probabilidades de (N, K, F) candidatas com máscara; as inválidas ficam com 0."""
        n, k, f = x.shape
        z = (x.reshape(n * k, f) - self.mean) / self.std
        scores = np.where(mask, self.net(z)[:, 0].reshape(n, k).astype(float), -1e9)
        e = np.exp(scores - scores.max(axis=1, keepdims=True)) * mask
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, kind=np.array(self.kind), mean=self.mean, std=self.std,
                            meta=np.array(json.dumps(self.meta)), **self.state)

    @classmethod
    def load(cls, path: Path) -> JevModel:
        with np.load(Path(path)) as data:
            files = {k: data[k] for k in data.files}
        meta = json.loads(str(files.pop("meta")))
        kind, mean, std = str(files.pop("kind")), files.pop("mean"), files.pop("std")
        return cls(kind, files, mean, std, meta)


# -- dados: as decisões do B2 -----------------------------------------------------------------------


@dataclass
class JevSet:
    x: np.ndarray  # (N, K, F) float32
    mask: np.ndarray  # (N, K) bool
    y: np.ndarray  # (N,) índice da escolha do B2
    planner_view: np.ndarray  # (N,) escolha do planejador re-otimizando com a visão da plataforma; -1 se não medido

    def __len__(self) -> int:
        return len(self.y)


class Recorder(Planner):
    """O B2 (o professor) que anota cada decisão com as features que a plataforma enxergaria."""

    def __init__(self, prep_model: PrepModel, eta_model: EtaModel, quantile: float = PREP_QUANTILE, planner_view: bool = False):
        super().__init__(oracle=True)
        self._models = (prep_model, eta_model, quantile)
        self.planner_view = planner_view
        self.rows: list[tuple[np.ndarray, int, int]] = []

    def bind(self, sim: Sim) -> None:
        super().bind(sim)
        prep_model, eta_model, quantile = self._models
        self.view = Planner(prep_model=prep_model, eta_model=eta_model, prep_quantile=quantile)
        self.view.bind(sim)

    def started(self, sim: Sim, courier: int, stop: tuple[int, int, float]) -> None:
        plan = self.plan[courier]
        cands = [s for s in plan if s[0] == PICKUP or sim.picked[s[1]]]
        if len(cands) >= 2:
            feats = candidate_features(self.view, sim, courier, cands)
            idx = cands.index((stop[0], stop[1]))
            pv = -1
            if self.planner_view:
                pv = cands.index(self.view.exhaustive_first_stop(sim, courier, list(plan)))
            self.rows.append((feats, idx, pv))
        super().started(sim, courier, stop)


def build_jev_set(
    city: City, world: World, paths: Paths, fleet: Fleet, days: range, prep_model: PrepModel, eta_model: EtaModel,
    planner_view: bool = False,
) -> JevSet:
    rows: list[tuple[np.ndarray, int, int]] = []
    for day in days:
        rec = Recorder(prep_model, eta_model, planner_view=planner_view)
        run_day(city, world, paths, fleet, rec, day)
        rows.extend(rec.rows)
    n = len(rows)
    x = np.zeros((n, MAX_CANDIDATES, N_FEATURES), dtype=np.float32)
    mask = np.zeros((n, MAX_CANDIDATES), dtype=bool)
    for i, (feats, _, _) in enumerate(rows):
        x[i, : len(feats)] = feats
        mask[i, : len(feats)] = True
    return JevSet(x, mask, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))


# -- treino -----------------------------------------------------------------------------------------


class SetScorer:
    """Aplica a mesma rede a cada candidata de uma decisão e mascara as que não existem."""

    def __init__(self, net: Net, k: int, f: int):
        self.net, self.k, self.f, self.dtype = net, k, f, net.dtype

    def parameters(self):
        return self.net.parameters()

    def state(self):
        return self.net.state()

    def load_state(self, state):
        self.net.load_state(state)

    def forward(self, x: torch.Tensor, ids=None) -> torch.Tensor:
        b = x.shape[0]
        feats = x[:, : self.k * self.f].reshape(b * self.k, self.f)
        mask = x[:, self.k * self.f :]
        scores = self.net.forward(feats).reshape(b, self.k)
        return scores.masked_fill(mask < 0.5, -1e9)


def set_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Entropia cruzada do softmax entre as candidatas: -log da probabilidade da escolha do B2."""
    picked = out.gather(1, y[:, :1].long())[:, 0]
    return (torch.logsumexp(out, dim=1) - picked).mean()


def _flat(s: JevSet, mean: np.ndarray, std: np.ndarray) -> Data:
    z = ((s.x - mean) / std) * s.mask[..., None]
    flat = np.concatenate([z.reshape(len(s), -1), s.mask.astype(np.float32)], axis=1)
    return Data(flat.astype(np.float32), s.y[:, None].astype(np.float32))


def train_jev(
    train: JevSet, val: JevSet, kind: str = "linear", epochs: int = 30, seed: int = 0, batch: int = 2048, log=print
) -> JevModel:
    hidden = () if kind == "linear" else (32, 16)
    valid = train.mask
    mean = train.x[valid].mean(axis=0)
    std = train.x[valid].std(axis=0) + 1e-6
    net = Net(N_FEATURES, hidden, 1, seed=seed)
    scorer = SetScorer(net, MAX_CANDIDATES, N_FEATURES)
    history = fit(scorer, set_nll, _flat(train, mean, std), _flat(val, mean, std), epochs=epochs, batch=batch,
                  lr=1e-2, lr_end=5e-4, patience=6, seed=seed, log=log)
    meta = {"train_decisions": len(train), "val_decisions": len(val), "best_epoch": history.best_epoch,
            "hidden": list(hidden), "seed": seed}
    return JevModel(kind, net.state(), mean, std, meta)


# -- avaliação --------------------------------------------------------------------------------------


def evaluate_jev(model: JevModel, test: JevSet, latency_sample: int = 4000) -> dict:
    """Acerto contra o B2, calibração e latência, contra regras simples e o planejador re-otimizando."""
    p = model.probabilities_batch(test.x, test.mask)
    top = p.argmax(axis=1)
    y = test.y
    correct = top == y
    conf = p.max(axis=1)
    nll = float(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-9))))
    edges = np.linspace(0.0, 1.0, 11)
    bins, ece = [], 0.0
    for lo, hi in zip(edges, edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())

    travel, slack = test.x[:, :, 0], test.x[:, :, 3]
    big = np.where(test.mask, 0.0, np.inf)
    nearest = (travel + big).argmin(axis=1)
    earliest = (slack + big).argmin(axis=1)
    n_cand = test.mask.sum(axis=1)
    out = {
        "decisions": len(y),
        "mean_candidates": float(n_cand.mean()),
        "top1": float(correct.mean()),
        "chance": float(np.mean(1.0 / n_cand)),
        "nll": nll,
        "prob_of_b2_choice": float(np.mean(p[np.arange(len(y)), y])),
        "ece": float(ece),
        "reliability": bins,
        "baselines_top1": {"parada_mais_proxima": float((nearest == y).mean()), "menor_folga_do_prazo": float((earliest == y).mean())},
        "by_candidates": {int(k): float(correct[n_cand == k].mean()) for k in sorted(set(n_cand.tolist()))},
    }
    seen = test.planner_view >= 0
    if seen.any():
        out["baselines_top1"]["planejador_reotimizando"] = float((test.planner_view[seen] == y[seen]).mean())
        out["top1_on_planner_subset"] = float(correct[seen].mean())

    # latência só do modelo (sem montar as features)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(latency_sample, len(y)), replace=False)
    items = [(test.x[i][test.mask[i]], [(0, 0)] * int(n_cand[i])) for i in idx]
    start = time.perf_counter()
    for feats, cands in items:
        model.decide(feats, cands)
    out["decide_us"] = (time.perf_counter() - start) / len(items) * 1e6
    return out


# -- a política: o Jev decidindo a sequência na frota ------------------------------------------------


class JevPlanner(Planner):
    """B1+ na atribuição (inserção mais barata com os modelos aprendidos) e o Jev na sequência.

    O plano continua existindo só para avaliar a inserção de pedidos novos. Na hora de despachar, o Jev
    escolhe a próxima parada entre as viáveis.
    """

    def __init__(self, prep_model: PrepModel, eta_model: EtaModel, jev: JevModel, record: bool = False, **kw):
        super().__init__(prep_model=prep_model, eta_model=eta_model, **kw)
        self.jev, self.record = jev, record
        self.name = "Jev" if jev.kind == "linear" else "Jev (rede)"
        self.decide_s: list[float] = []

    def bind(self, sim: Sim) -> None:
        super().bind(sim)
        self._last: dict[int, tuple[list[Stop], Decision | None]] = {}
        self.decisions: list[dict] = []

    def next_stop(self, sim: Sim, courier: int):
        plan = self.plan[courier]
        if not plan:
            return None
        cands = [s for s in plan if s[0] == PICKUP or sim.picked[s[1]]]
        decision = None
        if len(cands) == 1:
            chosen = cands[0]
        else:
            start = time.perf_counter()
            decision = self.jev.decide(candidate_features(self, sim, courier, cands), cands)
            self.decide_s.append(time.perf_counter() - start)
            chosen = decision.stop
        self._last[courier] = (cands, decision)
        return self.with_departure(sim, courier, chosen)

    def started(self, sim: Sim, courier: int, stop: tuple[int, int, float]) -> None:
        self.plan[courier].remove((stop[0], stop[1]))
        _, decision = self._last.pop(courier, (None, None))
        if self.record and decision is not None:
            self.decisions.append({"courier": courier, "t": sim.now, "decision": decision})
