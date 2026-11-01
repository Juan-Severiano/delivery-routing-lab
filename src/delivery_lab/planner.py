"""Planejadores B1 e B2: inserção mais barata + 2-opt, com coleta antes da entrega e mochila limitada.

Cada entregador tem um plano: a lista das paradas que ainda não começou. Quando um pedido nasce, o
planejador testa, para os K entregadores elegíveis mais próximos, todas as posições em que a coleta e a
entrega do pedido novo cabem no plano (a entrega depois da coleta) e fica com a que menos piora o
custo da frota. Depois o 2-opt (inverter um trecho do plano) melhora a ordem das paradas daquele
entregador. O custo de um plano é a soma, sobre os pedidos dele, do tempo desde a criação até a
entrega mais `LATE_WEIGHT` vezes o atraso sobre o prazo prometido.

O que separa os dois degraus é só o que eles enxergam:

- **B1** estima os tempos pelo tempo livre das ruas e supõe `PREP_GUESS_S` de preparo para todo
  restaurante (uma regra de bolso). Não vê trânsito nem a hora real em que o pedido fica pronto.
- **B2**, o teto, usa o tempo real das arestas na hora e a hora exata em que cada pedido fica pronto.
  Só não vê o ruído aleatório de cada viagem (8%), que o simulador sorteia ao despachar.

Os dois seguram a saída do entregador até "chegar quando o pedido fica pronto" e, enquanto ele
espera, o plano pode mudar (um pedido novo e perto entra antes da coleta).
"""

from __future__ import annotations

import time

import numpy as np

from .sim import DROPOFF, PICKUP, Sim

PREP_GUESS_S = 700.0  # B1: preparo (~12 min) que a plataforma supõe para qualquer restaurante; afinado nos dias 1 a 10
LATE_WEIGHT = 3.0  # cada segundo de atraso sobre o prazo pesa 3 vezes o tempo de fluxo
K_NEAREST = 40  # entregadores elegíveis mais próximos que são avaliados a cada pedido
# Sai um pouco antes do "chegar quando ficar pronto". O B1 erra mais a hora de ficar pronto, então
# ganha mais folga. Parâmetros afinados nos dias 1 a 10; as medições do README são dos dias 101 a 120.
DEPART_MARGIN_B1_S = 150.0
DEPART_MARGIN_B2_S = 30.0

Stop = tuple[int, int]  # (PICKUP ou DROPOFF, pedido)


class Planner:
    def __init__(
        self,
        oracle: bool = False,
        k: int = K_NEAREST,
        prep_guess_s: float = PREP_GUESS_S,
        late_weight: float = LATE_WEIGHT,
        depart_margin_s: float | None = None,
        know_ready: bool | None = None,
        know_traffic: bool | None = None,
    ):
        # `oracle` liga as duas coisas; `know_ready` e `know_traffic` as separam (ablação)
        self.know_ready = oracle if know_ready is None else know_ready
        self.know_traffic = oracle if know_traffic is None else know_traffic
        self.oracle = self.know_ready and self.know_traffic
        self.name = (
            "B2"
            if self.oracle
            else "B1" if not (self.know_ready or self.know_traffic)
            else "B1 com preparo real" if self.know_ready
            else "B1 com trânsito real"
        )
        self.k = k
        self.prep_guess_s = prep_guess_s
        self.late_weight = late_weight
        self.depart_margin_s = depart_margin_s if depart_margin_s is not None else (
            DEPART_MARGIN_B2_S if self.know_ready else DEPART_MARGIN_B1_S
        )
        self.plan: list[list[Stop]] = []
        self.decision_s: list[float] = []  # tempo de cada `assign`, para o profile

    # -- ciclo de vida -----------------------------------------------------------------------------
    def bind(self, sim: Sim) -> None:
        self.plan = [[] for _ in range(sim.n_couriers)]
        self.created = sim.t_created.tolist()
        self.promised = sim.promised.tolist()
        # o B2 enxerga a hora real em que o pedido fica pronto; o B1 usa a regra de bolso
        self.ready = (sim.t_ready if self.know_ready else sim.t_created + self.prep_guess_s).tolist()
        self.shop_node = sim.order_restaurant_node.tolist()
        self.home_node = sim.order_customer_node.tolist()
        cfg = sim.config
        self.handover, self.drop, self.delivered_after = cfg.handover_s, cfg.drop_s, cfg.delivered_after_arrival_s

    def started(self, sim: Sim, courier: int, stop: tuple[int, int, float]) -> None:
        head = self.plan[courier].pop(0)
        assert head == (stop[0], stop[1]), "o simulador começou uma parada que não é a primeira do plano"

    # -- tempos ------------------------------------------------------------------------------------
    def _node(self, stop: Stop) -> int:
        return self.shop_node[stop[1]] if stop[0] == PICKUP else self.home_node[stop[1]]

    def leg_times(self, sim: Sim, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Segundos de u até v, em lote: tempo livre, ou o tempo real das arestas na hora de agora."""
        if not self.know_traffic:
            return np.asarray(sim.paths.dist[u, v], dtype=float)
        seconds = sim.traffic.edge_seconds[int(sim.now // 3600) % 24]
        src, pred = sim.city.edge_src, sim.paths.pred
        total = np.zeros(len(u))
        cur = v.copy()
        alive = cur != u
        while alive.any():
            idx = np.nonzero(alive)[0]
            e = pred[u[idx], cur[idx]].astype(np.int64)
            total[idx] += seconds[e]
            cur[idx] = src[e]
            alive = cur != u
        return total

    def cost(self, seq: list[Stop], node: int, t: float, travel: dict[tuple[int, int], float]) -> float:
        """Custo de percorrer `seq` a partir de `node` na hora `t`."""
        total = 0.0
        for kind, o in seq:
            dest = self.shop_node[o] if kind == PICKUP else self.home_node[o]
            if dest != node:
                t += travel[(node, dest)]
            if kind == PICKUP:
                if self.ready[o] > t:
                    t = self.ready[o]  # espera o pedido ficar pronto
                t += self.handover
            else:
                delivered = t + self.delivered_after
                late = delivered - self.promised[o]
                total += delivered - self.created[o] + (self.late_weight * late if late > 0 else 0.0)
                t += self.drop
            node = dest
        return total

    # -- decisões ----------------------------------------------------------------------------------
    def assign(self, sim: Sim, order: int) -> int:
        started = time.perf_counter()
        eligible = sim.shift_ok & (sim.load < sim.config.capacity)
        n_eligible = int(eligible.sum())
        if n_eligible == 0:
            return -1
        dist = np.hypot(*(sim.anchor_xy - sim.order_restaurant_xy[order]).T)
        dist[~eligible] = np.inf
        k = min(self.k, n_eligible)
        candidates = np.argpartition(dist, k - 1)[:k].tolist() if k < len(dist) else list(range(len(dist)))

        shop, home = self.shop_node[order], self.home_node[order]
        contexts, u_all, v_all = [], [], []
        for c in candidates:
            leg = sim.leg_stop[c]
            node, t0, prefix = (sim.leg_src[c], sim.leg_t0[c], [leg]) if leg is not None else (sim.node[c], sim.now, [])
            pending = self.plan[c]
            nodes = {node, shop, home} | {self._node(s) for s in prefix} | {self._node(s) for s in pending}
            pairs = [(a, b) for a in nodes for b in nodes if a != b]
            contexts.append((c, node, t0, prefix, pending, pairs, len(u_all)))
            u_all.extend(a for a, _ in pairs)
            v_all.extend(b for _, b in pairs)
        times = self.leg_times(sim, np.array(u_all), np.array(v_all)).tolist()

        best = None  # ((aumento de custo, distância), entregador, plano, contexto)
        for c, node, t0, prefix, pending, pairs, lo in contexts:
            travel = dict(zip(pairs, times[lo : lo + len(pairs)]))
            increase, seq = self.best_insertion(order, prefix, pending, node, t0, travel)
            key = (increase, float(dist[c]))
            if best is None or key < best[0]:
                best = (key, c, seq, (node, t0, prefix, travel))
        _, c, seq, (node, t0, prefix, travel) = best
        self.plan[c] = self.two_opt(seq, prefix, node, t0, travel)
        self.decision_s.append(time.perf_counter() - started)
        return c

    def best_insertion(
        self, order: int, prefix: list[Stop], pending: list[Stop], node: int, t0: float, travel
    ) -> tuple[float, list[Stop]]:
        """Melhor posição para a coleta e a entrega do pedido: (aumento de custo, novo plano)."""
        base = self.cost(prefix + pending, node, t0, travel)
        best_increase, best_seq = float("inf"), pending
        n = len(pending)
        for i in range(n + 1):
            for j in range(i, n + 1):
                seq = pending[:i] + [(PICKUP, order)] + pending[i:j] + [(DROPOFF, order)] + pending[j:]
                increase = self.cost(prefix + seq, node, t0, travel) - base
                if increase < best_increase:
                    best_increase, best_seq = increase, seq
        return best_increase, best_seq

    def two_opt(self, seq: list[Stop], prefix: list[Stop], node: int, t0: float, travel) -> list[Stop]:
        """Inverte trechos do plano enquanto o custo cai, respeitando coleta antes da entrega."""
        best, best_cost = seq, self.cost(prefix + seq, node, t0, travel)
        improved, passes = True, 0
        while improved and passes < 4:
            improved, passes = False, passes + 1
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                    if not _valid(cand):
                        continue
                    cost = self.cost(prefix + cand, node, t0, travel)
                    if cost < best_cost - 1e-9:
                        best, best_cost, improved = cand, cost, True
        return best

    def next_stop(self, sim: Sim, courier: int) -> tuple[int, int, float] | None:
        plan = self.plan[courier]
        if not plan:
            return None
        kind, o = plan[0]
        depart = sim.now
        if kind == PICKUP:
            node, dest = sim.node[courier], self.shop_node[o]
            travel = 0.0 if node == dest else float(self.leg_times(sim, np.array([node]), np.array([dest]))[0])
            depart = max(sim.now, self.ready[o] - travel - self.depart_margin_s)
        return kind, o, depart


def _valid(seq: list[Stop]) -> bool:
    """A entrega de um pedido nunca vem antes da coleta dele (quando as duas estão no plano)."""
    pickups = {o for kind, o in seq if kind == PICKUP}
    seen: set[int] = set()
    for kind, o in seq:
        if kind == PICKUP:
            seen.add(o)
        elif o in pickups and o not in seen:
            return False
    return True
