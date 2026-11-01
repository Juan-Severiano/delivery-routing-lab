"""Simulador de frota por eventos: pedidos, preparo escondido, entregadores e trânsito escondido.

O que cada ator sabe é o que separa os degraus da escada de medida:
- o simulador sabe tudo (hora em que o pedido fica pronto, tempo real de cada trecho);
- a política (`Policy`) só recebe o que uma plataforma de verdade teria: onde estão os entregadores,
  o que cada um carrega, onde ficam restaurantes e clientes. O B0 não olha nem o trânsito nem o preparo.

Pedido: criado -> (atribuído a um entregador) -> pronto no restaurante -> coletado -> entregue.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .city import City
from .paths import Paths
from .traffic import TRIP_NOISE_SIGMA, Traffic
from .world import Orders, World

PICKUP, DROPOFF = 0, 1
# tipos de evento
ORDER_CREATED, ARRIVE, FREE, SHIFT_START, SHIFT_END, WAKE = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class SimConfig:
    capacity: int = 3  # pedidos ao mesmo tempo na mochila
    handover_s: float = 60.0  # retirada no restaurante
    drop_s: float = 60.0  # entrega ao cliente (o entregador fica ocupado)
    delivered_after_arrival_s: float = 45.0  # o pedido conta como entregue assim que o cliente recebe
    promise_base_s: float = 600.0  # prazo prometido: 10 min fixos...
    promise_per_km_s: float = 360.0  # ... mais 6 min por km em linha reta restaurante-cliente...
    # ... mais o preparo médio esperado do restaurante naquela hora (o que a plataforma estimaria)


@dataclass(frozen=True)
class Fleet:
    shift_start_s: np.ndarray  # (C,)
    shift_end_s: np.ndarray  # (C,) pode passar de 86400 (turno da noite)
    start_node: np.ndarray  # (C,) cruzamento onde o entregador aparece no começo do turno

    def __len__(self) -> int:
        return len(self.shift_start_s)


# Turnos genéricos: nome, fração da frota, janela em horas.
FLEET_KINDS = (
    ("almoço", 0.25, (10.0, 14.5)),
    ("dia todo", 0.25, (10.0, 23.0)),
    ("noite", 0.50, (17.5, 25.0)),
)


def generate_fleet(world: World, city: City, couriers: int, seed: int = 7) -> Fleet:
    rng = np.random.default_rng([seed, 4])
    kind = rng.choice(len(FLEET_KINDS), size=couriers, p=[k[1] for k in FLEET_KINDS])
    start = np.array([FLEET_KINDS[k][2][0] for k in kind]) + rng.uniform(-0.5, 0.5, couriers)
    end = np.array([FLEET_KINDS[k][2][1] for k in kind]) + rng.uniform(-0.5, 0.5, couriers)
    return Fleet(
        shift_start_s=start * 3600.0,
        shift_end_s=end * 3600.0,
        start_node=rng.choice(city.n_nodes, size=couriers, p=world.node_weight).astype(np.int32),
    )


class Policy(Protocol):
    """O que o simulador pede a uma política de despacho.

    Opcionalmente a política também define `bind(sim)`, chamado no começo de cada dia, e
    `started(sim, courier, stop)`, chamado quando o entregador de fato sai para a parada.
    """

    def assign(self, sim: Sim, order: int) -> int:
        """Entregador que recebe o pedido, ou -1 se ninguém pode agora."""

    def next_stop(self, sim: Sim, courier: int) -> tuple[int, int, float] | None:
        """Próxima parada: (PICKUP ou DROPOFF, pedido, hora mais cedo de saída), ou None.

        Se a hora de saída está no futuro, o entregador fica onde está e o simulador chama de novo
        nessa hora (ou antes, se chegar pedido novo): quem planeja pode reordenar enquanto espera.
        """


class NearestPolicy:
    """B0, o piso: o pedido vai para o entregador livre mais próximo e ele faz a parada mais próxima.

    Se ninguém está livre, vai para o mais próximo que ainda tem espaço na mochila. "Próximo" é
    distância em linha reta; o entregador é medido onde estará ao fim do trecho em curso. Não usa
    trânsito, preparo nem rota.
    """

    name = "B0"

    def assign(self, sim: Sim, order: int) -> int:
        eligible = sim.shift_ok & (sim.load < sim.config.capacity)
        if not eligible.any():
            return -1
        free = eligible & (sim.load == 0)
        candidates = free if free.any() else eligible
        d = np.hypot(*(sim.anchor_xy - sim.order_restaurant_xy[order]).T)
        d[~candidates] = np.inf
        return int(d.argmin())

    def next_stop(self, sim: Sim, courier: int) -> tuple[int, int, float] | None:
        bag = sim.bag[courier]
        if not bag:
            return None
        here = sim.city.node_xy[sim.node[courier]]
        best, best_d = None, np.inf
        for o in bag:
            if sim.picked[o]:
                kind, xy = DROPOFF, sim.order_customer_xy[o]
            else:
                kind, xy = PICKUP, sim.order_restaurant_xy[o]
            d = float(np.hypot(xy[0] - here[0], xy[1] - here[1]))
            if d < best_d:
                best, best_d = (kind, o, 0.0), d
        return best


@dataclass
class Leg:
    """Um trecho percorrido por um entregador, para o replay."""

    t0: float
    kind: int  # PICKUP ou DROPOFF: para onde ia
    order: int
    edges: np.ndarray  # ids das arestas, na ordem
    seconds: np.ndarray  # tempo real em cada aresta


@dataclass
class SimResult:
    orders: Orders
    t_created: np.ndarray
    t_ready: np.ndarray
    t_assigned: np.ndarray  # nan se nunca atribuído
    t_picked: np.ndarray
    t_delivered: np.ndarray
    promised: np.ndarray
    courier: np.ndarray  # -1 se não atribuído
    rejected: np.ndarray  # bool: criado quando nenhum entregador estava de turno, então foi recusado
    courier_distance_m: np.ndarray
    courier_busy_s: np.ndarray
    wait_at_restaurant_s: np.ndarray  # por pedido: quanto o entregador esperou o preparo
    legs: list[list[Leg]] | None
    fleet: Fleet
    traffic: Traffic
    elapsed_events: int = 0
    config: SimConfig = field(default_factory=SimConfig)


def sample_prep(world: World, orders: Orders, rain: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Preparo escondido de cada pedido, em segundos: (média esperada do restaurante na hora, valor sorteado).

    O sorteado é a média × lognormal (dispersão do tipo de restaurante), 10% mais lento com chuva. A
    média esperada não inclui a chuva: é a que a plataforma usa para prometer o prazo.
    """
    r = orders.restaurant
    hour = (orders.t_s // 3600).astype(int) % 24
    mean_prep = np.array([world.prep_minutes(h + 0.5) for h in range(24)])[hour, r] * 60.0
    return mean_prep, mean_prep * (1.1 if rain else 1.0) * rng.lognormal(0.0, world.prep_sigma[r])


class Sim:
    def __init__(
        self,
        city: City,
        world: World,
        paths: Paths,
        traffic: Traffic,
        fleet: Fleet,
        orders: Orders,
        policy: Policy,
        day: int = 1,
        config: SimConfig | None = None,
        record: bool = False,
    ):
        self.city, self.world, self.paths, self.traffic = city, world, paths, traffic
        self.fleet, self.orders, self.policy = fleet, orders, policy
        self.config = config or SimConfig()
        self.record = record
        self.rng = np.random.default_rng([world.seed, 3, day])

        n, c = len(orders), len(fleet)
        self.n_orders, self.n_couriers = n, c
        r = orders.restaurant
        self.order_restaurant_xy = world.restaurant_xy[r]
        self.order_customer_xy = orders.customer_xy
        self.order_restaurant_node = world.restaurant_node[r]
        self.order_customer_node = orders.customer_node

        mean_prep, prep = sample_prep(world, orders, traffic.rain, self.rng)
        self.t_created = orders.t_s.copy()
        self.t_ready = self.t_created + prep
        km = np.hypot(*(self.order_customer_xy - self.order_restaurant_xy).T) / 1000.0
        cfg = self.config
        self.promised = self.t_created + cfg.promise_base_s + mean_prep + cfg.promise_per_km_s * km

        # estado dos pedidos
        self.t_assigned = np.full(n, np.nan)
        self.t_picked = np.full(n, np.nan)
        self.t_delivered = np.full(n, np.nan)
        self.courier_of = np.full(n, -1, dtype=np.int64)
        self.rejected = np.zeros(n, dtype=bool)
        self.wait_at_restaurant = np.zeros(n)
        self.picked = [False] * n
        self.pending: list[int] = []

        # estado dos entregadores
        self.node = fleet.start_node.astype(np.int64).tolist()
        self.anchor_xy = city.node_xy[fleet.start_node].copy()  # onde o entregador estará ao fim do trecho
        self.load = np.zeros(c, dtype=np.int16)
        self.shift_ok = np.zeros(c, dtype=bool)
        self.busy = [False] * c
        self.bag: list[list[int]] = [[] for _ in range(c)]
        # trecho em curso (para quem planeja): de onde saiu, quando e para qual parada; None se parado
        self.leg_src = [0] * c
        self.leg_t0 = [0.0] * c
        self.leg_stop: list[tuple[int, int] | None] = [None] * c
        self.wake_token = [0] * c
        self.distance_m = np.zeros(c)
        self.busy_s = np.zeros(c)
        self.legs: list[list[Leg]] | None = [[] for _ in range(c)] if record else None

        self._events: list[tuple[float, int, int, int, int]] = []
        self._seq = 0
        self.now = 0.0
        self.n_events = 0

    # -- eventos ---------------------------------------------------------------------------------
    def _push(self, t: float, kind: int, a: int = 0, b: int = 0) -> None:
        self._seq += 1
        heapq.heappush(self._events, (t, self._seq, kind, a, b))

    def run(self) -> SimResult:
        if hasattr(self.policy, "bind"):
            self.policy.bind(self)
        for o in range(self.n_orders):
            self._push(self.t_created[o], ORDER_CREATED, o)
        for c in range(self.n_couriers):
            self._push(self.fleet.shift_start_s[c], SHIFT_START, c)
            self._push(self.fleet.shift_end_s[c], SHIFT_END, c)
        while self._events:
            t, _, kind, a, b = heapq.heappop(self._events)
            self.now = t
            self.n_events += 1
            if kind == ORDER_CREATED:
                if not self.shift_ok.any():  # ninguém de turno: o aplicativo recusa o pedido
                    self.rejected[a] = True
                    continue
                self.pending.append(a)
                self._assign_pending(t)
            elif kind == ARRIVE:
                self._arrive(t, a, b)
            elif kind == FREE:
                self.busy[a] = False
                self._dispatch(a, t)
                self._assign_pending(t)
            elif kind == SHIFT_START:
                self.shift_ok[a] = True
                self._assign_pending(t)
            elif kind == WAKE:
                if b == self.wake_token[a] and not self.busy[a]:
                    self._dispatch(a, t)
            else:
                self.shift_ok[a] = False
        return self._result()

    def _assign_pending(self, t: float) -> None:
        # todos os pedidos pendentes enfrentam a mesma disponibilidade: se o mais antigo não acha, ninguém acha
        while self.pending:
            o = self.pending[0]
            c = self.policy.assign(self, o)
            if c < 0:
                return
            self.pending.pop(0)
            self.courier_of[o] = c
            self.t_assigned[o] = t
            self.bag[c].append(o)
            self.load[c] += 1
            if not self.busy[c]:
                self._dispatch(c, t)

    def _dispatch(self, c: int, t: float) -> None:
        stop = self.policy.next_stop(self, c)
        self.wake_token[c] += 1  # cancela uma espera anterior: o plano pode ter mudado
        if stop is None:
            return
        kind, o, depart = stop
        if depart > t + 1e-6:
            self._push(depart, WAKE, c, self.wake_token[c])
            return
        if hasattr(self.policy, "started"):
            self.policy.started(self, c, stop)
        dest = int(self.order_restaurant_node[o] if kind == PICKUP else self.order_customer_node[o])
        edges = self.paths.edges(self.node[c], dest)
        noise = float(self.rng.lognormal(0.0, TRIP_NOISE_SIGMA))
        per_edge = self.traffic.edge_seconds[int(t // 3600) % 24, edges] * noise if edges else np.zeros(0)
        duration = float(per_edge.sum())
        self.busy[c] = True
        self.leg_src[c], self.leg_t0[c], self.leg_stop[c] = self.node[c], t, (kind, o)
        self.distance_m[c] += float(self.city.edge_len[edges].sum()) if edges else 0.0
        self.busy_s[c] += duration
        self.anchor_xy[c] = self.city.node_xy[dest]
        if self.legs is not None:
            self.legs[c].append(Leg(t, kind, o, np.asarray(edges, dtype=np.int32), per_edge.astype(np.float32)))
        self._push(t + duration, ARRIVE, c, o * 2 + kind)

    def _arrive(self, t: float, c: int, code: int) -> None:
        o, kind = divmod(code, 2)
        cfg = self.config
        self.leg_stop[c] = None
        if kind == PICKUP:
            self.node[c] = int(self.order_restaurant_node[o])
            wait = max(0.0, self.t_ready[o] - t)
            self.wait_at_restaurant[o] = wait
            done = t + wait + cfg.handover_s
            self.picked[o] = True
            self.t_picked[o] = done
            self.busy_s[c] += wait + cfg.handover_s
        else:
            self.node[c] = int(self.order_customer_node[o])
            self.t_delivered[o] = t + cfg.delivered_after_arrival_s
            done = t + cfg.drop_s
            self.busy_s[c] += cfg.drop_s
            self.bag[c].remove(o)
            self.load[c] -= 1
        self._push(done, FREE, c)

    def _result(self) -> SimResult:
        return SimResult(
            orders=self.orders,
            t_created=self.t_created,
            t_ready=self.t_ready,
            t_assigned=self.t_assigned,
            t_picked=self.t_picked,
            t_delivered=self.t_delivered,
            promised=self.promised,
            courier=self.courier_of,
            rejected=self.rejected,
            courier_distance_m=self.distance_m,
            courier_busy_s=self.busy_s,
            wait_at_restaurant_s=self.wait_at_restaurant,
            legs=self.legs,
            fleet=self.fleet,
            traffic=self.traffic,
            elapsed_events=self.n_events,
            config=self.config,
        )


def summarize(result: SimResult) -> dict[str, float]:
    """Métricas de um dia. Atraso = quanto passou do prazo prometido (0 se entregou a tempo)."""
    done = ~np.isnan(result.t_delivered)
    n = int(len(result.t_created))
    late = np.maximum(0.0, result.t_delivered[done] - result.promised[done]) / 60.0
    total = (result.t_delivered[done] - result.t_created[done]) / 60.0
    wait_assign = (result.t_assigned[done] - result.t_created[done]) / 60.0
    late_only = late[late > 0]
    on_shift_s = float(np.maximum(0.0, result.fleet.shift_end_s - result.fleet.shift_start_s).sum())
    return {
        "orders": n,
        "delivered": int(done.sum()),
        "unserved": int((~done).sum()),  # os recusados por falta de entregador de turno
        "rejected_pct": float(100.0 * result.rejected.mean()),
        "on_time_pct": float(100.0 * (late == 0).mean()),
        "late_mean_min": float(late.mean()),
        "late_p90_min": float(np.percentile(late, 90)),
        "late_p90_among_late_min": float(np.percentile(late_only, 90)) if len(late_only) else 0.0,
        "delivery_mean_min": float(total.mean()),
        "delivery_p50_min": float(np.percentile(total, 50)),
        "delivery_p90_min": float(np.percentile(total, 90)),
        "wait_for_courier_mean_min": float(wait_assign.mean()),
        "wait_for_courier_p90_min": float(np.percentile(wait_assign, 90)),
        "wait_at_restaurant_mean_min": float(result.wait_at_restaurant_s[done].mean() / 60.0),
        "fleet_km": float(result.courier_distance_m.sum() / 1000.0),
        "km_per_order": float(result.courier_distance_m.sum() / 1000.0 / max(1, int(done.sum()))),
        "courier_utilization_pct": float(100.0 * result.courier_busy_s.sum() / max(1.0, on_shift_s)),
    }


def run_day(
    city: City,
    world: World,
    paths: Paths,
    fleet: Fleet,
    policy: Policy,
    day: int,
    config: SimConfig | None = None,
    record: bool = False,
) -> SimResult:
    from .traffic import build_traffic
    from .world import sample_day

    orders = sample_day(world, city, day)
    traffic = build_traffic(city, world.seed, day)
    return Sim(city, world, paths, traffic, fleet, orders, policy, day, config, record).run()


__all__ = [
    "DROPOFF",
    "Fleet",
    "Leg",
    "NearestPolicy",
    "PICKUP",
    "Policy",
    "Sim",
    "SimConfig",
    "SimResult",
    "generate_fleet",
    "run_day",
    "summarize",
]
