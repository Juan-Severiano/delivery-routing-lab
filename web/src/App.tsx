import { useEffect, useMemo, useRef, useState } from "react";
import { CLASS_LABEL, loadCity, type City } from "./city";
import type { CandidateMark, Layers } from "./layers";
import { MapCanvas, type FitRequest } from "./MapCanvas";
import {
  MODE_AT_SHOP,
  MODE_OFFLINE,
  MODE_STOPPED,
  MODE_TO_DROP,
  MODE_TO_PICKUP,
  clockLabel,
  computeFrame,
  loadReplay,
  newFrame,
  openOrders,
  type Frame,
  type Replay,
} from "./replay";
import { loadWorld, type World } from "./world";

const int = new Intl.NumberFormat("pt-BR");
const km = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const one = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const SPEEDS = [60, 300, 1200];
type PolicyId = "b0" | "b1" | "b1plus" | "jev" | "b2";
const POLICIES: PolicyId[] = ["b0", "b1", "b1plus", "jev", "b2"];
const POLICY_INFO: Record<PolicyId, { name: string; role: string; text: string }> = {
  b0: {
    name: "B0",
    role: "piso",
    text: "Manda o entregador livre mais próximo assim que o pedido nasce e faz sempre a parada mais próxima. Não vê trânsito nem preparo.",
  },
  b1: {
    name: "B1",
    role: "clássico",
    text: "Inserção mais barata e 2-opt, com tempo livre das ruas e uma regra de bolso de ~12 min de preparo. Segura a saída até chegar quando o pedido fica pronto.",
  },
  b1plus: {
    name: "B1+",
    role: "aprendido",
    text: "O B1 com o modelo de preparo e o ETA aprendidos no lugar das regras. Só usa o que a plataforma sabe: restaurante, hora, previsão do tempo e posições.",
  },
  jev: {
    name: "Jev",
    role: "explica",
    text: "Atribui como o B1+ e escolhe a próxima parada com o Jev, um modelo linear treinado para imitar o B2 só com o que a plataforma enxerga. Clique num entregador para ver as candidatas, a probabilidade de cada uma e o que pesou.",
  },
  b2: {
    name: "B2",
    role: "teto",
    text: "O mesmo planejador, mas vendo o tempo real do trânsito e a hora exata em que cada pedido fica pronto. Só não vê o ruído de cada viagem.",
  },
};
const MODE_LABEL: Record<number, string> = {
  [MODE_OFFLINE]: "fora do turno",
  [MODE_TO_PICKUP]: "a caminho de coletar",
  [MODE_TO_DROP]: "a caminho de entregar",
  [MODE_AT_SHOP]: "no restaurante",
  [MODE_STOPPED]: "parado",
};

export function App() {
  const [city, setCity] = useState<City | null>(null);
  const [world, setWorld] = useState<World | null>(null);
  const [replays, setReplays] = useState<Partial<Record<PolicyId, Replay>>>({});
  const [policy, setPolicy] = useState<PolicyId>("b1plus");
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState<boolean[]>([]);
  const [fit, setFit] = useState<FitRequest>({ target: "focus", nonce: 0 });
  const [clock, setClock] = useState(12 * 3600 + 30 * 60);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [selected, setSelected] = useState<number | null>(null);
  const [layers, setLayers] = useState<Layers>({ heat: false, restaurants: true, orders: true, couriers: true });
  const buffer = useRef<Frame | null>(null);

  useEffect(() => {
    const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e));
    loadCity()
      .then((c) => {
        setCity(c);
        setVisible(c.payload.classes.map(() => true));
      })
      .catch(fail);
    loadWorld().then(setWorld).catch(fail);
    for (const id of POLICIES) {
      loadReplay(`/data/replay-${id}.json`)
        .then((r) => setReplays((all) => ({ ...all, [id]: r })))
        .catch(fail);
    }
  }, []);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let id = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.25, (now - last) / 1000);
      last = now;
      setClock((c) => (c + dt * speed >= 86400 ? 0 : c + dt * speed));
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, speed]);

  const replay = replays[policy] ?? null;

  const frame = useMemo(() => {
    if (!replay || !city) return null;
    if (!buffer.current || buffer.current.mode.length !== replay.shiftStart.length) {
      buffer.current = newFrame(replay.shiftStart.length);
    }
    return computeFrame(replay, city, clock, buffer.current);
    // o buffer é reaproveitado; o relógio é o que muda
  }, [replay, city, clock]);

  const legIndex = selected !== null && frame ? frame.leg[selected] : -1;
  const decision = useMemo(() => {
    const d = replay?.payload.decisions;
    if (!replay || !world || !d || legIndex < 0) return null;
    const i = replay.decisionOfLeg[legIndex];
    if (i < 0) return null;
    const r = world.payload.restaurants;
    const candidates: DecisionCandidate[] = [];
    for (let k = d.ptr[i]; k < d.ptr[i + 1]; k++) {
      const o = d.order[k];
      const shop = replay.payload.orders.restaurant[o];
      const pickup = d.kind[k] === 0;
      candidates.push({
        pickup,
        shopKind: replay.payload.orders.kind[o],
        prob: d.prob[k],
        x: pickup ? r.x[shop] : replay.order.x[o],
        y: pickup ? r.y[shop] : replay.order.y[o],
        contrib: d.contrib[k] ?? null,
        chosen: k - d.ptr[i] === d.chosen[i],
      });
    }
    return { t: replay.legT0[legIndex], candidates, labels: d.labels };
  }, [replay, world, legIndex]);
  const marks = useMemo<CandidateMark[]>(
    () => (decision ? decision.candidates.map((c) => ({ x: c.x, y: c.y, prob: c.prob, chosen: c.chosen })) : []),
    [decision],
  );

  const choosePolicy = (id: PolicyId) => {
    setPolicy(id);
    setSelected(null); // os entregadores de uma política não são os da outra
  };

  const refit = (target: FitRequest["target"]) => setFit((f) => ({ target, nonce: f.nonce + 1 }));
  const toggleRoad = (i: number) => setVisible((v) => v.map((on, k) => (k === i ? !on : on)));
  const toggleLayer = (name: keyof Layers) => setLayers((l) => ({ ...l, [name]: !l[name] }));

  const hour = Math.floor(clock / 3600) % 24;
  const scale = world?.payload.scale;
  const online = world ? world.payload.online_by_hour[hour] : 0;

  const live = useMemo(() => {
    if (!replay) return null;
    const open = openOrders(replay, clock);
    const count = { waiting: 0, active: 0, late: 0 };
    for (const o of open) count[o.state]++;
    const modes = [0, 0, 0, 0, 0];
    if (frame) for (let c = 0; c < frame.mode.length; c++) modes[frame.mode[c]]++;
    return { count, modes, online: frame ? frame.mode.length - modes[MODE_OFFLINE] : 0 };
  }, [replay, clock, frame]);

  return (
    <div className="app">
      <aside className="panel">
        <header>
          <p className="eyebrow">delivery-routing-lab</p>
          <h1>Itapipoca, CE</h1>
          <p className="lede">
            As ruas são reais, do OpenStreetMap. Restaurantes, pedidos, entregadores e trânsito são{" "}
            <strong>sintéticos</strong>, gerados pelo simulador.
          </p>
        </header>

        {error && (
          <p className="notice" role="alert">
            {error}
          </p>
        )}

        {world && (
          <section aria-labelledby="clock-title">
            <h2 id="clock-title">Relógio do dia simulado</h2>
            <div className="hour">
              <output className="hour-now" htmlFor="clock">
                {clockLabel(clock)}
              </output>
              <input
                id="clock"
                type="range"
                min={0}
                max={1439}
                step={1}
                value={Math.floor(clock / 60) % 1440}
                onChange={(e) => setClock(Number(e.target.value) * 60)}
                aria-label="Hora do dia simulado"
              />
              <button type="button" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
                {playing ? "Pausar" : "Rodar"}
              </button>
            </div>
            <div className="clock-row">
              <span className="note">Velocidade</span>
              <div className="speed" role="group" aria-label="Velocidade da simulação">
                {SPEEDS.map((s) => (
                  <button key={s} type="button" aria-pressed={speed === s} onClick={() => setSpeed(s)}>
                    {s}×
                  </button>
                ))}
              </div>
            </div>
          </section>
        )}

        {live && scale && (
          <section aria-labelledby="now-title">
            <h2 id="now-title">Agora</h2>
            <dl className="stats compact">
              <div>
                <dt>Entregadores no turno</dt>
                <dd>{int.format(live.online)}</dd>
              </div>
              <div>
                <dt>Restaurantes online</dt>
                <dd>
                  {int.format(online)}
                  <small> de {int.format(scale.active)}</small>
                </dd>
              </div>
              <div>
                <dt>Pedidos em aberto</dt>
                <dd>{int.format(live.count.waiting + live.count.active + live.count.late)}</dd>
              </div>
              <div>
                <dt>Atrasados</dt>
                <dd className={live.count.late ? "late" : undefined}>{int.format(live.count.late)}</dd>
              </div>
            </dl>
            <p className="note">
              {int.format(live.count.waiting)} sem entregador, {int.format(live.count.active)} em andamento.{" "}
              {int.format(live.modes[MODE_TO_PICKUP] + live.modes[MODE_AT_SHOP])} entregadores indo ou esperando
              para coletar, {int.format(live.modes[MODE_TO_DROP])} levando ao cliente.
            </p>
          </section>
        )}

        {replay && selected !== null && frame && (
          <CourierPanel replay={replay} frame={frame} courier={selected} clock={clock} onClose={() => setSelected(null)} />
        )}
        {decision && replay && <DecisionPanel decision={decision} kinds={replay.payload.kinds} />}

        {replay && (
          <section aria-labelledby="policy-title">
            <h2 id="policy-title">Política de despacho</h2>
            <div className="speed policies" role="group" aria-label="Política de despacho">
              {POLICIES.map((id) => (
                <button
                  key={id}
                  type="button"
                  aria-pressed={policy === id}
                  disabled={!replays[id]}
                  onClick={() => choosePolicy(id)}
                >
                  {POLICY_INFO[id].name} <small>{POLICY_INFO[id].role}</small>
                </button>
              ))}
            </div>
            <p className="note">{POLICY_INFO[policy].text}</p>
            <PolicyTable replays={replays} current={policy} />
            <p className="note">
              Dia {replay.payload.meta.day}: {int.format(replay.payload.meta.orders)} pedidos,{" "}
              {int.format(replay.payload.meta.couriers)} entregadores, mochila de {replay.payload.meta.capacity},{" "}
              {replay.payload.meta.rain ? "com chuva" : "sem chuva"}. Um dia só; as médias de 20 dias estão no README.
            </p>
          </section>
        )}

        {replay && world && scale && (
          <section aria-labelledby="layers-title">
            <h2 id="layers-title">Camadas</h2>
            <ul className="legend">
              <li>
                <label>
                  <input type="checkbox" checked={layers.couriers} onChange={() => toggleLayer("couriers")} />
                  <span className="swatch sq pickup" aria-hidden="true" />
                  <span className="label">Entregadores a caminho de coletar</span>
                </label>
              </li>
              <li>
                <label>
                  <span className="check-spacer" aria-hidden="true" />
                  <span className="swatch sq drop" aria-hidden="true" />
                  <span className="label">a caminho de entregar</span>
                </label>
              </li>
              <li>
                <label>
                  <span className="check-spacer" aria-hidden="true" />
                  <span className="swatch sq idle" aria-hidden="true" />
                  <span className="label">parados</span>
                </label>
              </li>
              <li>
                <label>
                  <input type="checkbox" checked={layers.orders} onChange={() => toggleLayer("orders")} />
                  <span className="swatch dot wait" aria-hidden="true" />
                  <span className="label">Pedido sem entregador</span>
                </label>
              </li>
              <li>
                <label>
                  <span className="check-spacer" aria-hidden="true" />
                  <span className="swatch dot order" aria-hidden="true" />
                  <span className="label">em andamento</span>
                </label>
              </li>
              <li>
                <label>
                  <span className="check-spacer" aria-hidden="true" />
                  <span className="swatch dot late" aria-hidden="true" />
                  <span className="label">atrasado</span>
                </label>
              </li>
              <li>
                <label>
                  <input type="checkbox" checked={layers.restaurants} onChange={() => toggleLayer("restaurants")} />
                  <span className="swatch dot shop" aria-hidden="true" />
                  <span className="label">Restaurantes ativos</span>
                  <span className="km">{int.format(scale.active)}</span>
                </label>
              </li>
              <li>
                <label>
                  <input type="checkbox" checked={layers.heat} onChange={() => toggleLayer("heat")} />
                  <span className="swatch heat" aria-hidden="true" />
                  <span className="label">Calor da demanda</span>
                  <span className="km">média</span>
                </label>
              </li>
            </ul>
            <p className="note">
              Clique num entregador para ver a rota do trecho em curso e a mochila dele. Passe o mouse num restaurante
              para ver o perfil de preparo, que o modelo não vê.
            </p>
          </section>
        )}

        {world && scale && (
          <details>
            <summary>Mundo simulado</summary>
            <dl className="stats">
              <div>
                <dt>Cadastrados</dt>
                <dd>{int.format(scale.restaurants)}</dd>
              </div>
              <div>
                <dt>Ativos</dt>
                <dd>{int.format(scale.active)}</dd>
              </div>
              <div>
                <dt>Pico online</dt>
                <dd>{int.format(Math.max(...world.payload.online_by_hour))}</dd>
              </div>
              <div>
                <dt>Pedidos por dia</dt>
                <dd>{int.format(scale.orders_per_day)}</dd>
              </div>
            </dl>
            <div className="heat-legend">
              <div className="ramp" aria-hidden="true" />
              <div className="ramp-labels">
                <span>0</span>
                <span>{one.format(world.heat.max)} pedidos por hora</span>
              </div>
              <p className="note">
                Calor: células de {world.heat.size} m, média de {world.payload.heat.days} dias simulados, mesma escala em
                todas as horas. Semente {scale.seed}.
              </p>
            </div>
          </details>
        )}

        {city && (
          <details>
            <summary>Grafo de ruas e vias</summary>
            <dl className="stats">
              <div>
                <dt>Cruzamentos</dt>
                <dd>{int.format(city.payload.stats.nodes)}</dd>
              </div>
              <div>
                <dt>Trechos (mão única conta 1)</dt>
                <dd>{int.format(city.payload.stats.edges)}</dd>
              </div>
              <div>
                <dt>Semáforos</dt>
                <dd>{int.format(city.payload.stats.signals)}</dd>
              </div>
              <div>
                <dt>Extensão</dt>
                <dd>{km.format(Object.values(city.payload.stats.km_by_class).reduce((a, b) => a + b, 0))} km</dd>
              </div>
            </dl>
            <ul className="legend">
              {city.payload.classes.map((name, i) => (
                <li key={name}>
                  <label>
                    <input type="checkbox" checked={visible[i] ?? true} onChange={() => toggleRoad(i)} />
                    <span className="swatch" style={{ background: `var(--road-${name})` }} aria-hidden="true" />
                    <span className="label">{CLASS_LABEL[name] ?? name}</span>
                    <span className="km">{km.format(city.payload.stats.km_by_class[name])} km</span>
                  </label>
                </li>
              ))}
            </ul>
          </details>
        )}

        {city && (
          <section aria-labelledby="view-title">
            <h2 id="view-title">Vista</h2>
            <div className="row">
              <button type="button" onClick={() => refit("focus")}>
                Centro urbano
              </button>
              <button type="button" onClick={() => refit("all")}>
                Município inteiro
              </button>
            </div>
          </section>
        )}

        <footer>
          <p>Passo 5: modelos próprios de preparo e ETA no B1+. Próximo: o Jev e a explicação da escolha.</p>
        </footer>
      </aside>

      <main>
        {city ? (
          <MapCanvas
            city={city}
            visible={visible}
            fit={fit}
            world={world}
            replay={replay}
            frame={frame}
            clock={clock}
            layers={layers}
            selected={selected}
            onSelect={setSelected}
            marks={marks}
          />
        ) : (
          <div className="map placeholder">{error ? "" : "Carregando as ruas…"}</div>
        )}
      </main>
    </div>
  );
}

function CourierPanel({
  replay,
  frame,
  courier,
  clock,
  onClose,
}: {
  replay: Replay;
  frame: Frame;
  courier: number;
  clock: number;
  onClose: () => void;
}) {
  const o = replay.order;
  const kinds = replay.payload.kinds;
  const shop = replay.payload.orders.kind;
  const bag: { index: number; collected: boolean }[] = [];
  let delivered = 0;
  for (let i = 0; i < o.courier.length; i++) {
    if (o.courier[i] !== courier) continue;
    if (!Number.isNaN(o.delivered[i]) && o.delivered[i] <= clock) delivered++;
    else if (!Number.isNaN(o.assigned[i]) && o.assigned[i] <= clock) {
      bag.push({ index: i, collected: !Number.isNaN(o.picked[i]) && o.picked[i] <= clock });
    }
  }
  return (
    <section aria-labelledby="courier-title">
      <div className="row between">
        <h3 id="courier-title">Entregador {courier + 1}</h3>
        <button type="button" onClick={onClose}>
          Fechar
        </button>
      </div>
      <p className="note">
        Turno das {clockLabel(replay.shiftStart[courier])} às {clockLabel(replay.shiftEnd[courier])}. Agora:{" "}
        {MODE_LABEL[frame.mode[courier]]}. {delivered} entregas até aqui.
      </p>
      {bag.length === 0 ? (
        <p className="note">Mochila vazia.</p>
      ) : (
        <ul className="bag">
          {bag.map(({ index, collected }) => {
            const left = (o.promised[index] - clock) / 60;
            return (
              <li key={index} className={left < 0 ? "late" : undefined}>
                <span>
                  {collected ? "Entregar" : "Coletar"}: {kinds[shop[index]]}
                </span>
                <span>{left >= 0 ? `${Math.round(left)} min para o prazo` : `${Math.round(-left)} min de atraso`}</span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

const COMPARE_ROWS: { key: string; label: string; better: "high" | "low"; digits: number }[] = [
  { key: "on_time_pct", label: "No prazo (%)", better: "high", digits: 1 },
  { key: "late_p90_min", label: "Atraso P90 (min)", better: "low", digits: 1 },
  { key: "delivery_mean_min", label: "Pedido, média (min)", better: "low", digits: 1 },
  { key: "wait_at_restaurant_mean_min", label: "Espera no restaurante (min)", better: "low", digits: 1 },
  { key: "km_per_order", label: "Km por pedido", better: "low", digits: 2 },
];

function PolicyTable({ replays, current }: { replays: Partial<Record<PolicyId, Replay>>; current: PolicyId }) {
  const ids = POLICIES.filter((id) => replays[id]);
  return (
    <table className="compare">
      <thead>
        <tr>
          <th scope="col">
            <span className="sr-only">Métrica</span>
          </th>
          {ids.map((id) => (
            <th key={id} scope="col" className={id === current ? "current" : undefined}>
              {POLICY_INFO[id].name}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {COMPARE_ROWS.map((row) => {
          const values = ids.map((id) => replays[id]!.payload.meta.summary[row.key]);
          const best = row.better === "high" ? Math.max(...values) : Math.min(...values);
          return (
            <tr key={row.key}>
              <th scope="row">{row.label}</th>
              {ids.map((id, i) => (
                <td key={id} className={[id === current ? "current" : "", values[i] === best ? "best" : ""].join(" ").trim() || undefined}>
                  {values[i].toLocaleString("pt-BR", { minimumFractionDigits: row.digits, maximumFractionDigits: row.digits })}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

interface DecisionCandidate {
  pickup: boolean;
  shopKind: number;
  prob: number;
  x: number;
  y: number;
  contrib: number[] | null;
  chosen: boolean;
}

interface DecisionView {
  t: number;
  candidates: DecisionCandidate[];
  labels: string[];
}

const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2).replace(".", ",")}`;

function DecisionPanel({ decision, kinds }: { decision: DecisionView; kinds: string[] }) {
  const stop = (c: DecisionCandidate) => `${c.pickup ? "Coletar" : "Entregar"} (${kinds[c.shopKind]})`;
  const chosen = decision.candidates.find((c) => c.chosen)!;
  const others = decision.candidates.filter((c) => !c.chosen).map((c) => c.prob);
  const margin = chosen.prob - Math.max(...others);
  const rows = chosen.contrib
    ? decision.labels
        .map((label, j) => ({ label, v: chosen.contrib![j] }))
        .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
        .slice(0, 6)
    : [];
  const maxAbs = Math.max(0.01, ...rows.map((r) => Math.abs(r.v)));
  return (
    <section aria-labelledby="why-title">
      <h3 id="why-title">Por que esta parada</h3>
      <p className="note">
        Decisão às {clockLabel(decision.t)}, entre {decision.candidates.length} candidatas. Escolheu{" "}
        <strong>{stop(chosen).toLowerCase()}</strong> com {Math.round(chosen.prob * 100)}% de confiança,{" "}
        {Math.round(margin * 100)} pontos à frente da segunda.
      </p>
      <ul className="cands">
        {decision.candidates.map((c, i) => (
          <li key={i} className={c.chosen ? "chosen" : undefined}>
            <span className="cand-label">{stop(c)}</span>
            <span className="cand-bar" aria-hidden="true">
              <i style={{ width: `${c.prob * 100}%` }} />
            </span>
            <span className="cand-pct">{Math.round(c.prob * 100)}%</span>
          </li>
        ))}
      </ul>
      {rows.length > 0 && (
        <>
          <h4 className="why-title">O que pesou na escolhida</h4>
          <ul className="why">
            {rows.map((r) => (
              <li key={r.label}>
                <span className="why-label">{r.label}</span>
                <span className="why-bar" aria-hidden="true">
                  <i className={r.v >= 0 ? "pos" : "neg"} style={{ width: `${(Math.abs(r.v) / maxAbs) * 100}%` }} />
                </span>
                <span className="why-val">{signed(r.v)}</span>
              </li>
            ))}
          </ul>
          <p className="note">
            Cada barra é quanto o fator puxou esta parada para cima (verde) ou para baixo (vermelho) em relação à
            média das alternativas, na pontuação do modelo. Os fatores andam juntos (atraso e folga, por exemplo),
            então leia o conjunto.
          </p>
        </>
      )}
    </section>
  );
}
