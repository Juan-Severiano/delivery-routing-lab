import { useEffect, useState } from "react";
import { CLASS_LABEL, loadCity, type City } from "./city";
import type { Layers } from "./layers";
import { MapCanvas, type FitRequest } from "./MapCanvas";
import { hourLabel, loadWorld, type World } from "./world";

const int = new Intl.NumberFormat("pt-BR");
const km = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const one = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });

export function App() {
  const [city, setCity] = useState<City | null>(null);
  const [world, setWorld] = useState<World | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState<boolean[]>([]);
  const [fit, setFit] = useState<FitRequest>({ target: "focus", nonce: 0 });
  const [hour, setHour] = useState(19);
  const [playing, setPlaying] = useState(false);
  const [layers, setLayers] = useState<Layers>({ heat: true, restaurants: true, orders: true });

  useEffect(() => {
    loadCity()
      .then((c) => {
        setCity(c);
        setVisible(c.payload.classes.map(() => true));
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    loadWorld()
      .then(setWorld)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setHour((h) => (h + 1) % 24), 900);
    return () => window.clearInterval(id);
  }, [playing]);

  const refit = (target: FitRequest["target"]) => setFit((f) => ({ target, nonce: f.nonce + 1 }));
  const toggleRoad = (i: number) => setVisible((v) => v.map((on, k) => (k === i ? !on : on)));
  const toggleLayer = (name: keyof Layers) => setLayers((l) => ({ ...l, [name]: !l[name] }));

  const scale = world?.payload.scale;
  const online = world ? world.payload.online_by_hour[hour] : 0;
  const peakOnline = world ? Math.max(...world.payload.online_by_hour) : 0;

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

        {world && scale && (
          <>
            <section aria-labelledby="hour-title">
              <h2 id="hour-title">Hora do dia</h2>
              <div className="hour">
                <output className="hour-now" htmlFor="hour">
                  {hourLabel(hour)}
                </output>
                <input
                  id="hour"
                  type="range"
                  min={0}
                  max={23}
                  step={1}
                  value={hour}
                  onChange={(e) => setHour(Number(e.target.value))}
                  aria-label="Hora do dia"
                />
                <button type="button" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
                  {playing ? "Pausar" : "Rodar o dia"}
                </button>
              </div>
              <dl className="stats compact">
                <div>
                  <dt>Restaurantes online</dt>
                  <dd>
                    {int.format(online)}
                    <small> de {int.format(scale.active)}</small>
                  </dd>
                </div>
                <div>
                  <dt>Pedidos nesta hora</dt>
                  <dd>{int.format(world.dayByHour[hour].length)}</dd>
                </div>
                <div className="wide">
                  <dt>Média de {world.payload.heat.days} dias nesta hora</dt>
                  <dd>
                    {one.format(world.heat.totalByHour[hour])}
                    <small> pedidos</small>
                  </dd>
                </div>
              </dl>
            </section>

            <section aria-labelledby="layers-title">
              <h2 id="layers-title">Camadas</h2>
              <ul className="legend">
                <li>
                  <label>
                    <input type="checkbox" checked={layers.heat} onChange={() => toggleLayer("heat")} />
                    <span className="swatch heat" aria-hidden="true" />
                    <span className="label">Calor da demanda</span>
                    <span className="km">média</span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={layers.orders}
                      onChange={() => toggleLayer("orders")}
                    />
                    <span className="swatch dot order" aria-hidden="true" />
                    <span className="label">Pedidos do dia {world.payload.day.seed_day}</span>
                    <span className="km">{int.format(world.payload.day.count)}</span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={layers.restaurants}
                      onChange={() => toggleLayer("restaurants")}
                    />
                    <span className="swatch dot shop" aria-hidden="true" />
                    <span className="label">Restaurantes ativos</span>
                    <span className="km">{int.format(scale.active)}</span>
                  </label>
                </li>
              </ul>
              <div className="heat-legend">
                <div className="ramp" aria-hidden="true" />
                <div className="ramp-labels">
                  <span>0</span>
                  <span>{one.format(world.heat.max)} pedidos por hora</span>
                </div>
                <p className="note">
                  Células de {world.heat.size} m, média de {world.payload.heat.days} dias simulados. A escala é a
                  mesma em todas as horas, então a madrugada fica apagada. Passe o mouse num restaurante para ver
                  o perfil dele.
                </p>
              </div>
            </section>

            <section aria-labelledby="world-title">
              <h2 id="world-title">Mundo simulado</h2>
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
                  <dd>{int.format(peakOnline)}</dd>
                </div>
                <div>
                  <dt>Pedidos por dia</dt>
                  <dd>{int.format(scale.orders_per_day)}</dd>
                </div>
              </dl>
              <p className="note">
                Média de pedidos por dia; no dia {world.payload.day.seed_day} foram{" "}
                {int.format(world.payload.day.count)}, e {int.format(world.payload.day.dropped)} não acharam
                restaurante online num raio de {km.format(scale.max_radius_m / 1000)} km (quase todos de madrugada).
                Semente {scale.seed}.
              </p>
            </section>
          </>
        )}

        {city && (
          <>
            <section aria-labelledby="graph-title">
              <h2 id="graph-title">Grafo de ruas</h2>
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
                  <dd>
                    {km.format(Object.values(city.payload.stats.km_by_class).reduce((a, b) => a + b, 0))} km
                  </dd>
                </div>
              </dl>
            </section>

            <section aria-labelledby="roads-title">
              <h2 id="roads-title">Vias</h2>
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
            </section>

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
          </>
        )}

        <footer>
          <p>Passo 2: restaurantes e demanda. Próximo: pedidos e entregadores andando pelas ruas.</p>
        </footer>
      </aside>

      <main>
        {city ? (
          <MapCanvas city={city} visible={visible} fit={fit} world={world} hour={hour} layers={layers} />
        ) : (
          <div className="map placeholder">{error ? "" : "Carregando as ruas…"}</div>
        )}
      </main>
    </div>
  );
}
