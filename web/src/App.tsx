import { useEffect, useState } from "react";
import { CLASS_LABEL, loadCity, type City } from "./city";
import { MapCanvas, type FitRequest } from "./MapCanvas";

const int = new Intl.NumberFormat("pt-BR");
const km = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });

export function App() {
  const [city, setCity] = useState<City | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState<boolean[]>([]);
  const [fit, setFit] = useState<FitRequest>({ target: "focus", nonce: 0 });

  useEffect(() => {
    loadCity()
      .then((c) => {
        setCity(c);
        setVisible(c.payload.classes.map(() => true));
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const refit = (target: FitRequest["target"]) =>
    setFit((f) => ({ target, nonce: f.nonce + 1 }));

  const toggle = (i: number) =>
    setVisible((v) => v.map((on, k) => (k === i ? !on : on)));

  return (
    <div className="app">
      <aside className="panel">
        <header>
          <p className="eyebrow">delivery-routing-lab</p>
          <h1>Itapipoca, CE</h1>
          <p className="lede">
            As ruas são reais, do OpenStreetMap. Restaurantes, pedidos, entregadores e trânsito
            serão <strong>sintéticos</strong>, gerados pelo simulador.
          </p>
        </header>

        {error && (
          <p className="notice" role="alert">
            {error}
          </p>
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
                    {km.format(Object.values(city.payload.stats.km_by_class).reduce((a, b) => a + b, 0))}{" "}
                    km
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
                      <input type="checkbox" checked={visible[i] ?? true} onChange={() => toggle(i)} />
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
          <p>
            Passo 1: a cidade. Próximo: restaurantes, pedidos e o mapa de calor de demanda.
          </p>
        </footer>
      </aside>

      <main>
        {city ? (
          <MapCanvas city={city} visible={visible} fit={fit} />
        ) : (
          <div className="map placeholder">{error ? "" : "Carregando as ruas…"}</div>
        )}
      </main>
    </div>
  );
}
