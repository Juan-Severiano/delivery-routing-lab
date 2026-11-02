export interface WorldPayload {
  scale: { restaurants: number; active: number; orders_per_day: number; max_radius_m: number; seed: number };
  kinds: string[];
  restaurants: {
    x: number[];
    y: number[];
    kind: number[];
    active: number[];
    popularity: number[];
    open_from: number[];
    open_until: number[];
    prep_mean_min: number[];
  };
  curve_by_hour: number[];
  online_by_hour: number[];
  heat: { cell_m: number; origin: [number, number]; days: number; cells: [number, number, number[]][] };
  day: {
    seed_day: number;
    count: number;
    dropped: number;
    t_min: number[];
    x: number[];
    y: number[];
    restaurant: number[];
  };
}

export interface World {
  payload: WorldPayload;
  heat: {
    size: number;
    x: Float32Array; // centro da célula, em metros
    y: Float32Array;
    v: Float32Array; // pedidos por hora: v[célula * 24 + hora]
    max: number;
    totalByHour: number[]; // pedidos por dia, em média, por hora
  };
  online: Uint8Array; // online[hora * R + restaurante]
  dayByHour: Uint32Array[]; // índices dos pedidos do dia mostrado, por hora
}

function isOnline(from: number, until: number, hour: number): boolean {
  return (from <= hour && hour < until) || (until > 24 && hour + 24 < until);
}

export async function loadWorld(url = `${import.meta.env.BASE_URL}data/world.json`): Promise<World> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`não achei ${url} (HTTP ${res.status}). Rode: uv run delivery-lab world build`);
  }
  const payload = (await res.json()) as WorldPayload;
  const { heat: h, restaurants: r, day } = payload;

  const n = h.cells.length;
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const v = new Float32Array(n * 24);
  const totalByHour = new Array<number>(24).fill(0);
  let max = 0;
  h.cells.forEach(([ix, iy, values], i) => {
    x[i] = h.origin[0] + (ix + 0.5) * h.cell_m;
    y[i] = h.origin[1] + (iy + 0.5) * h.cell_m;
    values.forEach((value, hour) => {
      v[i * 24 + hour] = value;
      totalByHour[hour] += value;
      max = Math.max(max, value);
    });
  });

  const count = r.x.length;
  const online = new Uint8Array(24 * count);
  for (let hour = 0; hour < 24; hour++) {
    for (let i = 0; i < count; i++) {
      if (r.active[i] && isOnline(r.open_from[i], r.open_until[i], hour + 0.5)) online[hour * count + i] = 1;
    }
  }

  const buckets: number[][] = Array.from({ length: 24 }, () => []);
  day.t_min.forEach((t, i) => buckets[Math.min(23, Math.floor(t / 60))].push(i));

  return {
    payload,
    heat: { size: h.cell_m, x, y, v, max, totalByHour },
    online,
    dayByHour: buckets.map((b) => Uint32Array.from(b)),
  };
}

export function hourLabel(hour: number): string {
  return `${String(hour).padStart(2, "0")}h`;
}

export function formatWindow(from: number, until: number): string {
  const f = (t: number) => {
    const h = Math.floor(t) % 24;
    const m = Math.round((t - Math.floor(t)) * 60);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  };
  return `${f(from)} às ${f(until)}`;
}
