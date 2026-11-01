import type { City } from "./city";

export interface ReplayPayload {
  meta: {
    policy: string;
    day: number;
    seed: number;
    rain: boolean;
    day_factor: number;
    couriers: number;
    orders: number;
    capacity: number;
    summary: Record<string, number>;
  };
  edges: { geom: number[]; rev: number[] };
  couriers: { shift_start: number[]; shift_end: number[]; x: number[]; y: number[]; first_leg: number[] };
  legs: { t0: number[]; kind: number[]; order: number[]; ptr: number[] };
  leg_edges: number[];
  leg_deciseconds: number[];
  orders: {
    x: number[];
    y: number[];
    restaurant: number[];
    kind: number[];
    created: number[];
    assigned: (number | null)[];
    ready: number[];
    picked: (number | null)[];
    delivered: (number | null)[];
    promised: number[];
    courier: number[];
    rejected: number[];
  };
  kinds: string[];
}

export const MODE_OFFLINE = 0;
export const MODE_TO_PICKUP = 1;
export const MODE_TO_DROP = 2;
export const MODE_AT_SHOP = 3;
export const MODE_STOPPED = 4;

export interface Replay {
  payload: ReplayPayload;
  edgeGeom: Int32Array;
  edgeRev: Uint8Array;
  legEdges: Int32Array;
  legSeconds: Float32Array; // tempo de cada aresta de cada trecho
  legT0: Float64Array;
  legEnd: Float64Array;
  legKind: Uint8Array;
  legPtr: Int32Array;
  firstLeg: Int32Array;
  shiftStart: Float64Array;
  shiftEnd: Float64Array;
  order: {
    x: Float32Array;
    y: Float32Array;
    created: Float64Array;
    assigned: Float64Array; // NaN se nunca
    picked: Float64Array;
    delivered: Float64Array;
    promised: Float64Array;
    courier: Int32Array;
    rejected: Uint8Array;
  };
}

const nan = (list: (number | null)[]) => Float64Array.from(list, (v) => (v === null ? NaN : v));

export async function loadReplay(url: string): Promise<Replay> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`não achei ${url} (HTTP ${res.status}). Rode: uv run delivery-lab sim run --policy b0 --render 'web/public/data/replay-{policy}.json'`);
  const p = (await res.json()) as ReplayPayload;
  const ptr = Int32Array.from(p.legs.ptr);
  const legSeconds = Float32Array.from(p.leg_deciseconds, (d) => d / 10);
  const legT0 = Float64Array.from(p.legs.t0);
  const legEnd = new Float64Array(legT0.length);
  for (let i = 0; i < legT0.length; i++) {
    let total = 0;
    for (let k = ptr[i]; k < ptr[i + 1]; k++) total += legSeconds[k];
    legEnd[i] = legT0[i] + total;
  }
  const o = p.orders;
  return {
    payload: p,
    edgeGeom: Int32Array.from(p.edges.geom),
    edgeRev: Uint8Array.from(p.edges.rev),
    legEdges: Int32Array.from(p.leg_edges),
    legSeconds,
    legT0,
    legEnd,
    legKind: Uint8Array.from(p.legs.kind),
    legPtr: ptr,
    firstLeg: Int32Array.from(p.couriers.first_leg),
    shiftStart: Float64Array.from(p.couriers.shift_start),
    shiftEnd: Float64Array.from(p.couriers.shift_end),
    order: {
      x: Float32Array.from(o.x),
      y: Float32Array.from(o.y),
      created: Float64Array.from(o.created),
      assigned: nan(o.assigned),
      picked: nan(o.picked),
      delivered: nan(o.delivered),
      promised: Float64Array.from(o.promised),
      courier: Int32Array.from(o.courier),
      rejected: Uint8Array.from(o.rejected),
    },
  };
}

// --- geometria das arestas ---------------------------------------------------------------------

const cumulativeCache = new Map<number, Float32Array>();

function cumulative(city: City, geom: number): Float32Array {
  let cum = cumulativeCache.get(geom);
  if (!cum) {
    const pts = city.streets[geom].pts;
    cum = new Float32Array(pts.length / 2);
    for (let i = 1; i < cum.length; i++) {
      cum[i] = cum[i - 1] + Math.hypot(pts[2 * i] - pts[2 * i - 2], pts[2 * i + 1] - pts[2 * i - 1]);
    }
    cumulativeCache.set(geom, cum);
  }
  return cum;
}

/** Ponto da aresta a uma fração f (0 a 1) do caminho, no sentido em que ela é percorrida. */
function pointOnEdge(city: City, geom: number, rev: boolean, f: number, out: [number, number]) {
  const pts = city.streets[geom].pts;
  const cum = cumulative(city, geom);
  const total = cum[cum.length - 1];
  const d = (rev ? 1 - f : f) * total;
  let i = 1;
  while (i < cum.length - 1 && cum[i] < d) i++;
  const seg = cum[i] - cum[i - 1] || 1;
  const u = Math.min(1, Math.max(0, (d - cum[i - 1]) / seg));
  out[0] = pts[2 * i - 2] + (pts[2 * i] - pts[2 * i - 2]) * u;
  out[1] = pts[2 * i - 1] + (pts[2 * i + 1] - pts[2 * i - 1]) * u;
}

function legEndPoint(r: Replay, city: City, courier: number, leg: number, out: [number, number]) {
  for (let i = leg; i >= r.firstLeg[courier]; i--) {
    if (r.legPtr[i + 1] > r.legPtr[i]) {
      const e = r.legEdges[r.legPtr[i + 1] - 1];
      pointOnEdge(city, r.edgeGeom[e], r.edgeRev[e] === 1, 1, out);
      return;
    }
  }
  out[0] = r.payload.couriers.x[courier];
  out[1] = r.payload.couriers.y[courier];
}

export interface Frame {
  x: Float32Array;
  y: Float32Array;
  mode: Uint8Array;
  leg: Int32Array; // trecho em curso ou o último concluído; -1 se nenhum
}

export function newFrame(count: number): Frame {
  return {
    x: new Float32Array(count),
    y: new Float32Array(count),
    mode: new Uint8Array(count),
    leg: new Int32Array(count).fill(-1),
  };
}

const scratch: [number, number] = [0, 0];
const LINGER_S = 120; // o entregador some do mapa 2 min depois da última entrega, se o turno acabou

/** Onde está cada entregador no instante `t` (segundos desde a meia-noite). */
export function computeFrame(r: Replay, city: City, t: number, f: Frame): Frame {
  const count = r.shiftStart.length;
  for (let c = 0; c < count; c++) {
    const a = r.firstLeg[c];
    const b = r.firstLeg[c + 1];
    // último trecho que já começou
    let lo = a;
    let hi = b;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (r.legT0[mid] <= t) lo = mid + 1;
      else hi = mid;
    }
    const i = lo - 1;
    f.leg[c] = i >= a ? i : -1;

    if (t < r.shiftStart[c]) {
      f.mode[c] = MODE_OFFLINE;
      continue;
    }
    if (i < a) {
      f.x[c] = r.payload.couriers.x[c];
      f.y[c] = r.payload.couriers.y[c];
      f.mode[c] = MODE_STOPPED;
      continue;
    }
    if (t < r.legEnd[i]) {
      let elapsed = t - r.legT0[i];
      let placed = false;
      for (let k = r.legPtr[i]; k < r.legPtr[i + 1]; k++) {
        const dt = r.legSeconds[k];
        if (elapsed <= dt) {
          const e = r.legEdges[k];
          pointOnEdge(city, r.edgeGeom[e], r.edgeRev[e] === 1, dt > 0 ? elapsed / dt : 1, scratch);
          placed = true;
          break;
        }
        elapsed -= dt;
      }
      if (!placed) legEndPoint(r, city, c, i, scratch);
      f.x[c] = scratch[0];
      f.y[c] = scratch[1];
      f.mode[c] = r.legKind[i] === 0 ? MODE_TO_PICKUP : MODE_TO_DROP;
      continue;
    }
    legEndPoint(r, city, c, i, scratch);
    f.x[c] = scratch[0];
    f.y[c] = scratch[1];
    if (i === b - 1 && t > r.shiftEnd[c] && t > r.legEnd[i] + LINGER_S) f.mode[c] = MODE_OFFLINE;
    else f.mode[c] = r.legKind[i] === 0 ? MODE_AT_SHOP : MODE_STOPPED;
  }
  return f;
}

/** Pontos (x, y, x, y, ...) do trecho `leg`, seguindo as ruas. */
export function legPolyline(r: Replay, city: City, leg: number): Float32Array {
  const out: number[] = [];
  for (let k = r.legPtr[leg]; k < r.legPtr[leg + 1]; k++) {
    const e = r.legEdges[k];
    const pts = city.streets[r.edgeGeom[e]].pts;
    const n = pts.length / 2;
    for (let j = 0; j < n; j++) {
      const idx = r.edgeRev[e] === 1 ? n - 1 - j : j;
      if (out.length >= 2 && j === 0) continue; // o ponto inicial repete o final do trecho anterior
      out.push(pts[2 * idx], pts[2 * idx + 1]);
    }
  }
  return Float32Array.from(out);
}

export type OrderState = "waiting" | "active" | "late";

/** Pedidos em aberto em `t` e o estado de cada um. */
export function openOrders(r: Replay, t: number): { index: number; state: OrderState }[] {
  const o = r.order;
  const out: { index: number; state: OrderState }[] = [];
  for (let i = 0; i < o.created.length; i++) {
    if (o.created[i] > t || o.rejected[i] === 1) continue;
    const delivered = o.delivered[i];
    if (!Number.isNaN(delivered) && delivered <= t) continue;
    const late = t > o.promised[i];
    const assigned = !Number.isNaN(o.assigned[i]) && o.assigned[i] <= t;
    out.push({ index: i, state: late ? "late" : assigned ? "active" : "waiting" });
  }
  return out;
}

export function clockLabel(seconds: number): string {
  const total = Math.floor(seconds / 60) % 1440;
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}
