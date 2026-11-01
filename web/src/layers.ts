import {
  MODE_AT_SHOP,
  MODE_OFFLINE,
  MODE_STOPPED,
  MODE_TO_DROP,
  MODE_TO_PICKUP,
  type Frame,
  type OrderState,
  type Replay,
} from "./replay";
import type { World } from "./world";

export interface View {
  x: number; // metros no centro da tela
  y: number;
  scale: number; // pixels por metro
  w: number;
  h: number;
}

export interface Layers {
  heat: boolean;
  restaurants: boolean;
  orders: boolean;
  couriers: boolean;
}

export interface LayerColors {
  heat: string;
  shop: string;
  ring: string;
  accent: string;
  waiting: string;
  active: string;
  late: string;
  toPickup: string;
  toDrop: string;
  stopped: string;
}

const sx = (v: View, wx: number) => (wx - v.x) * v.scale + v.w / 2;
const sy = (v: View, wy: number) => v.h / 2 - (wy - v.y) * v.scale;

/** Calor da demanda: média de dias simulados por célula, na hora escolhida. */
export function drawHeat(ctx: CanvasRenderingContext2D, v: View, world: World, hour: number, color: string) {
  const { heat } = world;
  const px = heat.size * v.scale;
  const margin = heat.size;
  const left = v.x - v.w / 2 / v.scale - margin;
  const right = v.x + v.w / 2 / v.scale + margin;
  const bottom = v.y - v.h / 2 / v.scale - margin;
  const top = v.y + v.h / 2 / v.scale + margin;
  ctx.fillStyle = color;
  for (let i = 0; i < heat.x.length; i++) {
    const value = heat.v[i * 24 + hour];
    if (value <= 0) continue;
    const cx = heat.x[i];
    const cy = heat.y[i];
    if (cx < left || cx > right || cy < bottom || cy > top) continue;
    // raiz para os valores baixos aparecerem; o máximo vale para todas as horas, então a madrugada fica apagada
    ctx.globalAlpha = Math.min(0.92, Math.sqrt(value / heat.max) * 0.95);
    ctx.fillRect(sx(v, cx) - px / 2, sy(v, cy) - px / 2, px + 0.6, px + 0.6);
  }
  ctx.globalAlpha = 1;
}

export function restaurantRadius(popularity: number): number {
  return Math.min(7, 3.2 + Math.sqrt(popularity) * 1.1);
}

/** Restaurantes ativos: cheios quando online na hora, vazados quando fora do horário. */
export function drawRestaurants(
  ctx: CanvasRenderingContext2D,
  v: View,
  world: World,
  hour: number,
  color: string,
  ring: string,
) {
  const r = world.payload.restaurants;
  const count = r.x.length;
  ctx.lineWidth = 1.4;
  for (let pass = 0; pass < 2; pass++) {
    // primeiro os fechados, depois os online por cima
    for (let i = 0; i < count; i++) {
      if (!r.active[i]) continue;
      const online = world.online[hour * count + i] === 1;
      if (online !== (pass === 1)) continue;
      const x = sx(v, r.x[i]);
      const y = sy(v, r.y[i]);
      if (x < -8 || y < -8 || x > v.w + 8 || y > v.h + 8) continue;
      const radius = restaurantRadius(r.popularity[i]);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (online) {
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = ring;
        ctx.stroke();
      } else {
        ctx.globalAlpha = 0.55;
        ctx.strokeStyle = color;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
  }
}

/** O restaurante ativo mais perto do ponteiro, até `maxPx` de distância. */
export function nearestRestaurant(world: World, v: View, px: number, py: number, maxPx = 10): number | null {
  const r = world.payload.restaurants;
  let best: number | null = null;
  let bestD = maxPx;
  for (let i = 0; i < r.x.length; i++) {
    if (!r.active[i]) continue;
    const d = Math.hypot(sx(v, r.x[i]) - px, sy(v, r.y[i]) - py);
    if (d < bestD) {
      best = i;
      bestD = d;
    }
  }
  return best;
}

/** Pedidos em aberto: sem entregador, em andamento ou já atrasados. */
export function drawLiveOrders(
  ctx: CanvasRenderingContext2D,
  v: View,
  replay: Replay,
  list: { index: number; state: OrderState }[],
  colors: LayerColors,
) {
  const o = replay.order;
  const radius = Math.max(2, Math.min(3.6, v.scale * 16));
  const fill: Record<OrderState, string> = { waiting: colors.waiting, active: colors.active, late: colors.late };
  for (const state of ["active", "waiting", "late"] as OrderState[]) {
    ctx.fillStyle = fill[state];
    ctx.beginPath();
    for (const { index, state: s } of list) {
      if (s !== state) continue;
      const x = sx(v, o.x[index]);
      const y = sy(v, o.y[index]);
      if (x < -6 || y < -6 || x > v.w + 6 || y > v.h + 6) continue;
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
    }
    ctx.fill();
  }
}

/** A rota do trecho em curso do entregador selecionado. */
export function drawRoute(ctx: CanvasRenderingContext2D, v: View, line: Float32Array, color: string) {
  if (line.length < 4) return;
  ctx.beginPath();
  ctx.moveTo(sx(v, line[0]), sy(v, line[1]));
  for (let i = 2; i < line.length; i += 2) ctx.lineTo(sx(v, line[i]), sy(v, line[i + 1]));
  ctx.lineWidth = 4;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.85;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/** Entregadores: quadrados coloridos pelo que estão fazendo; o selecionado ganha um anel. */
export function drawCouriers(
  ctx: CanvasRenderingContext2D,
  v: View,
  frame: Frame,
  selected: number | null,
  colors: LayerColors,
) {
  const size = Math.max(5, Math.min(9, v.scale * 36));
  const half = size / 2;
  const palette: Record<number, string> = {
    [MODE_TO_PICKUP]: colors.toPickup,
    [MODE_TO_DROP]: colors.toDrop,
    [MODE_AT_SHOP]: colors.toPickup,
    [MODE_STOPPED]: colors.stopped,
  };
  ctx.lineWidth = 1.2;
  for (let c = 0; c < frame.mode.length; c++) {
    const mode = frame.mode[c];
    if (mode === MODE_OFFLINE) continue;
    const x = sx(v, frame.x[c]);
    const y = sy(v, frame.y[c]);
    if (x < -10 || y < -10 || x > v.w + 10 || y > v.h + 10) continue;
    ctx.fillStyle = palette[mode];
    ctx.strokeStyle = colors.ring;
    ctx.fillRect(x - half, y - half, size, size);
    ctx.strokeRect(x - half, y - half, size, size);
  }
  if (selected !== null && frame.mode[selected] !== MODE_OFFLINE) {
    const x = sx(v, frame.x[selected]);
    const y = sy(v, frame.y[selected]);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = colors.accent;
    ctx.beginPath();
    ctx.arc(x, y, size + 4, 0, Math.PI * 2);
    ctx.stroke();
  }
}

export function nearestCourier(frame: Frame, v: View, px: number, py: number, maxPx = 12): number | null {
  let best: number | null = null;
  let bestD = maxPx;
  for (let c = 0; c < frame.mode.length; c++) {
    if (frame.mode[c] === MODE_OFFLINE) continue;
    const d = Math.hypot(sx(v, frame.x[c]) - px, sy(v, frame.y[c]) - py);
    if (d < bestD) {
      best = c;
      bestD = d;
    }
  }
  return best;
}

export interface CandidateMark {
  x: number;
  y: number;
  prob: number;
  chosen: boolean;
}

/** As paradas que o Jev estava comparando: o tamanho do anel é a probabilidade, o escolhido é cheio. */
export function drawCandidates(ctx: CanvasRenderingContext2D, v: View, marks: CandidateMark[], colors: LayerColors) {
  ctx.lineWidth = 2;
  ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "middle";
  for (const m of marks) {
    const x = sx(v, m.x);
    const y = sy(v, m.y);
    const r = 8 + 14 * m.prob;
    ctx.strokeStyle = colors.accent;
    ctx.fillStyle = colors.accent;
    ctx.globalAlpha = m.chosen ? 0.28 : 0.1;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = m.chosen ? 1 : 0.6;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = colors.accent;
    ctx.fillText(`${Math.round(m.prob * 100)}%`, x + r + 4, y);
  }
}
