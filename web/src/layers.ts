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
}

export interface LayerColors {
  heat: string;
  shop: string;
  order: string;
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

/** Os pedidos do dia mostrado que caem na hora escolhida. */
export function drawOrders(ctx: CanvasRenderingContext2D, v: View, world: World, hour: number, color: string) {
  const { day } = world.payload;
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  const radius = Math.max(1.6, Math.min(3.2, v.scale * 14));
  for (const i of world.dayByHour[hour]) {
    const x = sx(v, day.x[i]);
    const y = sy(v, day.y[i]);
    if (x < -4 || y < -4 || x > v.w + 4 || y > v.h + 4) continue;
    ctx.moveTo(x + radius, y);
    ctx.arc(x, y, radius, 0, Math.PI * 2);
  }
  ctx.fill();
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
