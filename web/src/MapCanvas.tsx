import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toLatLon, type City, type Street } from "./city";

interface Camera {
  x: number; // metros, centro da tela
  y: number;
  scale: number; // pixels por metro
}

export interface FitRequest {
  target: "focus" | "all";
  nonce: number;
}

interface Props {
  city: City;
  visible: boolean[];
  fit: FitRequest;
}

// Largura real da rua em metros (some quando o zoom é pequeno) e limites em pixels.
const ROAD_WIDTH_M: Record<string, number> = {
  trunk: 16,
  primary: 14,
  secondary: 11,
  tertiary: 9,
  unclassified: 6,
  residential: 7,
};
const ROAD_MIN_PX: Record<string, number> = {
  trunk: 1.6,
  primary: 1.4,
  secondary: 1.2,
  tertiary: 1,
  unclassified: 0.7,
  residential: 0.8,
};
const ROAD_MAX_PX = 14;
const MAX_SCALE = 4; // px por metro
const SCALE_BAR_STEPS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function readPalette(classes: string[]) {
  const css = getComputedStyle(document.documentElement);
  const get = (name: string) => css.getPropertyValue(name).trim();
  return {
    roads: classes.map((c) => get(`--road-${c}`)),
    text: get("--text"),
    scale: get("--scale-bar"),
  };
}

export function MapCanvas({ city, visible, fit }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const cam = useRef<Camera>({ x: 0, y: 0, scale: 0.05 });
  const size = useRef({ w: 1, h: 1, dpr: 1 });
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const frame = useRef(0);
  const visibleRef = useRef(visible);
  const palette = useRef(readPalette(city.payload.classes));
  const [cursor, setCursor] = useState<[number, number] | null>(null);

  const { classes, bounds, focus } = city.payload;

  // ruas agrupadas por classe, para um traçado por classe
  const byClass = useMemo(() => {
    const groups: Street[][] = classes.map(() => []);
    for (const s of city.streets) groups[s.cls].push(s);
    return groups;
  }, [city, classes]);

  const draw = useCallback(() => {
    frame.current = 0;
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const { w, h, dpr } = size.current;
    const { x, y, scale } = cam.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const left = x - w / 2 / scale;
    const right = x + w / 2 / scale;
    const bottom = y - h / 2 / scale;
    const top = y + h / 2 / scale;
    const sx = (wx: number) => (wx - x) * scale + w / 2;
    const sy = (wy: number) => h / 2 - (wy - y) * scale;

    // da via menor para a maior, para as principais ficarem por cima
    for (let cls = classes.length - 1; cls >= 0; cls--) {
      if (!visibleRef.current[cls]) continue;
      const name = classes[cls];
      ctx.beginPath();
      for (const s of byClass[cls]) {
        if (s.maxX < left || s.minX > right || s.maxY < bottom || s.minY > top) continue;
        const p = s.pts;
        ctx.moveTo(sx(p[0]), sy(p[1]));
        for (let i = 2; i < p.length; i += 2) ctx.lineTo(sx(p[i]), sy(p[i + 1]));
      }
      ctx.lineWidth = clamp(ROAD_WIDTH_M[name] * scale, ROAD_MIN_PX[name], ROAD_MAX_PX);
      ctx.strokeStyle = palette.current.roads[cls];
      ctx.stroke();
    }

    drawScaleBar(ctx, h, scale, palette.current.scale, palette.current.text);
  }, [byClass, classes]);

  const requestDraw = useCallback(() => {
    if (!frame.current) frame.current = requestAnimationFrame(draw);
  }, [draw]);

  const zoomAt = useCallback(
    (px: number, py: number, factor: number) => {
      const { w, h } = size.current;
      const c = cam.current;
      const wx = c.x + (px - w / 2) / c.scale;
      const wy = c.y - (py - h / 2) / c.scale;
      const next = clamp(c.scale * factor, minScale(w, h, bounds), MAX_SCALE);
      c.scale = next;
      c.x = wx - (px - w / 2) / next;
      c.y = wy + (py - h / 2) / next;
      requestDraw();
    },
    [bounds, requestDraw],
  );

  const panBy = useCallback(
    (dx: number, dy: number) => {
      cam.current.x -= dx / cam.current.scale;
      cam.current.y += dy / cam.current.scale;
      requestDraw();
    },
    [requestDraw],
  );

  const applyFit = useCallback(
    (target: FitRequest["target"]) => {
      const { w, h } = size.current;
      if (target === "focus") {
        cam.current = {
          x: focus.x,
          y: focus.y,
          scale: Math.min(w, h) / (2 * focus.radius * 1.15),
        };
      } else {
        const bw = bounds[2] - bounds[0];
        const bh = bounds[3] - bounds[1];
        cam.current = {
          x: (bounds[0] + bounds[2]) / 2,
          y: (bounds[1] + bounds[3]) / 2,
          scale: Math.min(w / bw, h / bh) * 0.94,
        };
      }
      requestDraw();
    },
    [bounds, focus, requestDraw],
  );

  // tamanho do canvas acompanha o container (e o devicePixelRatio)
  useEffect(() => {
    const el = wrap.current;
    const cv = canvas.current;
    if (!el || !cv) return;
    let first = true;
    const observer = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, el.clientWidth);
      const h = Math.max(1, el.clientHeight);
      size.current = { w, h, dpr };
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      if (first) {
        first = false;
        applyFit("focus");
      } else {
        requestDraw();
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [applyFit, requestDraw]);

  // "Centro urbano" e "Município inteiro" na barra lateral
  const lastNonce = useRef(fit.nonce);
  useEffect(() => {
    if (lastNonce.current === fit.nonce) return;
    lastNonce.current = fit.nonce;
    applyFit(fit.target);
  }, [fit, applyFit]);

  useEffect(() => {
    visibleRef.current = visible;
    requestDraw();
  }, [visible, requestDraw]);

  // tema claro/escuro: relê as cores do CSS
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      palette.current = readPalette(classes);
      requestDraw();
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [classes, requestDraw]);

  // a roda precisa de listener não passivo para impedir a página de rolar
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  useEffect(
    () => () => {
      cancelAnimationFrame(frame.current);
      frame.current = 0; // sem isso o StrictMode deixa um id velho e nunca mais desenha
    },
    [],
  );

  const local = (e: React.PointerEvent) => {
    const r = wrap.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, local(e));
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const p = local(e);
    const { w, h } = size.current;
    const { x, y, scale } = cam.current;
    setCursor(toLatLon(city.payload.origin, x + (p.x - w / 2) / scale, y - (p.y - h / 2) / scale));
    const prev = pointers.current.get(e.pointerId);
    if (!prev) return;
    if (pointers.current.size === 1) {
      panBy(p.x - prev.x, p.y - prev.y);
    } else if (pointers.current.size === 2) {
      const other = [...pointers.current.entries()].find(([id]) => id !== e.pointerId)![1];
      const before = Math.hypot(prev.x - other.x, prev.y - other.y);
      const after = Math.hypot(p.x - other.x, p.y - other.y);
      const mid = { x: (p.x + other.x) / 2, y: (p.y + other.y) / 2 };
      const midBefore = { x: (prev.x + other.x) / 2, y: (prev.y + other.y) / 2 };
      panBy(mid.x - midBefore.x, mid.y - midBefore.y);
      if (before > 0) zoomAt(mid.x, mid.y, after / before);
    }
    pointers.current.set(e.pointerId, p);
  };

  const onPointerEnd = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const { w, h } = size.current;
    const step = 80;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [step, 0],
      ArrowRight: [-step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    if (moves[e.key]) {
      e.preventDefault();
      panBy(...moves[e.key]);
    } else if (e.key === "+" || e.key === "=") {
      zoomAt(w / 2, h / 2, 1.5);
    } else if (e.key === "-") {
      zoomAt(w / 2, h / 2, 1 / 1.5);
    }
  };

  const zoomButton = (factor: number) => () => {
    const { w, h } = size.current;
    zoomAt(w / 2, h / 2, factor);
  };

  return (
    <div
      ref={wrap}
      className="map"
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onPointerLeave={() => setCursor(null)}
      onDoubleClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        zoomAt(e.clientX - r.left, e.clientY - r.top, 2);
      }}
      onKeyDown={onKeyDown}
    >
      <canvas
        ref={canvas}
        role="img"
        aria-label="Mapa das ruas de Itapipoca. Arraste para mover, role para aproximar; setas e + e - funcionam com o mapa focado."
      />
      <div className="map-zoom" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
        <button type="button" aria-label="Aproximar" onClick={zoomButton(1.6)}>
          +
        </button>
        <button type="button" aria-label="Afastar" onClick={zoomButton(1 / 1.6)}>
          −
        </button>
      </div>
      {cursor && (
        <div className="map-readout" aria-hidden="true">
          {cursor[0].toFixed(5)}, {cursor[1].toFixed(5)}
        </div>
      )}
      <a
        className="map-credit"
        href="https://www.openstreetmap.org/copyright"
        target="_blank"
        rel="noreferrer"
        onPointerDown={(e) => e.stopPropagation()}
      >
        © OpenStreetMap contributors
      </a>
    </div>
  );
}

function minScale(w: number, h: number, bounds: [number, number, number, number]) {
  return Math.min(w / (bounds[2] - bounds[0]), h / (bounds[3] - bounds[1])) * 0.5;
}

function drawScaleBar(
  ctx: CanvasRenderingContext2D,
  h: number,
  scale: number,
  color: string,
  textColor: string,
) {
  const length = SCALE_BAR_STEPS.find((m) => m * scale >= 70) ?? SCALE_BAR_STEPS.at(-1)!;
  const px = length * scale;
  const x0 = 16;
  const y0 = h - 16;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, y0 - 5);
  ctx.lineTo(x0, y0);
  ctx.lineTo(x0 + px, y0);
  ctx.lineTo(x0 + px, y0 - 5);
  ctx.stroke();
  ctx.fillStyle = textColor;
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "bottom";
  ctx.fillText(length >= 1000 ? `${length / 1000} km` : `${length} m`, x0, y0 - 8);
}
