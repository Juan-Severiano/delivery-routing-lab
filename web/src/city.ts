export interface CityPayload {
  origin: [number, number];
  classes: string[];
  bounds: [number, number, number, number];
  focus: { x: number; y: number; radius: number };
  stats: {
    nodes: number;
    edges: number;
    signals: number;
    km_by_class: Record<string, number>;
  };
  geoms: { c: number; p: number[] }[];
}

export interface Street {
  cls: number;
  pts: Float32Array;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface City {
  payload: CityPayload;
  streets: Street[];
}

export const CLASS_LABEL: Record<string, string> = {
  trunk: "Rodovia (trunk)",
  primary: "Via principal",
  secondary: "Via secundária",
  tertiary: "Via terciária",
  unclassified: "Estrada e via sem classe",
  residential: "Rua residencial",
};

export async function loadCity(url = "/data/city.json"): Promise<City> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `não achei ${url} (HTTP ${res.status}). Rode: uv run delivery-lab city build`,
    );
  }
  const payload = (await res.json()) as CityPayload;
  const streets = payload.geoms.map((g): Street => {
    const pts = Float32Array.from(g.p);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < pts.length; i += 2) {
      minX = Math.min(minX, pts[i]);
      maxX = Math.max(maxX, pts[i]);
      minY = Math.min(minY, pts[i + 1]);
      maxY = Math.max(maxY, pts[i + 1]);
    }
    return { cls: g.c, pts, minX, minY, maxX, maxY };
  });
  return { payload, streets };
}

const EARTH_RADIUS_M = 6_371_008.8;

export function toLatLon(origin: [number, number], x: number, y: number): [number, number] {
  const [lat0, lon0] = origin;
  const lat = lat0 + ((y / EARTH_RADIUS_M) * 180) / Math.PI;
  const lon = lon0 + ((x / (EARTH_RADIUS_M * Math.cos((lat0 * Math.PI) / 180))) * 180) / Math.PI;
  return [lat, lon];
}
