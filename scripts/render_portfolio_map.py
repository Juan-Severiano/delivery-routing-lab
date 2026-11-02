#!/usr/bin/env python3
"""Desenha a malha e uma decisão real do replay em SVG, sem rede ou dependências.

    python3 scripts/render_portfolio_map.py

O desenho usa o pacote demo congelado. O entregador 162 já chegou à coleta às
12:30; o traçado representa a última decisão registrada, às 12:26, não um trecho
ao vivo. A malha é OpenStreetMap e toda a operação é sintética.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import gzip
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1000, 850
COURIER, CLOCK = 161, 12 * 3600 + 30 * 60
INK, ROAD, PRIMARY = "#193341", "#CBD6DD", "#A6B8C7"
ROUTE, ALTERNATIVE, PAPER = "#2452D6", "#B44C24", "#F8FBFD"


def load(name: str, manifest: dict) -> dict:
    entry = next(e for e in manifest["payloads"] if e["name"] == name)
    raw = gzip.decompress((ROOT / "demo" / entry["archive"]).read_bytes())
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"O snapshot não confere: {name}")
    return json.loads(raw)


def render(out: Path) -> None:
    manifest = json.loads((ROOT / "demo/manifest.json").read_text(encoding="utf-8"))
    city, world, replay = (load(name, manifest) for name in ("city.json", "world.json", "replay-jev.json"))
    first, last = replay["couriers"]["first_leg"][COURIER:COURIER + 2]
    leg = bisect_right(replay["legs"]["t0"], CLOCK, first, last) - 1
    decision = replay["decisions"]["leg"].index(leg)
    start, stop = replay["decisions"]["ptr"][decision:decision + 2]
    chosen = replay["decisions"]["chosen"][decision]
    candidates = []
    for index in range(start, stop):
        order = replay["decisions"]["order"][index]
        pickup = replay["decisions"]["kind"][index] == 0
        restaurant = replay["orders"]["restaurant"][order]
        payload = world["restaurants"] if pickup else replay["orders"]
        item = restaurant if pickup else order
        candidates.append({"x": payload["x"][item], "y": payload["y"][item],
                           "probability": replay["decisions"]["prob"][index], "pickup": pickup})
    if len(candidates) != 2 or not all(c["pickup"] for c in candidates):
        raise ValueError("A composição espera as duas coletas desta decisão do snapshot")

    # O mesmo centro urbano do mapa interativo, com um recorte de 5,5 x 4,7 km.
    # A margem superior contém a coleta A; o sul preserva o desenho do centro.
    center_x = city["focus"]["x"] + 120
    center_y = city["focus"]["y"] + 130
    scale = WIDTH / 5500
    half_w, half_h = WIDTH / scale / 2, HEIGHT / scale / 2
    left, right = center_x - half_w, center_x + half_w
    bottom, top = center_y - half_h, center_y + half_h

    def xy(x: float, y: float) -> tuple[float, float]:
        return (x - left) * scale, (top - y) * scale

    def path(points: list[tuple[float, float]]) -> str:
        return "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in (xy(*point) for point in points))

    grouped = [[] for _ in city["classes"]]
    street_count = 0
    for geom in city["geoms"]:
        coordinates = geom["p"]
        xs, ys = coordinates[::2], coordinates[1::2]
        if max(xs) < left or min(xs) > right or max(ys) < bottom or min(ys) > top:
            continue
        grouped[geom["c"]].append(path(list(zip(xs, ys))))
        street_count += 1

    start_edge, end_edge = replay["legs"]["ptr"][leg:leg + 2]
    route = []
    for index in range(start_edge, end_edge):
        edge = replay["leg_edges"][index]
        geom = city["geoms"][replay["edges"]["geom"][edge]]["p"]
        points = list(zip(geom[::2], geom[1::2]))
        if replay["edges"]["rev"][edge]:
            points.reverse()
        route.extend(points if not route else points[1:])
    if not route:
        raise ValueError("O trecho da decisão não contém geometria")
    departure = replay["legs"]["t0"][leg]
    duration = sum(replay["leg_deciseconds"][start_edge:end_edge]) / 10
    if departure + duration > CLOCK:
        raise ValueError("A descrição pressupõe que a coleta já foi alcançada às 12:30")

    title = "Entregador 162: a coleta escolhida pelo Jev"
    description = (
        "Malha real de Itapipoca, derivada do OpenStreetMap. Replay sintético do dia 101, "
        "semente 7, às 12:30. A última decisão foi às 12:26: coleta A com 86,3% e "
        "coleta B com 13,7%. O trecho azul foi concluído antes das 12:30."
    )
    metadata = {"courier_index": COURIER, "clock_s": CLOCK, "leg": leg,
                "decision_s": departure, "leg_duration_s": duration, "chosen_candidate": chosen,
                "candidates": candidates, "extent_m": [left, bottom, right, top],
                "source": "demo/manifest.json", "osm_attribution": "© OpenStreetMap contributors (ODbL 1.0)"}
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        f"<title id=\"title\">{escape(title)}</title>",
        f"<desc id=\"description\">{escape(description)}</desc>",
        f"<metadata>{escape(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}</metadata>",
        f'<defs><clipPath id="urban"><rect width="{WIDTH}" height="{HEIGHT}"/></clipPath></defs>',
        '<g clip-path="url(#urban)" fill="none" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for index in reversed(range(len(grouped))):
        if not grouped[index]:
            continue
        major = city["classes"][index] in ("trunk", "primary")
        width = "1.45" if major else "0.7"
        color = PRIMARY if major else ROAD
        parts.append(f'<path d="{"".join(grouped[index])}" stroke="{color}" stroke-width="{width}" vector-effect="non-scaling-stroke"/>')
    trace = path(route)
    parts.extend([
        f'<path d="{trace}" stroke="{PAPER}" stroke-width="8" vector-effect="non-scaling-stroke"/>',
        f'<path d="{trace}" stroke="{ROUTE}" stroke-width="4" vector-effect="non-scaling-stroke"/>',
        '</g>',
        f'<g font-family="Arial, sans-serif" fill="{INK}">',
    ])
    origin_x, origin_y = xy(*route[0])
    parts.extend([
        f'<circle cx="{origin_x:.1f}" cy="{origin_y:.1f}" r="5" fill="{PAPER}" stroke="{ROUTE}" stroke-width="2.5"/>',
        f'<text x="{origin_x + 14:.1f}" y="{origin_y + 5:.1f}" font-size="13" fill="{PAPER}" stroke="{PAPER}" stroke-width="5" stroke-linejoin="round">Saída às 12:26</text>',
        f'<text x="{origin_x + 14:.1f}" y="{origin_y + 5:.1f}" font-size="13">Saída às 12:26</text>',
    ])
    for index, candidate in enumerate(candidates):
        x, y = xy(candidate["x"], candidate["y"])
        color = ROUTE if index == chosen else ALTERNATIVE
        letter = chr(65 + index)
        percentage = round(candidate["probability"] * 100)
        caption = "coleta escolhida" if index == chosen else "outra coleta"
        parts.extend([
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="23" fill="{PAPER}" fill-opacity="0.88"/>',
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="{color}"/>',
            f'<text x="{x:.1f}" y="{y + 4.7:.1f}" text-anchor="middle" fill="white" font-size="13" font-weight="700">{letter}</text>',
            f'<g transform="translate({x + 25:.1f},{y - 21:.1f})">',
            f'<rect x="-4" y="-2" width="126" height="48" rx="3" fill="{PAPER}" fill-opacity="0.95"/>',
            f'<text x="4" y="19" fill="{color}" font-size="23" font-weight="700">{percentage}%</text>',
            f'<text x="4" y="37" font-size="12">{caption}</text>',
            '</g>',
        ])
    scale_x, scale_y, km = 38, HEIGHT - 63, 1000 * scale
    parts.extend([
        f'<path d="M{scale_x},{scale_y - 6}v6h{km:.2f}v-6" fill="none" stroke="{INK}" stroke-width="1.4"/>',
        f'<text x="{scale_x}" y="{scale_y - 13}" font-size="12">1 km</text>',
        f'<text x="{scale_x}" y="{HEIGHT - 21}" font-size="10" fill="{INK}">© OpenStreetMap contributors · ODbL 1.0</text>',
        f'<text x="{WIDTH - 38}" y="{HEIGHT - 21}" text-anchor="end" font-size="10" fill="{INK}">Replay sintético / dia 101</text>',
        '</g>',
        '</svg>',
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}: {street_count} geometrias, "
          f"trecho {leg}, saída {departure:.1f}s, duração {duration:.1f}s, "
          f"{len(route)} pontos; probabilidades {[c['probability'] for c in candidates]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets/mapa-estudo.svg")
    args = parser.parse_args()
    render(args.out.resolve())


if __name__ == "__main__":
    main()
