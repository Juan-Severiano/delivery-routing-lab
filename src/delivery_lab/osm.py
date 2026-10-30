"""Ruas de uma cidade a partir do OpenStreetMap (Overpass API): baixar e ler o JSON.

Dados © OpenStreetMap contributors (ODbL). O arquivo baixado não é versionado.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "delivery-routing-lab/0.1 (projeto pessoal de portfolio)"

# Só vias onde carro e moto andam; calçadas, trilhas, serviço e pista de corrida ficam de fora.
DRIVABLE = ("trunk", "primary", "secondary", "tertiary", "unclassified", "residential")


@dataclass(frozen=True)
class RawWay:
    id: int
    highway: str  # classe sem o sufixo `_link`
    link: bool
    nodes: list[int]
    oneway: int  # 0 = mão dupla, 1 = só no sentido dos nós, -1 = só no sentido contrário
    maxspeed_kmh: float | None


@dataclass(frozen=True)
class RawOsm:
    coords: dict[int, tuple[float, float]]  # id do nó -> (lat, lon)
    ways: list[RawWay]
    signals: frozenset[int]  # nós com semáforo


def overpass_query(place: str) -> str:
    """Consulta com todas as vias dirigíveis do município `place` e os nós delas."""
    if '"' in place or "\\" in place:
        raise ValueError(f"nome de lugar inválido: {place!r}")
    classes = "|".join(DRIVABLE)
    return (
        "[out:json][timeout:180];\n"
        f'area["name"="{place}"]["boundary"="administrative"]["admin_level"="8"]->.a;\n'
        f'way(area.a)["highway"~"^({classes})(_link)?$"]->.w;\n'
        "(.w; >;);\n"
        "out body qt;"
    )


def fetch(place: str, out: Path) -> int:
    """Baixa as ruas de `place` para `out` e devolve o tamanho em bytes. Um pedido só, sem repetir.

    Escreve num arquivo `.part` e só renomeia se o JSON veio inteiro e sem erro do servidor; uma
    conexão que cai no meio não deixa em `out` um arquivo pela metade.
    """
    data = urllib.parse.urlencode({"data": overpass_query(place)}).encode()
    request = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": USER_AGENT})
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=240) as response, part.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                size += len(chunk)
        remark = json.loads(part.read_text(encoding="utf-8")).get("remark", "")
        if remark:  # o Overpass responde 200 com resultado parcial e o erro neste campo
            raise RuntimeError(f"o Overpass respondeu com erro: {remark}")
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)
    return size


def parse_maxspeed(value: str | None) -> float | None:
    """`40`, `40 km/h` e `30 mph` viram km/h; listas e valores como `BR:urban` viram None."""
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(km/h|kmh|mph)?\s*", value.lower())
    if not match:
        return None
    speed = float(match.group(1))
    return speed * 1.609344 if match.group(2) == "mph" else speed


def parse_oneway(tags: dict[str, str]) -> int:
    value = tags.get("oneway")
    if value in ("yes", "true", "1"):
        return 1
    if value in ("-1", "reverse"):
        return -1
    if value is None and tags.get("junction") in ("roundabout", "circular"):
        return 1
    return 0


def read_osm(path: Path) -> RawOsm:
    elements = json.loads(Path(path).read_text(encoding="utf-8"))["elements"]
    coords: dict[int, tuple[float, float]] = {}
    signals: set[int] = set()
    for element in elements:
        if element["type"] == "node":
            coords[element["id"]] = (element["lat"], element["lon"])
            if element.get("tags", {}).get("highway") == "traffic_signals":
                signals.add(element["id"])
    ways: list[RawWay] = []
    for element in elements:
        if element["type"] != "way":
            continue
        tags = element.get("tags", {})
        highway = tags.get("highway", "")
        link = highway.endswith("_link")
        kind = highway.removesuffix("_link")
        nodes = element["nodes"]
        if kind not in DRIVABLE or len(nodes) < 2 or any(n not in coords for n in nodes):
            continue
        ways.append(
            RawWay(
                id=element["id"],
                highway=kind,
                link=link,
                nodes=nodes,
                oneway=parse_oneway(tags),
                maxspeed_kmh=parse_maxspeed(tags.get("maxspeed")),
            )
        )
    return RawOsm(coords=coords, ways=ways, signals=frozenset(signals))
