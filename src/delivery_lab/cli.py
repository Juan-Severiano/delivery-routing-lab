"""Linha de comando: baixar as ruas, montar o grafo e gerar os arquivos do mapa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import osm
from .city import City, build_city, render_payload


def cmd_fetch(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} já existe; use --force para baixar de novo")
    size = osm.fetch(args.place, out)
    print(f"{size / 1e6:.2f} MB -> {out}  (© OpenStreetMap contributors, ODbL)")


def cmd_build(args: argparse.Namespace) -> None:
    city, report = build_city(osm.read_osm(Path(args.osm)))
    for line in report.lines():
        print(line)
    city.save(Path(args.out))
    print(f"grafo -> {args.out}")
    if args.render:
        render = Path(args.render)
        render.parent.mkdir(parents=True, exist_ok=True)
        render.write_text(json.dumps(render_payload(city), separators=(",", ":")), encoding="utf-8")
        print(f"mapa -> {render} ({render.stat().st_size / 1e6:.2f} MB)")


def cmd_stats(args: argparse.Namespace) -> None:
    city = City.load(Path(args.city))
    print(f"nós: {city.n_nodes}, arestas: {city.n_edges}, semáforos: {int(city.node_signal.sum())}")
    for name, km in city.km_by_class().items():
        print(f"  {name:<13}{km:8.1f} km")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="delivery-lab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    city = sub.add_parser("city", help="ruas e grafo da cidade").add_subparsers(
        dest="city_command", required=True
    )

    fetch = city.add_parser("fetch", help="baixa as ruas do OpenStreetMap (um pedido ao Overpass)")
    fetch.add_argument("--place", default="Itapipoca")
    fetch.add_argument("--out", default="data/raw/itapipoca-roads.json")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    build = city.add_parser("build", help="monta o grafo a partir do JSON do OSM")
    build.add_argument("--osm", default="data/raw/itapipoca-roads.json")
    build.add_argument("--out", default="data/out/city.npz")
    build.add_argument("--render", default="web/public/data/city.json", help="JSON do mapa ('' pula)")
    build.set_defaults(func=cmd_build)

    stats = city.add_parser("stats", help="resumo do grafo montado")
    stats.add_argument("--city", default="data/out/city.npz")
    stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
