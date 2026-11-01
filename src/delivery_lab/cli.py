"""Linha de comando: baixar as ruas, montar o grafo e gerar os arquivos do mapa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import osm, world
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


def cmd_world(args: argparse.Namespace) -> None:
    preset = world.PRESETS[args.preset]
    restaurants = args.restaurants or preset["restaurants"]
    orders = args.orders or preset["orders_per_day"]
    city = City.load(Path(args.city))
    w = world.generate_world(city, args.seed, restaurants, orders, args.radius, preset["active_share"])
    payload = world.render_payload(w, city, days=args.days)
    online = payload["online_by_hour"]
    print(
        f"restaurantes: {w.n_restaurants} cadastrados, {int(w.active.sum())} ativos;"
        f" online por hora: mín {min(online)}, máx {max(online)}"
    )
    day = payload["day"]
    print(f"dia {day['seed_day']}: {day['count']} pedidos ({day['dropped']} sem restaurante dentro de {w.max_radius_m:.0f} m)")
    peak = int(np.argmax(payload["curve_by_hour"]))
    print(f"hora de pico: {peak}h ({payload['curve_by_hour'][peak]:.1%} dos pedidos do dia)")
    out = Path(args.render)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"mundo -> {out} ({out.stat().st_size / 1e6:.2f} MB, calor = média de {args.days} dias)")


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

    wd = sub.add_parser("world", help="restaurantes e demanda sintéticos").add_subparsers(
        dest="world_command", required=True
    )
    build_world = wd.add_parser("build", help="gera restaurantes, curva de demanda e o JSON do mapa")
    build_world.add_argument("--preset", choices=sorted(world.PRESETS), default="plataforma")
    build_world.add_argument("--restaurants", type=int, default=0, help="sobrescreve o preset")
    build_world.add_argument("--orders", type=float, default=0, help="pedidos por dia, sobrescreve o preset")
    build_world.add_argument("--radius", type=float, default=4000.0, help="raio máximo de entrega em metros")
    build_world.add_argument("--seed", type=int, default=7)
    build_world.add_argument("--days", type=int, default=30, help="dias simulados para o calor médio")
    build_world.add_argument("--city", default="data/out/city.npz")
    build_world.add_argument("--render", default="web/public/data/world.json")
    build_world.set_defaults(func=cmd_world)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
