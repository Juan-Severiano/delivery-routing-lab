"""Linha de comando: baixar as ruas, montar o grafo e gerar os arquivos do mapa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import time

from . import osm, paths, planner, replay, sim, world
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


def cmd_paths(args: argparse.Namespace) -> None:
    city = City.load(Path(args.city))
    start = time.perf_counter()
    dist, pred = paths.apsp_python(city)
    elapsed = time.perf_counter() - start
    paths.Paths(city, dist, pred).save(Path(args.out))
    print(f"caminhos mínimos de {city.n_nodes} origens em {elapsed:.1f} s (Python puro, {elapsed / city.n_nodes * 1000:.1f} ms por origem)")
    print(f"-> {args.out} ({dist.nbytes / 1e6:.0f} MB de tempos + {pred.nbytes / 1e6:.0f} MB de predecessoras)")


def _setup(args: argparse.Namespace):
    preset = world.PRESETS[args.preset]
    city = City.load(Path(args.city))
    w = world.generate_world(
        city, args.seed, preset["restaurants"], preset["orders_per_day"], 4000.0, preset["active_share"]
    )
    fleet = sim.generate_fleet(w, city, args.couriers or preset["couriers"], args.seed)
    return city, w, paths.Paths.load(city, Path(args.paths)), fleet


def cmd_sim(args: argparse.Namespace) -> None:
    city, w, pth, fleet = _setup(args)
    def learned(prep: bool, eta_on: bool):
        from .learn.eta import EtaModel
        from .learn.prep import PrepModel

        models = Path(args.models)
        return planner.Planner(
            prep_model=PrepModel.load(models / "prep.npz") if prep else None,
            eta_model=EtaModel.load(models / "eta.npz") if eta_on else None,
            prep_quantile=args.prep_quantile,
            depart_margin_s=args.margin,
        )

    policy = {
        "b0": sim.NearestPolicy,
        "b1": lambda: planner.Planner(depart_margin_s=args.margin),
        "b2": lambda: planner.Planner(oracle=True),
        "b1-preparo": lambda: planner.Planner(know_ready=True),
        "b1-transito": lambda: planner.Planner(know_traffic=True),
        "b1+preparo": lambda: learned(True, False),
        "b1+eta": lambda: learned(False, True),
        "b1+": lambda: learned(True, True),
    }[args.policy]()
    keys = ("on_time_pct", "late_mean_min", "late_p90_min", "delivery_mean_min", "delivery_p90_min",
            "wait_for_courier_mean_min", "wait_at_restaurant_mean_min", "km_per_order", "courier_utilization_pct",
            "rejected_pct")
    rows, elapsed, events = [], [], 0
    for day in range(args.first_day, args.first_day + args.days):
        start = time.perf_counter()
        result = sim.run_day(city, w, pth, fleet, policy, day)
        elapsed.append(time.perf_counter() - start)
        events += result.elapsed_events
        rows.append(sim.summarize(result))
    print(f"{policy.name}: {len(fleet)} entregadores, {args.days} dias (média ± desvio entre dias)")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"  {k:<30}{v.mean():8.2f} ± {v.std():.2f}")
    unserved = sum(r["unserved"] for r in rows)
    print(f"  pedidos sem entrega: {unserved} de {sum(r['orders'] for r in rows)}")
    print(f"  tempo: {np.mean(elapsed):.2f} s por dia simulado ({events / sum(elapsed):,.0f} eventos por segundo)")
    if getattr(policy, "decision_s", None):
        d = np.array(policy.decision_s) * 1000.0
        print(f"  planejar um pedido (assign): média {d.mean():.2f} ms, P99 {np.percentile(d, 99):.2f} ms, {len(d)} decisões")
    if args.render:
        result = sim.run_day(city, w, pth, fleet, policy, args.render_day, record=True)
        payload = replay.render_replay(result, city, w, policy.name, args.render_day)
        out = Path(args.render.replace("{policy}", policy.name.lower().replace("+", "plus")))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"replay do dia {args.render_day} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


def _world(args: argparse.Namespace):
    preset = world.PRESETS[args.preset]
    city = City.load(Path(args.city))
    return city, world.generate_world(
        city, args.seed, preset["restaurants"], preset["orders_per_day"], 4000.0, preset["active_share"]
    )


def cmd_train_prep(args: argparse.Namespace) -> None:
    from .learn import prep  # só o treino precisa do torch

    city, w = _world(args)
    start = time.perf_counter()
    train = prep.build_prep_set(city, w, range(21, 61))
    val = prep.build_prep_set(city, w, range(61, 71))
    test = prep.build_prep_set(city, w, range(101, 121))
    print(f"dados: {len(train)} pedidos de treino (dias 21-60), {len(val)} de validação (61-70), {len(test)} de teste (101-120)")
    model, history = prep.train_prep(w, train, val, epochs=args.epochs, seed=args.seed)
    metrics = prep.evaluate_prep(model, test, train)
    metrics["training_s"] = time.perf_counter() - start
    out = Path(args.out)
    model.save(out)
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"melhor época {history.best_epoch}, {metrics['training_s']:.0f} s no total -> {out}")
    print("erro absoluto médio do preparo nos dias de teste (min):")
    for name, v in metrics["mae_min"].items():
        print(f"  {name:<24}{v:6.2f}")
    print(f"NLL: modelo {metrics['nll']['modelo']:.4f}, verdade do simulador {metrics['nll']['verdade_do_simulador']:.4f}")
    cal = metrics["calibration"]
    print("calibração (quantil previsto -> fração observada): " + ", ".join(f"{q:.1f}->{c:.2f}" for q, c in zip(cal["quantiles"], cal["coverage"])))
    print(f"  erro médio de calibração {cal['mean_abs_error']:.3f}; desvio do log: modelo {metrics['sigma']['modelo_medio']:.3f}, verdade {metrics['sigma']['verdade_media']:.3f}")


def cmd_train_eta(args: argparse.Namespace) -> None:
    from .learn import eta

    city, w = _world(args)
    pth = paths.Paths.load(city, Path(args.paths))
    fleet = sim.generate_fleet(w, city, world.PRESETS[args.preset]["couriers"], args.seed)
    start = time.perf_counter()
    train = eta.collect_legs(city, w, pth, fleet, range(21, 61))
    val = eta.collect_legs(city, w, pth, fleet, range(61, 71))
    test = eta.collect_legs(city, w, pth, fleet, range(101, 121))
    print(f"dados: {len(train)} trechos de treino (dias 21-60), {len(val)} de validação (61-70), {len(test)} de teste (101-120)")
    model, history = eta.train_eta(city, train, val, epochs=args.epochs, seed=args.seed)
    metrics = eta.evaluate_eta(city, model, test, train)
    metrics["training_s"] = time.perf_counter() - start
    out = Path(args.out)
    model.save(out)
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"melhor época {history.best_epoch}, {metrics['training_s']:.0f} s no total -> {out}")
    print("erro do tempo de um trecho nos dias de teste (todos os trechos | só horas de pico):")
    for name in ("tempo_livre", "tabela_por_hora", "modelo", "verdade_sem_ruido_de_viagem"):
        a, b = metrics[name]["todos"], metrics[name]["pico"]
        print(f"  {name:<28} MAE {a['mae_s']:6.1f} s ({a['mape_pct']:4.1f}%, viés {a['bias_pct']:+5.1f}%) | pico {b['mae_s']:6.1f} s ({b['mape_pct']:4.1f}%, viés {b['bias_pct']:+5.1f}%)")


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

    pt = sub.add_parser("paths", help="caminhos mínimos entre todos os cruzamentos").add_subparsers(
        dest="paths_command", required=True
    )
    build_paths = pt.add_parser("build", help="calcula e guarda os caminhos mínimos (tempo livre)")
    build_paths.add_argument("--city", default="data/out/city.npz")
    build_paths.add_argument("--out", default="data/out/paths")
    build_paths.set_defaults(func=cmd_paths)

    sm = sub.add_parser("sim", help="simula a frota").add_subparsers(dest="sim_command", required=True)
    run_sim = sm.add_parser("run", help="roda dias simulados com uma política e mostra as métricas")
    run_sim.add_argument("--policy", choices=["b0", "b1", "b2", "b1-preparo", "b1-transito", "b1+preparo", "b1+eta", "b1+"], default="b0")
    run_sim.add_argument("--models", default="models", help="pasta dos modelos aprendidos (b1+...)")
    run_sim.add_argument("--prep-quantile", type=float, default=planner.PREP_QUANTILE, help="quantil do preparo previsto que o b1+ usa")
    run_sim.add_argument("--margin", type=float, default=None, help="margem de saída em segundos (padrão do planejador)")
    run_sim.add_argument("--preset", choices=sorted(world.PRESETS), default="plataforma")
    run_sim.add_argument("--couriers", type=int, default=0, help="sobrescreve o tamanho da frota do preset")
    run_sim.add_argument("--days", type=int, default=10)
    run_sim.add_argument("--first-day", type=int, default=1, help="primeiro dia simulado (101 a 120 são os dias de teste)")
    run_sim.add_argument("--seed", type=int, default=7)
    run_sim.add_argument("--render", default="", help="grava o replay de um dia; {policy} vira b0, b1, b1plus ou b2")
    run_sim.add_argument("--render-day", type=int, default=1)
    run_sim.add_argument("--city", default="data/out/city.npz")
    run_sim.add_argument("--paths", default="data/out/paths")
    run_sim.set_defaults(func=cmd_sim)

    tr = sub.add_parser("train", help="treina os modelos próprios (precisa do extra `train`: torch)").add_subparsers(
        dest="train_command", required=True
    )
    train_prep = tr.add_parser("prep", help="modelo de preparo por restaurante e hora")
    train_prep.add_argument("--preset", choices=sorted(world.PRESETS), default="plataforma")
    train_prep.add_argument("--seed", type=int, default=7)
    train_prep.add_argument("--epochs", type=int, default=40)
    train_prep.add_argument("--city", default="data/out/city.npz")
    train_prep.add_argument("--out", default="models/prep.npz")
    train_prep.set_defaults(func=cmd_train_prep)

    train_eta = tr.add_parser("eta", help="modelo de ETA dos trechos")
    train_eta.add_argument("--preset", choices=sorted(world.PRESETS), default="plataforma")
    train_eta.add_argument("--seed", type=int, default=7)
    train_eta.add_argument("--epochs", type=int, default=30)
    train_eta.add_argument("--city", default="data/out/city.npz")
    train_eta.add_argument("--paths", default="data/out/paths")
    train_eta.add_argument("--out", default="models/eta.npz")
    train_eta.set_defaults(func=cmd_train_eta)

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
