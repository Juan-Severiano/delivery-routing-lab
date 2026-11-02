#!/usr/bin/env python3
"""Mede as políticas locais (--collect) ou recria os gráficos do snapshot versionado.

Só a coleta usa o simulador e os modelos. Renderizar os gráficos não baixa o OSM,
retreina modelos nem executa simulação. Execute a partir da raiz do repositório:

    uv run --with matplotlib python scripts/plot_results.py
    uv run --extra train --with matplotlib python scripts/plot_results.py --collect --workers 3

As barras de erro são desvios padrão populacionais ENTRE os 20 dias (ddof=0),
como na CLI do laboratório. Não são intervalos de confiança.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs/resultados.json"
POLICIES = ("b0", "b1", "b1+preparo", "b1+eta", "b1+", "jev", "jev-mlp", "b2")
FIRST_DAY, LAST_DAY, SEED = 101, 120, 7


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_policy(key: str) -> tuple[str, dict]:
    """Reproduz os defaults da CLI; não mede latência sob concorrência."""
    import numpy as np
    from delivery_lab import paths, planner, sim, world
    from delivery_lab.city import City

    city = City.load(ROOT / "data/out/city.npz")
    preset = world.PRESETS["plataforma"]
    w = world.generate_world(city, SEED, preset["restaurants"], preset["orders_per_day"], 4000.0, preset["active_share"])
    fleet = sim.generate_fleet(w, city, preset["couriers"], SEED)
    pth = paths.Paths.load(city, ROOT / "data/out/paths")
    if key == "b0":
        policy = sim.NearestPolicy()
    elif key == "b1":
        policy = planner.Planner()
    elif key == "b2":
        policy = planner.Planner(oracle=True)
    else:
        from delivery_lab.learn.eta import EtaModel
        from delivery_lab.learn.prep import PrepModel
        prep = PrepModel.load(ROOT / "models/prep.npz") if key != "b1+eta" else None
        eta = EtaModel.load(ROOT / "models/eta.npz") if key != "b1+preparo" else None
        if key.startswith("jev"):
            from delivery_lab.learn.jev import JevModel, JevPlanner
            kind = "linear" if key == "jev" else "mlp"
            policy = JevPlanner(prep, eta, JevModel.load(ROOT / f"models/jev-{kind}.npz"))
        else:
            policy = planner.Planner(prep_model=prep, eta_model=eta)
    days = [{"day": day, **sim.summarize(sim.run_day(city, w, pth, fleet, policy, day))}
            for day in range(FIRST_DAY, LAST_DAY + 1)]
    metrics = {key: {"mean": float(np.mean([row[key] for row in days])),
                     "std_between_days": float(np.std([row[key] for row in days], ddof=0))}
               for key in days[0] if key != "day"}
    return key, {"label": policy.name, "metrics": metrics, "days": days}


def collect(workers: int) -> dict:
    import numpy as np
    from delivery_lab import planner, sim, world
    from delivery_lab.city import City

    required = [ROOT / "data/out/city.npz", ROOT / "data/out/paths/dist.f32", ROOT / "data/out/paths/pred.i16"]
    required += list((ROOT / "models").glob("*.npz"))
    required += [ROOT / "models" / name for name in ("prep.json", "eta.json", "jev.json")]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Ausente: {path.relative_to(ROOT)}. Siga o preparo de dados no README.")
    city = City.load(ROOT / "data/out/city.npz")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = None
    data = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": "Reexecução local das políticas com os modelos versionados, sem retreino. Só as ruas são reais; toda a operação é sintética.",
        "provenance": {
            "base_git_revision": revision,
            "python": platform.python_version(), "numpy": np.__version__,
            "collector": f"scripts/plot_results.py --collect --workers {workers}",
            "metric_implementation": "src/delivery_lab/sim.py:summarize",
            "on_time_denominator": "Pedidos entregues. Os sem entrega/rejeitados são reportados separadamente em cada dia.",
            "aggregation": "Média aritmética das métricas diárias; std_between_days usa ddof=0. Não é IC nem teste de significância.",
            "latency": "Não coletada aqui: processos concorrentes tornam comparação de latência inadequada.",
            "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in required},
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))},
            "osm_note": "O recorte OSM não é versionado. Um novo download pode mudar o grafo e os resultados; compare os hashes dos dados.",
        },
        "config": {"preset": "plataforma", "seed": SEED, "test_days": list(range(FIRST_DAY, LAST_DAY + 1)),
                   "training_days_prep_eta": [21, 60], "validation_days_prep_eta": [61, 70],
                   "training_days_jev": [21, 50], "validation_days_jev": [61, 65],
                   "planner_tuning_days": [1, 10],
                   "restaurants": world.PRESETS["plataforma"]["restaurants"], "active_share": world.PRESETS["plataforma"]["active_share"],
                   "orders_per_day_target": world.PRESETS["plataforma"]["orders_per_day"], "couriers": 400,
                   "radius_m": 4000, "city_nodes": city.n_nodes, "city_directed_edges": city.n_edges,
                   "simulator": asdict(sim.SimConfig()),
                   "planner": {"k_nearest": planner.K_NEAREST, "prep_guess_s": planner.PREP_GUESS_S,
                               "late_weight": planner.LATE_WEIGHT, "margin_b1_s": planner.DEPART_MARGIN_B1_S,
                               "margin_b2_s": planner.DEPART_MARGIN_B2_S,
                               "margin_learned_s": planner.DEPART_MARGIN_LEARNED_S,
                               "prep_quantile": planner.PREP_QUANTILE}},
        "offline_evaluation": {name: {"source": f"models/{name}.json", "sha256": digest(ROOT / f"models/{name}.json"),
                                             "metrics": json.loads((ROOT / f"models/{name}.json").read_text())}
                               for name in ("prep", "eta", "jev")},
        "policies": {},
    }
    print(f"Coletando {len(POLICIES)} políticas × 20 dias; {workers} processos. Sem retreino ou rede.", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect_policy, key): key for key in POLICIES}
        for future in as_completed(futures):
            key, result = future.result()
            data["policies"][key] = result
            metric = result["metrics"]["on_time_pct"]
            print(f"{key}: {metric['mean']:.4f}% ± {metric['std_between_days']:.4f} p.p.", flush=True)
    data["policies"] = {key: data["policies"][key] for key in POLICIES}
    source, better = data["policies"]["b1"]["days"], data["policies"]["b1+"]["days"]
    delta = [b["on_time_pct"] - a["on_time_pct"] for a, b in zip(source, better)]
    data["paired_b1plus_minus_b1_pp"] = {"mean": float(np.mean(delta)), "min": min(delta), "max": max(delta),
                                           "positive_days": sum(d > 0 for d in delta), "days": len(delta)}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    temp = SNAPSHOT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(SNAPSHOT)
    print(f"Snapshot: {SNAPSHOT.relative_to(ROOT)}", flush=True)
    return data


def plot(data: dict, output: Path, preview_dir: Path | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as error:
        raise SystemExit("Falta matplotlib. Use: uv run --with matplotlib python scripts/plot_results.py") from error

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titleweight": "bold",
                         "axes.labelcolor": "#40516a", "text.color": "#162a43", "xtick.color": "#40516a",
                         "ytick.color": "#40516a", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "svg.fonttype": "path", "svg.hashsalt": "delivery-routing-lab-v1"})
    output.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    colors = {"b0": "#8593a3", "b1": "#596b83", "b1+preparo": "#4486b9", "b1+eta": "#96b5cc",
              "b1+": "#2764c2", "jev": "#00877d", "jev-mlp": "#9865a8", "b2": "#b27d20"}
    def pct(n: float, digits: int = 1) -> str:
        return f"{n:.{digits}f}".replace(".", ",") + "%"
    def mean(key: str) -> float:
        return data["policies"][key]["metrics"]["on_time_pct"]["mean"]
    def std(key: str) -> float:
        return data["policies"][key]["metrics"]["on_time_pct"]["std_between_days"]
    def style(ax, *, grid: str = "y"):
        ax.set_axisbelow(True)
        ax.grid(axis=grid, color="#dfe6ed", linewidth=0.8)
        ax.tick_params(length=0, pad=9)
    def save(fig, name: str, title: str, description: str):
        svg = output / f"{name}.svg"
        fig.savefig(svg, facecolor="white", metadata={"Title": title, "Description": description, "Date": None})
        # Matplotlib deixa espaços no fim dos segmentos; normalize sem mudar os caminhos.
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        if preview_dir:
            fig.savefig(preview_dir / f"{name}.png", dpi=150, facecolor="white")
        plt.close(fig)
    def header(fig, title, subtitle):
        fig.text(0.075, 0.94, title, fontsize=20, weight="bold", va="top")
        fig.text(0.075, 0.878, subtitle, fontsize=11.5, color="#53647a", va="top")
    def footer(fig, text):
        fig.text(0.075, 0.055, text, fontsize=10, color="#53647a", va="bottom", linespacing=1.6)

    keys = ["b0", "b1", "b1+", "jev", "b2"]
    labels = ["B0\nMais próximo", "B1\nHeurística", "B1+\nModelos aprendidos", "Jev linear\nPróxima parada", "B2\nVisão privilegiada"]
    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.23, top=0.77)
    header(fig, "Modelos aprendidos melhoram a pontualidade na simulação", "Mesma frota de 400 entregadores · 20 dias de teste (101–120) · semente 7")
    bars = ax.bar(range(len(keys)), [mean(k) for k in keys], width=0.56, color=[colors[k] for k in keys],
                  yerr=[std(k) for k in keys], capsize=5, error_kw={"ecolor": "#263a50", "elinewidth": 1.2})
    bars[-1].set_hatch("//")
    for i, key in enumerate(keys):
        ax.text(i, mean(key) + std(key) + 1.6, pct(mean(key)), ha="center", fontsize=13, weight="bold")
    ax.set(ylim=(0, 105), xticks=range(len(keys)), xticklabels=labels, ylabel="Entregas no prazo (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    style(ax)
    footer(fig, "Barras: média diária; hastes: ±1 desvio padrão entre dias (não é intervalo de confiança).\nOperação sintética; ruas OSM reais. B2 é uma referência heurística, com informação indisponível às outras políticas.")
    save(fig, "politicas-no-prazo", "Pontualidade por política", "Comparação de cinco políticas no mesmo conjunto de 20 dias sintéticos. B1+ e Jev têm cerca de 91% no prazo; B2 é uma referência privilegiada, não ótimo global.")

    keys = ["b1", "b1+preparo", "b1+eta", "b1+"]
    labels = ["Regras do B1", "Só preparo\naprendido", "Só ETA\naprendido", "Preparo + ETA\nB1+"]
    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.23, top=0.77)
    header(fig, "A configuração com preparo aprendido concentra o ganho", "Mesmo planejador e dias de teste; variantes com preparo também ajustam a margem de saída")
    ax.bar(range(4), [mean(k) for k in keys], width=0.55, color=[colors[k] for k in keys],
           yerr=[std(k) for k in keys], capsize=5, error_kw={"ecolor": "#263a50", "elinewidth": 1.2})
    for i, key in enumerate(keys):
        ax.text(i, mean(key) + std(key) + 1.8, pct(mean(key)), ha="center", fontsize=13, weight="bold")
    ax.set(ylim=(0, 105), xticks=range(4), xticklabels=labels, ylabel="Entregas no prazo (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    style(ax)
    footer(fig, "Média ±1 desvio padrão entre os dias 101–120; semente 7; 400 entregadores.\nPreparo aprendido: quantil 0,6 e margem de 30 s; B1 e só ETA: 150 s. Ajustes nos dias 1–10.\nA comparação inclui a margem de saída; os resultados dependem do simulador e não medem uma operação real.")
    save(fig, "ablacao-preparo-eta", "Ablação dos modelos aprendidos", "Pontualidade do B1 ao adicionar os modelos de preparo e ETA isolados e em conjunto. O ganho maior vem do modelo de preparo neste simulador.")

    offline = data["offline_evaluation"]["jev"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.28, top=0.70, wspace=0.28)
    header(fig, "Imitar melhor o B2 não trouxe melhor resultado na frota", "Duas avaliações diferentes: concordância de decisões e pontualidade da política em simulação")
    ax = axes[0]
    vals = [offline["linear"]["baselines_top1"]["planejador_reotimizando"] * 100,
            offline["linear"]["top1"] * 100, offline["mlp"]["top1"] * 100]
    ax.bar(range(3), vals, width=0.53, color=[colors["b1"], colors["jev"], colors["jev-mlp"]])
    for i, value in enumerate(vals):
        ax.text(i, value + 2.4, pct(value), ha="center", fontsize=12, weight="bold")
    ax.set(ylim=(0, 105), xticks=range(3), xticklabels=["Planejador*", "Jev linear", "Jev rede"], ylabel="Mesma escolha que o B2 (%)")
    ax.set_title("Imitação offline · 129.013 decisões", loc="left", pad=17, fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    style(ax)
    ax = axes[1]
    keys = ["b1+", "jev", "jev-mlp"]
    for i, key in enumerate(keys):
        ax.errorbar(i, mean(key), yerr=std(key), fmt="o", markersize=9, color=colors[key], capsize=6, elinewidth=1.5)
        ax.text(i, mean(key) + std(key) + 0.4, pct(mean(key), 2), ha="center", fontsize=12, weight="bold", color=colors[key])
    ax.set(xlim=(-0.6, 2.6), ylim=(87, 97.3), xticks=range(3), xticklabels=["B1+", "Jev linear", "Jev rede"], ylabel="Entregas no prazo (%)")
    ax.set_title("Na frota · 20 dias de teste", loc="left", pad=17, fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    style(ax)
    footer(fig, "À direita, escala ampliada e pontos com ±1 desvio padrão entre dias; não é teste de significância.\n*Planejador re-otimizando com a mesma visão da plataforma. Offline: models/jev.json; frota: docs/resultados.json.\nA rede imita melhor; nenhum ganho operacional do Jev sobre o B1+ foi demonstrado neste experimento sintético.")
    save(fig, "jev-imitacao-operacao", "Imitação offline e resultado na frota", "O Jev rede acerta mais escolhas do B2 que o Jev linear, mas não melhora a pontualidade da política. À direita, os intervalos mostram variação entre dias, não incerteza da média.")
    print(f"3 gráficos SVG -> {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect", action="store_true", help="reexecuta 8 políticas × 20 dias com dados e modelos locais")
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5), help="processos de coleta (1–4, padrão 3)")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT, help="snapshot de entrada no modo de renderização")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets", help="diretório de saída dos SVGs")
    parser.add_argument("--preview-dir", type=Path, help="diretório opcional para PNGs de conferência")
    args = parser.parse_args()
    if args.collect:
        data = collect(args.workers)
    else:
        if not args.snapshot.is_file():
            parser.error(f"snapshot ausente: {args.snapshot}; prepare os dados e execute --collect")
        data = json.loads(args.snapshot.read_text())
    plot(data, args.output, args.preview_dir)


if __name__ == "__main__":
    main()
