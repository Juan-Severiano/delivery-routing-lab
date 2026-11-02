#!/usr/bin/env python3
"""Monta apresentação e replay público sem rede de dados ou treino.

Antes: npm --prefix web ci
Depois: python3 scripts/build_pages.py
Saída: _site/ (sirva com python3 -m http.server --directory _site).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
PAYLOADS = {"city.json", "world.json", *(f"replay-{p}.json" for p in ("b0", "b1", "b1plus", "jev", "b2"))}
SITE_URL = "https://juan-severiano.github.io/delivery-routing-lab/"


def verified_archive(entry: dict) -> bytes:
    archive = entry["archive"]
    if Path(archive).name != archive:
        raise ValueError(f"Nome de arquivo inválido no manifesto: {archive}")
    data = (DEMO / archive).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["gzip_sha256"]:
        raise ValueError(f"Arquivo compactado não confere: {archive}")
    return data


def build(output: Path) -> None:
    if output.exists() and not (output / ".pages-build").is_file():
        raise SystemExit(f"Destino já existe e não foi criado por este script: {output}")
    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    if {entry["name"] for entry in manifest["payloads"]} != PAYLOADS:
        raise ValueError("O manifesto deve conter os sete JSONs esperados pelo mapa")

    # No modo pages, Vite não copia web/public: só o pacote público abaixo entra no site.
    subprocess.run(["npm", "--prefix", str(ROOT / "web"), "run", "build", "--", "--mode", "pages"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="delivery-pages-") as temporary:
        site = Path(temporary) / "site"
        site.mkdir()
        shutil.copy2(ROOT / "docs/delivery-routing-lab.html", site / "index.html")
        guide = site / "como-funciona"
        guide.mkdir()
        shutil.copy2(ROOT / "docs/como-funciona/index.html", guide / "index.html")
        shutil.copytree(ROOT / "docs/assets", site / "assets")
        shutil.copy2(ROOT / "docs/resultados.json", site / "resultados.json")
        shutil.copytree(ROOT / "web/dist", site / "mapa")
        data_dir = site / "mapa/data"
        data_dir.mkdir()
        total = 0
        for entry in manifest["payloads"]:
            data = gzip.decompress(verified_archive(entry))
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"JSON não confere: {entry['name']}")
            json.loads(data)  # O navegador deve receber JSON, não HTML ou gzip sem cabeçalho.
            (data_dir / entry["name"]).write_bytes(data)
            total += len(data)

        downloads = site / "dados"
        downloads.mkdir()
        source = manifest["osm_source"]
        (downloads / source["archive"]).write_bytes(verified_archive(source))
        shutil.copy2(DEMO / "manifest.json", downloads / "manifest.json")
        shutil.copy2(DEMO / "LICENSE-data.txt", downloads / "LICENSE-data.txt")
        shutil.copy2(DEMO / "LICENSE-data.txt", data_dir / "LICENSE-data.txt")
        shutil.copy2(DEMO / "manifest.json", data_dir / "manifest.json")
        (site / ".nojekyll").touch()
        (site / ".pages-build").write_text("delivery-routing-lab\n", encoding="utf-8")
        (site / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n", encoding="utf-8")
        (site / "sitemap.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"<url><loc>{SITE_URL}</loc></url><url><loc>{SITE_URL}como-funciona/</loc></url>"
            f"<url><loc>{SITE_URL}mapa/</loc></url></urlset>\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)  # Somente uma saída anterior identificada pelo marcador acima.
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(site, output)
    print(f"Site -> {output}\nMapa: 7 JSONs verificados, {total / 1e6:.1f} MB, sem treino ou novo download OSM.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
