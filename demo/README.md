# Snapshot público do mapa

Este pacote congela a demonstração usada no GitHub Pages: as ruas de Itapipoca e o replay do **dia 101**, preset `plataforma`, semente **7**, **400 entregadores** e **5.330 pedidos sintéticos**. O mapa compara B0, B1, B1+, Jev e B2, sem executar modelos ou simulação no navegador.

- `city.json.gz`: malha derivada do OpenStreetMap para o canvas.
- `world.json.gz`: restaurantes e demanda sintéticos; calor por hora, média de 30 dias, células de 250 m.
- `replay-*.json.gz`: um dia por política, com decisões e contribuições dos fatores no Jev.
- `itapipoca-roads.osm.json.gz`: recorte OSM original usado para montar o grafo e gerar o experimento.
- `manifest.json`: origem, configuração, tamanho e SHA-256 dos JSONs e dos arquivos compactados.

Os gzip são determinísticos (`mtime=0`). O [build do Pages](../scripts/build_pages.py) confere os hashes, descompacta os sete JSONs do navegador e monta `_site/` sem alterar `web/public/data/`. O modo `pages` do Vite não copia arquivos locais de `web/public/`.

```bash
npm --prefix web ci
python3 scripts/build_pages.py
python3 -m http.server --directory _site
```

O campo `day` de `world.json` guarda uma amostra auxiliar do dia 1; a interface usa esse arquivo para restaurantes e calor, e os pedidos em andamento vêm dos replays do dia 101. Essa amostra auxiliar não muda a comparação mostrada no mapa.

O snapshot usa os modelos versionados no projeto. Ele é uma demonstração congelada, não um novo experimento nem os 20 dias completos de avaliação. As métricas por dia desses 20 dias estão em [`docs/resultados.json`](../docs/resultados.json).

## Fonte e licença

Ruas e dados derivados: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). O pacote de dados é disponibilizado sob a [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/), conforme [`LICENSE-data.txt`](LICENSE-data.txt). O recorte Overpass informa o timestamp OSM `2026-11-01T17:26:28Z`; o manifesto registra seu hash.

O código permanece sob MIT. A distribuição dos dados tem aviso próprio. Restaurantes, pedidos e entregadores são sintéticos. Os JSONs derivados são disponibilizados na [demo pública](https://juan-severiano.github.io/delivery-routing-lab/mapa/data/manifest.json); o [recorte OSM de origem](https://juan-severiano.github.io/delivery-routing-lab/dados/itapipoca-roads.osm.json.gz) também pode ser baixado.
