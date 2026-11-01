import json
from pathlib import Path

import pytest

from delivery_lab.city import build_city
from delivery_lab.osm import read_osm


@pytest.fixture(scope="session")
def grid_city(tmp_path_factory):
    """Cidade de brinquedo: grade 40x40 de ruas residenciais (~110 m de lado) com uma avenida no meio."""
    size, step = 40, 1e-3
    node = lambda r, c: r * size + c + 1  # noqa: E731
    elements = [
        {"type": "node", "id": node(r, c), "lat": r * step, "lon": c * step}
        for r in range(size)
        for c in range(size)
    ]
    ways = []
    for k in range(size):
        kind = "primary" if k == size // 2 else "residential"
        ways.append({"nodes": [node(k, c) for c in range(size)], "tags": {"highway": kind}})
        ways.append({"nodes": [node(r, k) for r in range(size)], "tags": {"highway": "residential"}})
    elements += [{"type": "way", "id": 10_000 + i, **w} for i, w in enumerate(ways)]
    path = Path(tmp_path_factory.mktemp("osm")) / "grid.json"
    path.write_text(json.dumps({"elements": elements}))
    return build_city(read_osm(path))[0]
