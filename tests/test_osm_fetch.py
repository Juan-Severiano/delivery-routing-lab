import io
import json

import pytest

from delivery_lab import osm


class FakeResponse(io.BytesIO):
    """Resposta HTTP de mentira: opcionalmente cai no meio do corpo."""

    def __init__(self, body: bytes, fail_after: int | None = None):
        super().__init__(body)
        self.fail_after = fail_after

    def read(self, n=-1):
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise TimeoutError("The read operation timed out")
        return super().read(min(n, 16) if self.fail_after is not None else n)


def patch_urlopen(monkeypatch, response):
    monkeypatch.setattr(osm.urllib.request, "urlopen", lambda *a, **k: response)


def test_fetch_writes_the_file_and_leaves_no_part_file(tmp_path, monkeypatch):
    body = json.dumps({"elements": []}).encode()
    patch_urlopen(monkeypatch, FakeResponse(body))
    out = tmp_path / "raw" / "roads.json"
    assert osm.fetch("Itapipoca", out) == len(body)
    assert out.read_bytes() == body
    assert not list(out.parent.glob("*.part"))


def test_fetch_that_drops_midway_leaves_nothing_behind(tmp_path, monkeypatch):
    body = json.dumps({"elements": [{"type": "node", "id": i} for i in range(50)]}).encode()
    patch_urlopen(monkeypatch, FakeResponse(body, fail_after=40))
    out = tmp_path / "roads.json"
    with pytest.raises(TimeoutError):
        osm.fetch("Itapipoca", out)
    assert not out.exists()  # senão o `city fetch` se recusaria a baixar de novo
    assert not list(tmp_path.glob("*.part"))


def test_fetch_rejects_a_server_side_error_in_the_remark(tmp_path, monkeypatch):
    body = json.dumps({"elements": [], "remark": "runtime error: Query timed out"}).encode()
    patch_urlopen(monkeypatch, FakeResponse(body))
    out = tmp_path / "roads.json"
    with pytest.raises(RuntimeError, match="timed out"):
        osm.fetch("Itapipoca", out)
    assert not out.exists()


def test_a_failed_fetch_keeps_an_older_good_file(tmp_path, monkeypatch):
    out = tmp_path / "roads.json"
    out.write_text('{"elements": []}')
    patch_urlopen(monkeypatch, FakeResponse(b'{"elements": [', fail_after=5))
    with pytest.raises(TimeoutError):
        osm.fetch("Itapipoca", out)
    assert out.read_text() == '{"elements": []}'


def test_place_names_with_quotes_are_rejected():
    with pytest.raises(ValueError):
        osm.overpass_query('Itapipoca"]; out;')
