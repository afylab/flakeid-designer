"""Merging two AFM point lists (.pgm): every point of the first, then every point of the second
(flakeid_designer/afmplan.merge_pgm and the designer's /api/afm/merge)."""
import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid_designer import afmplan as AP  # noqa: E402


def pgm(d, stem, pts):
    sites = [dict(n=i + 1, flake=f"f{i}", afm=list(p), chip=[0, 0]) for i, p in enumerate(pts)]
    return AP.write_lists(d, stem, sites)


def points(text):
    out, cur = [], {}
    for q in text.split("\r\n"):
        if q.startswith("\\X point:"):
            cur = {"x": float(q.split(":")[1])}
        elif q.startswith("\\Y point:"):
            cur["y"] = float(q.split(":")[1])
        elif q.startswith("\\Z point:"):
            cur["z"] = float(q.split(":")[1])
        elif q.startswith("\\Coordinate System"):
            out.append(cur)
    return out


A = [(-86662.5, -105646.88, -16944), (-86000, -105000, -16950)]
B = [(1.5, 2.25), (3, 4), (5, 6)]


def test_merge_order_and_format(tmp_path):
    a, b = pgm(tmp_path, "one", A), pgm(tmp_path, "two", B)
    txt, n1, n2 = AP.merge_pgm(a.read_bytes().decode("latin-1"), b.read_bytes().decode("latin-1"))
    assert (n1, n2) == (2, 3)
    assert txt.startswith("Version: 1.0\r\n") and txt.count("Version") == 1
    assert "\n" not in txt.replace("\r\n", "")                       # CRLF lines only
    got = points(txt)
    assert [(p["x"], p["y"]) for p in got] == [(p[0], p[1]) for p in A + B]
    assert [p.get("z") for p in got] == [-16944, -16950, None, None, None]
    assert txt == a.read_bytes().decode("latin-1") + b.read_bytes().decode("latin-1").split("\r\n", 1)[1]


def test_merge_lf_and_empty(tmp_path):
    a = pgm(tmp_path, "one", A).read_bytes().decode("latin-1").replace("\r\n", "\n")   # (a file saved with LF lines)
    txt, n1, n2 = AP.merge_pgm(a, "Version: 1.0\n\\X point: 1\n\\Y point: 2\n\\Coordinate System: 0\n\n")
    assert (n1, n2) == (2, 1) and points(txt)[-1] == {"x": 1, "y": 2}
    with pytest.raises(ValueError):
        AP.merge_pgm(a, "Version: 1.0\r\n")


def test_merge_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    import flakeid_designer.__main__ as D
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "root"
    root.mkdir()
    store = D.Store([root], root / "_designer_data")
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        a, b = pgm(tmp_path, "one", A), pgm(tmp_path, "two", B)
        (tmp_path / "one_afm_sites+two_afm_sites.pgm").write_text("x")                         # (never written over)

        def post(body):
            r = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/afm/merge", data=json.dumps(body).encode(),
                                       method="POST", headers={"Content-Type": "application/json", "X-Designer": "1"})
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read())
        r = post(dict(first=str(a), second=str(b)))
        assert r["n"] == 5 and Path(r["saved"]).name == "one_afm_sites+two_afm_sites_2.pgm"
        assert [(p["x"], p["y"]) for p in points(Path(r["saved"]).read_bytes().decode("latin-1"))] == [(p[0], p[1]) for p in A + B]
        assert (tmp_path / "one_afm_sites+two_afm_sites.pgm").read_text() == "x"
        monkeypatch.setattr(D, "pick_file", lambda *a, **k: None)            # (the dialog cancelled)
        assert post({}) == {"cancelled": True}
    finally:
        srv.shutdown()
        srv.server_close()
