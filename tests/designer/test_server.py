"""Smoke tests of the vdW Stack Designer's server (flakeid_designer/__main__.py) on a tiny synthetic library: runs list,
collections with ETag / 304, stacks saved with conflict detection, keep / discard decisions, file serving, the build lock
and the installation of a build into a run's folder.  python3 -m pytest tests/designer -q  (a few seconds)"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flakeid_designer.__main__ as D  # noqa: E402

# (the servers' offline copies of the test libraries: not in this computer's app data)
import tempfile  # noqa: E402
os.environ.setdefault("FLAKEID_DESIGNER_OFFLINE_DIR", str(Path(tempfile.gettempdir()) / "flakeid_designer_offline_tests"))


def make_run(base, name="20261016_run_1", files=("x100_nat_a_0123456abc.jpg",)):
    dd = base / name / "designer"
    dd.mkdir(parents=True)
    for f in files:
        (dd / f).write_bytes(b"\xff\xd8" + os.urandom(2000) + b"\xff\xd9")
    bundle = dict(kind="flakeid-designer-bundle", version=2, run=name, title=name, created="2026-10-16 08:00", flakes=[],
                  atlases=[], thumbs=[], scans={}, materials=["hbn"], files=list(files))
    (dd / "bundle.json").write_text(json.dumps(bundle))
    (dd / "run.json").write_text(json.dumps(dict(kind="flakeid-designer-run", run=name, title=name, n_flakes=0,
                                                 materials=["hbn"], scans=[], files=list(files))))
    return dd


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "root"
    make_run(root)
    store = D.Store([root], root / "_designer_data")
    srv = D.make_server(0, store)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    store.scan()
    yield dict(url=f"http://127.0.0.1:{srv.server_address[1]}", store=store, root=root, tmp=tmp_path)
    srv.shutdown()
    srv.server_close()


def req(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    h = dict(headers or {})
    if data is not None:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_runs_and_etag(server):
    st, h, b = req(server["url"] + "/api/db/runs")
    assert st == 200 and [r["id"] for r in json.loads(b)] == ["20261016_run_1"]
    st2, _, b2 = req(server["url"] + "/api/db/runs", headers={"If-None-Match": h["ETag"]})
    assert st2 == 304 and b2 == b""
    make_run(server["root"], "20261016_run_2")
    server["store"].scan()                                # (the background walk, or Refresh)
    st3, h3, b3 = req(server["url"] + "/api/db/runs", headers={"If-None-Match": h["ETag"]})
    assert st3 == 200 and len(json.loads(b3)) == 2 and h3["ETag"] != h["ETag"]


def test_collection_etag_and_reads(server, monkeypatch):
    u = server["url"] + "/api/db/manual"
    assert req(u + "/m1", "PUT", dict(name="a"))[0] == 200
    st, h, b = req(u)
    assert st == 200 and json.loads(b)[0]["data"]["name"] == "a"
    reads = []
    orig = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: reads.append(self.name) or orig(self, *a, **k))
    assert req(u, headers={"If-None-Match": h["ETag"]})[0] == 304
    assert not reads                                      # (unchanged: no file read)
    assert req(u + "/m2", "PUT", dict(name="b"))[0] == 200
    st, h2, b = req(u, headers={"If-None-Match": h["ETag"]})
    assert st == 200 and len(json.loads(b)) == 2
    assert reads.count("m1.json") == 0                    # (only the new file read)


def test_stack_conflict(server):
    u = server["url"] + "/api/db/stacks/s1"
    st, _, b = req(u, "PUT", dict(name="A", session="p1", updated=1))
    sig1 = json.loads(b)["sig"]
    lst = json.loads(req(server["url"] + "/api/db/stacks")[2])
    assert lst[0]["data"]["_sig"] == sig1 and lst[0]["data"]["rev"] == 1
    # another page (p2) opened version 1 and saves: fine
    st, _, b = req(u, "PUT", dict(name="B", session="p2", _base_sig=sig1, updated=2))
    assert st == 200
    sig2 = json.loads(b)["sig"]
    # p1 still has version 1: refused, even though its clock is far ahead
    st, _, b = req(u, "PUT", dict(name="C", session="p3", _base_sig=sig1, updated=10 ** 13))
    assert st == 409 and json.loads(b)["conflict"]
    # the page that saved last goes on; forced saves go through
    assert req(u, "PUT", dict(name="D", session="p2", _base_sig=sig2))[0] == 200
    assert req(u, "PUT", dict(name="E", session="p3", _base_sig=sig1, _force=True))[0] == 200
    # an older designer's page (no _base_sig): the old rule on `updated`
    cur = json.loads(req(server["url"] + "/api/db/stacks")[2])[0]["data"]
    assert req(u, "PUT", dict(name="F", session="old", _base=cur["updated"] - 1))[0] == 409
    assert req(u, "PUT", dict(name="F", session="old", _base=cur["updated"]))[0] == 200
    # stamped later than the version it replaced, and _sig never stored
    d = json.loads((server["root"] / "_designer_data" / "stacks" / "s1.json").read_text())
    assert d["updated"] > cur["updated"] and "_sig" not in d and d["rev"] == 5


def test_review_latest_wins_and_incremental(server):
    u = server["url"] + "/api/db/review"
    rdir = server["root"] / "_designer_data" / "review"
    assert req(u + "/t1", "PUT", dict(decision="keep", updated=1))[0] == 200
    # another computer whose clock is an hour ahead decided "discard"
    ahead = int(time.time() * 1000) + 3600 * 1000
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "zz_other.jsonl", "a") as fh:
        fh.write(json.dumps(dict(id="t1", updated=ahead, by="other", body=dict(decision="discard"))) + "\n")
    st, h, b = req(u)
    assert {x["id"]: x["data"]["decision"] for x in json.loads(b)} == {"t1": "discard"}
    # a decision made here after seeing that one wins although this clock is behind
    assert req(u + "/t1", "PUT", dict(decision="keep", updated=5))[0] == 200
    st, h2, b = req(u, headers={"If-None-Match": h["ETag"]})
    assert st == 200 and json.loads(b)[0]["data"]["decision"] == "keep"
    assert req(u, headers={"If-None-Match": h2["ETag"]})[0] == 304
    # a half-written line is read once it is complete
    with open(rdir / "zz_other.jsonl", "a") as fh:
        fh.write('{"id": "t2", "updated": 1, "body": {"decision": "kee')
    assert "t2" not in {x["id"] for x in json.loads(req(u)[2])}
    with open(rdir / "zz_other.jsonl", "a") as fh:
        fh.write('p"}}\n')
    assert "t2" in {x["id"] for x in json.loads(req(u)[2])}


def test_file_etag_head_and_cache(server):
    u = server["url"] + "/_blob/20261016_run_1/bundle.json"
    st, h, b = req(u)
    assert st == 200 and h["Cache-Control"] == "no-cache" and h.get("ETag") and h.get("Last-Modified")
    assert req(u, headers={"If-None-Match": h["ETag"]})[0] == 304
    assert req(u, headers={"If-Modified-Since": h["Last-Modified"]})[0] == 304      # (what Chromium sends)
    assert req(u, headers={"If-Modified-Since": "Thu, 01 Jan 2026 00:00:00 GMT"})[0] == 200
    st, hh, bh = req(u, "HEAD")
    assert st == 200 and bh == b"" and int(hh["Content-Length"]) == len(b)
    assert any(server["tmp"].joinpath("cache").iterdir())     # (kept on this computer)
    img = server["url"] + "/_blob/20261016_run_1/x100_nat_a_0123456abc.jpg"
    st, hi, bi = req(img)
    assert st == 200 and hi["Content-Type"] == "image/jpeg" and "immutable" in hi["Cache-Control"]     # (named by its content)
    runs = json.loads(req(server["url"] + "/api/db/runs")[2])
    assert runs[0]["data"]["files"]["x100_nat_a_0123456abc.jpg"] == "20261016_run_1/x100_nat_a_0123456abc.jpg"   # (no ?v=)
    assert req(img)[2] == bi                                  # (second time from the cache)
    f = server["root"] / "20261016_run_1" / "designer" / "bundle.json"
    time.sleep(0.01)
    f.write_text(f.read_text() + " ")
    st, h2, b2 = req(u, headers={"If-None-Match": h["ETag"]})
    assert st == 200 and h2["ETag"] != h["ETag"] and b2.endswith(b" ")
    assert req(server["url"] + "/_blob/20261016_run_1/nothing.jpg")[0] == 404
    assert req(server["url"] + "/_blob/20261016_run_1/..")[0] == 404


def test_scan_concurrent(server):
    errs = []

    def go():
        try:
            server["store"].scan()
        except Exception as e:                            # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(server["store"].runs) == 1


def test_build_lock(tmp_path):
    B = D.Builder()
    lock = tmp_path / ".building"
    assert B.take_lock(lock) is None
    assert B.take_lock(lock) == D.user_name()             # (taken: a second taker is refused)
    old = time.time() - B.LOCK_STALE_S - 5
    os.utime(lock, (old, old))
    assert B.take_lock(lock) is None                      # (not renewed: a computer that stopped, taken over)
    lock.write_text(json.dumps(dict(by="old designer")))  # (an older designer's lock: no renewal, kept LOCK_H hours)
    os.utime(lock, (old, old))
    assert B.take_lock(lock) == "old designer"
    lock.unlink()
    got = []
    ts = [threading.Thread(target=lambda: got.append(B.take_lock(lock))) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert got.count(None) == 1


def test_install_build(tmp_path):
    ddir = make_run(tmp_path, "r", files=("x100_nat_a_0123456abc.jpg", "x100_nat_b_aaaaaaaaaa.jpg"))
    (ddir / "build_failed.txt").write_text("old failure")
    new = tmp_path / "new"
    new.mkdir()
    (new / "x100_nat_a_0123456abc.jpg").write_bytes((ddir / "x100_nat_a_0123456abc.jpg").read_bytes())
    (new / "x100_nat_c_bbbbbbbbbb.jpg").write_bytes(b"\xff\xd8new\xff\xd9")
    (new / "bundle.json").write_text(json.dumps(dict(kind="flakeid-designer-bundle", files=["x100_nat_a_0123456abc.jpg", "x100_nat_c_bbbbbbbbbb.jpg"])))
    (new / "run.json").write_text("{}")
    t_same = (ddir / "x100_nat_a_0123456abc.jpg").stat().st_mtime_ns
    D.install_build(new, ddir)
    names = sorted(p.name for p in ddir.iterdir())
    assert names == ["bundle.json", "run.json", "x100_nat_a_0123456abc.jpg", "x100_nat_c_bbbbbbbbbb.jpg"]
    assert (ddir / "x100_nat_a_0123456abc.jpg").stat().st_mtime_ns == t_same      # (same content: not copied again)
    assert "x100_nat_c" in (ddir / "bundle.json").read_text()


def test_sg_picture(server):
    """(2026-11-03m) Key 2 in the 100x window: the scanner's black-and-white green picture (<frame>_sg.jpg) of the frame
    the flake's designer picture was copied from (of several frames: the one of the same size)."""
    run = server["root"] / "20261016_run_1"
    piece = run / "20261016_1"
    (piece / "100x").mkdir(parents=True)
    H = {"7": [dict(f="100x/a_7_001.jpg"), dict(f="100x/a_7_002.jpg", sg="100x/a_7_002_sg.jpg"), dict(f="100x/dom.jpg", domain=True)],
         "9": [dict(f="100x/../../x.jpg")]}
    (piece / "index.html").write_text("<script>const D={};\nconst H100=" + json.dumps(H) + ";\n</script>")
    for n, size in (("a_7_001", 100), ("a_7_002", 200)):
        (piece / "100x" / f"{n}.jpg").write_bytes(b"x" * size)
        (piece / "100x" / f"{n}_sg.jpg").write_bytes(n.encode())
    (run / "designer" / "x100_nat_f7.jpg").write_bytes(b"y" * 200)            # (the designer's copy of frame 002)
    u = server["url"] + "/api/sg?run=20261016_run_1&scan=20261016_1&flake=7"
    st, _, b = req(u + "&img=" + urllib.parse.quote("/_blob/20261016_run_1/x100_nat_f7.jpg"))
    assert st == 200 and b == b"a_7_002"
    st, _, b = req(u)                                                       # (no picture to compare: the first)
    assert st == 200 and b == b"a_7_001"
    assert req(server["url"] + "/api/sg?run=20261016_run_1&scan=20261016_1&flake=9")[0] == 404   # (outside the folder)
    assert req(server["url"] + "/api/sg?run=20261016_run_1&scan=20261016_1&flake=8")[0] == 404
    assert req(server["url"] + "/api/sg?run=nope&scan=20261016_1&flake=7")[0] == 404


def test_afm_typed_and_pgm_dir(server):
    """(2026-11-03q) The AFM list's reference X / Y / Z are saved as typed (/api/afm/typed) and read back; the run's
    afm_pgm folder (made if missing) is where Generate AFM file saves."""
    u, H = server["url"] + "/api/afm/typed", {"X-Designer": "1"}
    k = "20261016_run_1:20261016_1|new"
    assert json.loads(req(u + "?key=" + urllib.parse.quote(k))[2]) == {"data": None}
    d = {"at": 1, "refs": [{"flake": "a", "afm": ["1.5", "", "3"]}]}
    st, _, b = req(u, "POST", {"key": k, "data": d}, H)
    assert st == 200 and json.loads(b) == {"ok": True}
    assert json.loads(req(u + "?key=" + urllib.parse.quote(k))[2]) == {"data": d}
    assert json.loads(req(u + "?key=" + urllib.parse.quote(k + "x"))[2]) == {"data": None}   # (another list)
    assert req(u, "POST", {"key": k, "data": d})[0] == 403                                       # (designer page only)
    assert req(u, "POST", {"key": "", "data": d}, H)[0] == 400
    req(u, "POST", {"key": k, "data": None}, H)                                                  # (removed)
    assert json.loads(req(u + "?key=" + urllib.parse.quote(k))[2]) == {"data": None}
    p = D.afm_pgm_dir(server["store"], "20261016_run_1")
    assert p == str(server["root"] / "20261016_run_1" / "afm_pgm") and Path(p).is_dir()
    assert D.afm_pgm_dir(server["store"], "nope") is None


def test_afm_plan_preview_writes_nothing(tmp_path):
    """(2026-11-03s) The 3D check before Generate AFM file: make_plan {preview} gives the points and writes nothing."""
    try:
        import numpy  # noqa: F401
        D.builder_main()
        import flakeid_designer.afmplan as AP
    except Exception as e:                                  # (no numpy / flakeid here)
        pytest.skip(str(e))
    refs = [dict(flake="a", chip=[0, 0], afm=[100, 200, 1]), dict(flake="b", chip=[1000, 0], afm=[1100, 200, 2]),
            dict(flake="c", chip=[0, 1000], afm=[100, 1200, 3])]
    b = dict(scan="r:c1", run="r", refs=refs, zoff=0.5, sites=[dict(n=1, flake="a", chip=[0, 0]), dict(n=2, flake="x", chip=[500, 500])])
    pv = AP.make_plan(dict(b, preview=True), tmp_path)
    assert pv["preview"] and not list(tmp_path.iterdir())
    assert [s["n"] for s in pv["sites"]] == [1, 2] and pv["has_z"] and len(pv["refs"]) == 3
    assert pv["sites"][1]["afm"] == [600.0, 700.0, 3.0]                # (x, y fitted; z on the plane 2.5 + offset 0.5)
    made = AP.make_plan(b, tmp_path)
    assert [s["afm"] for s in made["sites"]] == [s["afm"] for s in pv["sites"]]
