"""Kept flakes' 100x pictures copied into the run's kept/ folder, one subfolder per kind (flakeid_designer/kept.py)."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flakeid_designer.__main__ as D  # noqa: E402
from flakeid_designer import kept as KP  # noqa: E402
from test_server import make_run, req  # noqa: E402

RUN = "20261016_run_1"


def make_piece(run_dir, piece="chipA", flakes=(12, 13)):
    """A piece's viewer on the share (viewer page on top) with 100x pictures of the given flakes."""
    v = run_dir / piece
    (v / "100x" / "domains").mkdir(parents=True)
    (v / "data").mkdir()
    H = {}
    for n in flakes:
        s = f"{piece}_100x_f{n}"
        names = [f"{s}.jpg", f"{s}_look.jpg", f"{s}_c8.jpg", f"{s}_sg.jpg", f"{s}_dt.jpg", f"{s}_debris.jpg",
                 f"{s}_avg.jpg", f"{s}_avg_look.jpg", f"{s}_avg_c8.jpg", f"{s}_avg_sg.jpg", f"{s}_mosaic.jpg",
                 f"{s}_ir.jpg", f"domains/{s}_ir0.png", f"domains/{s}_ir0.jpg"]
        for nm in names:
            (v / "100x" / nm).write_bytes(nm.encode() * 10)
        H[str(n)] = [dict(f=f"100x/{s}.jpg", p=f"100x/preview/{s}.jpg", lab=f"f{n}",
                          look=dict(f=f"100x/{s}_look.jpg", a=f"100x/{s}_avg_look.jpg"),
                          a=f"100x/{s}_avg.jpg", a_c8=f"100x/{s}_avg_c8.jpg", a_sg=f"100x/{s}_avg_sg.jpg",
                          c8=f"100x/{s}_c8.jpg", sg=f"100x/{s}_sg.jpg", deb=f"100x/{s}_debris.jpg",
                          ir=f"100x/{s}_ir.jpg", mos=f"100x/{s}_mosaic.jpg"),
                     dict(f=f"100x/domains/{s}_ir0.jpg", raw=f"100x/domains/{s}_ir0.png", domain=True)]
    tiff = run_dir / f"{piece}_100x"
    tiff.mkdir()
    (tiff / f"{piece}_100x_f{flakes[0]}.tif").write_bytes(b"II*\x00" * 50)
    (v / "index.html").write_text("<script>\nconst D=" + json.dumps(dict(name=f"{piece}_tiles", flakes=[])) + ";\n"
                                  + "const H100=" + json.dumps(H) + ";\n</script>")
    return v


def test_sync_copies_by_kind_and_removes(tmp_path):
    run = tmp_path / RUN
    run.mkdir()
    make_piece(run)
    st = KP.sync_run(run, {("chipA", 12): dict(by="avram", at=1, terraces=[])})
    k = run / "kept"
    assert st["flakes"] == 1 and not st["missing"]
    for kind, names in (("raw", ["f12.jpg"]), ("raw_avg", ["f12_avg.jpg"]), ("raw_tiff", ["f12.tif"]),
                        ("natural", ["f12_look.jpg"]), ("natural_avg", ["f12_avg_look.jpg"]),
                        ("color_8percent", ["f12_c8.jpg"]), ("color_8percent_avg", ["f12_avg_c8.jpg"]),
                        ("bw", ["f12_sg.jpg"]), ("bw_avg", ["f12_avg_sg.jpg"]), ("thickness_change", ["f12_dt.jpg"]),
                        ("debris", ["f12_debris.jpg"]), ("ir", ["f12_ir.jpg"]), ("mosaic", ["f12_mosaic.jpg"]),
                        ("ir_domains", ["f12_ir0.png"]), ("ir_domains_preview", ["f12_ir0.jpg"])):
        # (one version of each picture per folder: no _avg next to the plain one)
        assert sorted(p.name for p in (k / kind).iterdir()) == [f"chipA-f0012_chipA_100x_{n}" for n in names], kind
    assert not list(k.rglob("*f13*"))
    assert "chipA-f0012" in (k / "kept_flakes.csv").read_text()
    # again: nothing copied; flake 13 kept and 12 discarded: 12's copies removed
    assert KP.sync_run(run, {("chipA", 12): {}})["copied"] == 0
    st = KP.sync_run(run, {("chipA", 13): {}})
    assert st["removed"] >= 14 and not list(k.rglob("chipA-f0012*")) and list(k.rglob("chipA-f0013*"))


def test_kept_flakes_from_decisions():
    latest = {f"{RUN}:chipA-f0012.a": dict(by="a", updated=5, body=dict(decision="keep")),
              f"{RUN}:chipA-f0012.b": dict(by="b", updated=6, body=dict(decision="discard")),
              f"{RUN}:chipA-f0013.a": dict(by="a", updated=7, body=dict(decision="discard")),
              f"{RUN}:chipB-f0002.u1x": dict(by="c", updated=8, body=dict(decision="used"))}
    assert set(KP.kept_flakes(latest)) == {(RUN, "chipA", 12)}     # (picked up, "used": not in the kept list)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "config.json")
    monkeypatch.setattr(KP, "DEBOUNCE_S", 0.1)
    root = tmp_path / "root"
    make_run(root, RUN)
    make_piece(root / RUN)
    store = D.Store([root], root / "_designer_data")
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    store.scan()
    yield dict(url=f"http://127.0.0.1:{srv.server_address[1]}", root=root, store=store)
    srv.shutdown()
    srv.server_close()


def test_keep_decision_fills_kept_folder(server):
    tid = f"{RUN}:chipA-f0013.a"
    s, _, _ = req(f"{server['url']}/api/db/review/{tid}", "PUT", dict(decision="keep", terrace=tid))
    assert s == 200
    f = server["root"] / RUN / "kept" / "color_8percent" / "chipA-f0013_chipA_100x_f13_c8.jpg"
    for _ in range(100):
        if f.is_file():
            break
        time.sleep(0.1)
    assert f.is_file()
    req(f"{server['url']}/api/db/review/{tid}", "PUT", dict(decision="discard", terrace=tid))
    for _ in range(100):
        if not f.is_file():
            break
        time.sleep(0.1)
    assert not f.is_file()


def test_old_layout_is_rearranged(tmp_path):
    run = tmp_path / RUN
    run.mkdir()
    make_piece(run)
    k = run / "kept"
    (k / "raw").mkdir(parents=True)
    old = k / "raw" / "chipA-f0012_chipA_100x_f12_avg.jpg"           # the first layout: the average next to the plain one
    old.write_bytes(b"x")
    (k / KP.MANIFEST).write_text(json.dumps(dict(flakes={"chipA-f0012": ["raw/" + old.name]})))
    assert KP.needs_sync(run, ["chipA-f0012"])
    KP.sync_run(run, {("chipA", 12): {}})
    assert not old.exists() and (k / "raw_avg" / old.name).is_file()
    assert not KP.needs_sync(run, ["chipA-f0012"])


def test_update_button_endpoint(server):
    tid = f"{RUN}:chipA-f0012.a"
    req(f"{server['url']}/api/db/review/{tid}", "PUT", dict(decision="keep", terrace=tid))
    k = server["root"] / RUN / "kept"
    stray = k / "raw" / "chipA-f0099_old_copy.jpg"                    # (left over: not in the folder's list)
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"x")
    s, _, body = req(f"{server['url']}/api/kept/sync", "POST", dict(run=RUN))
    r = json.loads(body)
    assert s == 200 and r["results"][0]["flakes"] == 1, r
    assert not stray.exists() and (k / "raw" / "chipA-f0012_chipA_100x_f12.jpg").is_file()
    s, _, _ = req(f"{server['url']}/api/kept/sync", "POST", dict(run="nope"))
    assert s == 404


def test_stars_listed_and_reset_screening(server):
    u = server["url"] + "/api/db/review"
    for tid, dec in ((f"{RUN}:chipA-f0012.a", "keep"), (f"{RUN}:chipA-f0013.a", "discard"), ("other_run:chipB-f0001.a", "keep")):
        req(f"{u}/{tid}", "PUT", dict(decision=dec, terrace=tid))
    req(f"{u}/{RUN}:chipA-f0013.stars", "PUT", dict(stars=2, flake=f"{RUN}:chipA-f0013"))
    req(f"{u}/{RUN}:chipA-f0013.clean", "PUT", dict(clean=[120.5, 80], flake=f"{RUN}:chipA-f0013"))
    ids = {x["id"]: x["data"] for x in json.loads(req(u)[2])}
    assert ids[f"{RUN}:chipA-f0013.stars"]["stars"] == 2
    assert ids[f"{RUN}:chipA-f0013.clean"]["clean"] == [120.5, 80]
    s, _, body = req(server["url"] + "/api/review/reset", "POST", dict(run=RUN))
    assert s == 200 and json.loads(body)["cleared"] == 2
    ids = {x["id"]: x["data"] for x in json.loads(req(u)[2])}
    assert f"{RUN}:chipA-f0012.a" not in ids and f"{RUN}:chipA-f0013.a" not in ids
    assert "other_run:chipB-f0001.a" in ids and ids[f"{RUN}:chipA-f0013.stars"]["stars"] == 2   # (other runs, stars stay)


def test_afm_center_travels_with_the_100x_pictures(server):
    """An AFM center is written next to the 100x pictures (100x/afm_centers.json) and read back for a run made again
    from them (another run name, the same pictures): matched by the 100x picture's file name."""
    import shutil
    tid = f"{RUN}:chipA-f0012.clean"
    s, _, _ = req(f"{server['url']}/api/db/review/{tid}", "PUT",
                  dict(clean=[100, 80], rel=[0.25, 0.4], um=[-40.5, -12.0], flake=f"{RUN}:chipA-f0012"))
    assert s == 200
    f = server["root"] / RUN / "chipA" / "100x" / "afm_centers.json"
    for _ in range(100):
        if f.is_file():
            break
        time.sleep(0.05)
    C = json.loads(f.read_text())["centers"]
    assert C["chipA_100x_f12.jpg"]["rel"] == [0.25, 0.4] and C["chipA_100x_f12.jpg"]["flake"] == 12
    # a new run from the same 100x pictures (stage positions recalibrated): the centers come with them
    new = "20261029_run_14"
    shutil.copytree(server["root"] / RUN, server["root"] / new)
    for name in ("bundle.json", "run.json"):
        p = server["root"] / new / "designer" / name
        p.write_text(p.read_text().replace(RUN, new))
    D.Store.scan(server["store"]) if "store" in server else None
    s, _, body = req(f"{server['url']}/api/afmcenters?run={new}")
    assert s == 200
    assert json.loads(body)["centers"]["chipA-f0012"]["rel"] == [0.25, 0.4]
    # removed: gone from the file too
    req(f"{server['url']}/api/db/review/{tid}", "PUT", dict(clean=None, flake=f"{RUN}:chipA-f0012"))
    for _ in range(100):
        if "chipA_100x_f12.jpg" not in json.loads(f.read_text())["centers"]:
            break
        time.sleep(0.05)
    assert "chipA_100x_f12.jpg" not in json.loads(f.read_text())["centers"]
