"""Libraries keep their own folders (2026-11-03f).  A folder window left open while the library was changed saved stale
settings back with one library's folders under another's name: BN_15_auto/20261101_run_1 then opened GR13's empty
designer data, and its decisions and AFM lists looked lost.  python3 -m pytest tests/designer -q"""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flakeid_designer.__main__ as D  # noqa: E402
from test_server import make_run, req  # noqa: E402


@pytest.fixture()
def libs(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "settings.json")
    a, g, c = tmp_path / "BN_20261101_run_1", tmp_path / "GR13_20261029_1", tmp_path / "other"
    for p, run in ((a, "20261101_run_1"), (g, "20261029_run_1"), (c, "20261030_run_2")):
        make_run(p, run)
    D.save_config(dict(library=str(a), roots=[str(a)], data_dir=str(a / "_designer_data"),
                       libraries={str(a): dict(roots=[str(a)], data_dir=str(a / "_designer_data"), used=1)}))
    roots, data, library = D.start_library(D.load_config())
    store = D.Store(roots, data, library)
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    store.scan()
    yield dict(url=f"http://127.0.0.1:{srv.server_address[1]}", store=store, a=a, g=g, c=c)
    srv.shutdown()
    srv.server_close()


def post(url, body=None):
    st, _, b = req(url, "POST", body if body is not None else {})
    return st, json.loads(b or b"{}")


def test_folder_window_open_while_switching(libs, monkeypatch):
    """The user's case: Change directory… (its window behind the browser), the library switched meanwhile, then the
    first library chosen in the window.  Each library must keep its own folders."""
    a, g, u = libs["a"], libs["g"], libs["url"]
    gate, chosen = threading.Event(), [None]

    def slow_pick(*_a, **_k):
        gate.wait(10)
        return chosen[0]
    monkeypatch.setattr(D, "pick_folder", slow_pick)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", post(u + "/api/library/pick")))
    t.start()
    time.sleep(0.3)
    st, r = post(u + "/api/library/pick")                # a second window while one is open: refused
    assert st == 409 and "already open" in r["error"]
    st, r = post(u + "/api/roots/pick")
    assert st == 409
    st, r = post(u + "/api/library/use", dict(path=str(g)))
    assert st == 200 and Path(libs["store"].data) == g / "_designer_data"
    chosen[0] = str(a)
    gate.set()
    t.join(10)
    st, r = out["r"]
    assert st == 200 and r["picked"] == str(a)
    store = libs["store"]
    assert store.roots == [str(a)] and Path(store.data) == a / "_designer_data" and store.library == str(a)
    cfg = D.load_config()
    assert cfg["libraries"][str(a)]["roots"] == [str(a)]
    assert cfg["libraries"][str(a)]["data_dir"] == str(a / "_designer_data")
    assert cfg["libraries"][str(g)]["roots"] == [str(g)]
    assert cfg["library"] == str(a) and cfg["roots"] == [str(a)]


def test_stale_settings_do_not_rename_folders(libs):
    """Another request (or window) wrote a different current library to the settings: the store's folders are still
    saved under the store's own library."""
    a, g, store = libs["a"], libs["g"], libs["store"]
    D.switch_library(D.load_config(), store, str(g))
    cfg = D.load_config()
    cfg["library"] = str(a)                              # (stale: written by something that read it earlier)
    D.save_config(cfg)
    D.switch_library(D.load_config(), store, str(libs["c"]))
    cfg = D.load_config()
    assert cfg["libraries"][str(a)]["roots"] == [str(a)]
    assert cfg["libraries"][str(g)]["roots"] == [str(g)]
    assert cfg["libraries"][str(libs["c"])]["roots"] == [str(libs["c"])]


def test_saved_under_wrong_name_is_repaired(libs):
    """The settings as found on 2026-11-03: the BN library saved with GR13's folders and data."""
    a, g, store = libs["a"], libs["g"], libs["store"]
    bad = dict(roots=[str(g)], data_dir=str(g / "_designer_data"), used=5)
    D.save_config(dict(library=str(a), roots=[str(g)], data_dir=str(g / "_designer_data"),
                       libraries={str(a): bad, str(libs["c"]): dict(roots=[str(libs["c"])], data_dir=str(libs["c"] / "_designer_data"))}))
    roots, data, library = D.start_library(D.load_config())             # (at start: put back and saved)
    assert roots == [str(a)] and Path(data) == a / "_designer_data" and library == str(a)
    cfg = D.load_config()
    assert cfg["libraries"][str(a)] == dict(roots=[str(a)], data_dir=str(a / "_designer_data"), used=5)
    assert cfg["libraries"][str(libs["c"])]["roots"] == [str(libs["c"])]   # (good ones untouched)
    cfg["libraries"][str(a)] = bad                                     # switching to a bad entry: its own folder
    D.save_config(cfg)
    store.library = str(libs["c"])
    store.roots, store.data = [str(libs["c"])], libs["c"] / "_designer_data"
    D.switch_library(D.load_config(), store, str(a))
    assert store.roots == [str(a)] and Path(store.data) == a / "_designer_data"


def test_library_with_added_folders_kept(libs, monkeypatch):
    """A library with other folders added (Add folder…) is its own: kept as it is, and switched back to with them."""
    a, g, u, store = libs["a"], libs["g"], libs["url"], libs["store"]
    monkeypatch.setattr(D, "pick_folder", lambda *_a, **_k: str(g))
    st, r = post(u + "/api/roots/pick")
    assert st == 200 and r["roots"] == [str(a), str(g)] and Path(r["data_dir"]) == a / "_designer_data"
    D.switch_library(D.load_config(), store, str(libs["c"]))
    D.switch_library(D.load_config(), store, str(a))
    assert store.roots == [str(a), str(g)] and Path(store.data) == a / "_designer_data"
    assert D.repair_libraries(D.load_config()) == []
    st, r = post(u + "/api/roots/remove", dict(path=str(g)))
    assert st == 200 and r["roots"] == [str(a)]
    assert D.load_config()["libraries"][str(a)]["roots"] == [str(a)]


def test_lib_owns():
    assert D.lib_owns("/x/lib", ["/x/lib"], None)
    assert D.lib_owns("/x/lib", ["/y"], "/x/lib/_designer_data")
    assert not D.lib_owns("/x/lib", ["/y/z"], "/y/z/_designer_data")
    assert not D.lib_owns("/x/lib", None, None)


def test_parent_library_reads_its_runs_own_data(tmp_path, monkeypatch):
    """(2026-11-03g) BN_15_auto/20261101_run_1 was screened (and its AFM list made) while the run's folder was the
    library; opening the parent BN_15_auto showed it "not screened", 5 kept, no AFM lists.  The parent reads the run's
    own _designer_data too, and the run's library reads the parent's."""
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "settings.json")
    parent = tmp_path / "BN_15_auto"
    run = parent / "20261101_run_1"
    make_run(run, "20261101_1")
    make_run(parent, "20261030_run_2")
    rd = run / "_designer_data"
    (rd / "review").mkdir(parents=True)
    (rd / "review" / "a@pc.jsonl").write_text("\n".join(json.dumps(dict(id=f"20261101_run_1:20261101_1-f{i:04d}.a", updated=1000 + i, by="a@pc",
                                                                         body=dict(decision="keep"))) for i in range(30)) + "\n")
    (rd / "afm" / "L1").mkdir(parents=True)
    (rd / "afm" / "L1" / "plan.json").write_text(json.dumps(dict(id="L1", scan="20261101_1", run="20261101_run_1", sites=[], updated=5)))
    (rd / "afm" / "L1" / "001_a.png").write_bytes(b"\x89PNG")
    (rd / "terraces").mkdir()
    (rd / "terraces" / "t1.json").write_text(json.dumps(dict(by="a")))
    (rd / "screened_runs.json").write_text(json.dumps({"20261101_run_1": dict(by="a", at=1)}))
    store = D.Store([str(parent)], parent / "_designer_data", str(parent))
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    u = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        store.scan()
        assert store.related == [rd]
        rev = {r["id"]: r["data"] for r in store.list_review()}
        assert len(rev) == 30 and all(v["decision"] == "keep" for v in rev.values())
        assert [p["id"] for p in store.afm_plans()] == ["L1"]
        assert store.afm_plan("L1")[1] == rd / "afm" / "L1" / "plan.json"
        st, _, b = req(u + "/afmfile/L1/001_a.png")
        assert st == 200 and b == b"\x89PNG"
        assert [t["id"] for t in store.list_terraces()] == ["t1"]
        assert "20261101_run_1" in [s["id"] for s in store.list_seen()]
        # a decision made in the parent (later) wins, and is written to the parent's own data
        store.put_review("20261101_run_1:20261101_1-f0003.a", dict(decision="discard"))
        assert {r["id"]: r["data"] for r in store.list_review()}["20261101_run_1:20261101_1-f0003.a"]["decision"] == "discard"
        assert (parent / "_designer_data" / "review").is_dir() and not (rd / "review" / f"{D.user_name()}.jsonl").exists()
        # Reset screening clears the run's decisions wherever they were made
        assert D._reset_review(store, "20261101_run_1") == 30
        assert not [r for r in store.list_review() if (r["data"] or {}).get("decision")]
        # the run's own library sees the parent's decisions (its folder's parent)
        sub = D.Store([str(run)], rd, str(run))
        sub.scan()
        assert sub.related == [parent / "_designer_data"]
        assert not [r for r in sub.list_review() if (r["data"] or {}).get("decision")]
    finally:
        srv.shutdown()
        srv.server_close()


def test_added_run_outside_library_reads_its_folders_data(tmp_path, monkeypatch):
    """(2026-11-05t) GR13'_auto/20261102_run_1 added by hand while the library is BN_15_auto: its screening and AFM
    lists (in GR13'_auto/_designer_data) were not read, the run looked unscreened with no AFM lists."""
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "settings.json")
    lib = tmp_path / "BN_15_auto"
    make_run(lib, "20261101_run_1")
    gr = tmp_path / "GR13'_auto"
    run = gr / "20261102_run_1"
    make_run(gr, "20261102_run_1")
    gd = gr / "_designer_data"
    (gd / "review").mkdir(parents=True)
    (gd / "review" / "a@pc.jsonl").write_text("\n".join(json.dumps(dict(id=f"20261102_run_1:20261102_1-f{i:04d}.a", updated=1000 + i, by="a@pc",
                                                                         body=dict(decision="keep"))) for i in range(7)) + "\n")
    (gd / "afm" / "L9").mkdir(parents=True)
    (gd / "afm" / "L9" / "plan.json").write_text(json.dumps(dict(id="L9", scan="20261102_1", run="20261102_run_1", sites=[], updated=5)))
    (gd / "screened_runs.json").write_text(json.dumps({"20261102_run_1": dict(by="a", at=1)}))
    store = D.Store([str(lib)], lib / "_designer_data", str(lib), added=[str(lib / "20261101_run_1"), str(run)])
    store.scan()
    assert gd in store.related
    rev = {r["id"]: r["data"] for r in store.list_review()}
    assert len(rev) == 7 and all(v["decision"] == "keep" for v in rev.values())
    assert [p["id"] for p in store.afm_plans()] == ["L9"]
    assert "20261102_run_1" in [s["id"] for s in store.list_seen()]
