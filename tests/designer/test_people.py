"""People (2026-11-23, people.py): each person's run list and loaded runs in their own file on the share, switching
person on a shared computer, and merging the lists of two computers.  python3 -m pytest tests/designer -q"""
import json
import ntpath
import posixpath
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flakeid_designer.__main__ as D  # noqa: E402
import flakeid_designer.people as P  # noqa: E402
from test_server import make_run, req  # noqa: E402


def wait_sync(t=10):
    t0 = time.time()
    while time.time() - t0 < t:
        with D._PSYNC["lock"]:
            if not D._PSYNC["busy"]:
                return
        time.sleep(0.05)
    raise AssertionError("people sync did not finish")


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "settings.json")
    monkeypatch.setattr(D, "legacy_user", lambda: "jpack@labpc")
    share = tmp_path / "share"
    jl, ll = share / "Jordan" / "scanresults", share / "Ludwig" / "exf"
    for base, run in ((jl, "20261004_run_5"), (jl, "20261005_run_1"), (ll, "CL_run_1")):
        make_run(base, run)
    runs = [str(jl / "20261004_run_5"), str(jl / "20261005_run_1")]
    D.save_config(dict(library=str(jl), roots=[str(jl)], data_dir=str(jl / "_designer_data"), added_runs=runs,
                       libraries={str(jl): dict(roots=[str(jl)], data_dir=str(jl / "_designer_data"), used=1)}))
    cfg = D.load_config()
    roots, data, library = D.start_library(cfg)
    store = D.Store(roots, data, library, added=cfg.get("added_runs"))
    D.apply_person(store, cfg, cfg.get("person"))
    srv = D.make_server(0, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    store.scan()
    yield dict(url=f"http://127.0.0.1:{srv.server_address[1]}", store=store, jl=jl, ll=ll, runs=runs)
    wait_sync()
    srv.shutdown()
    srv.server_close()
    D.PERSON.update(name=None, home=None, data=None, doc=None)


def post(url, body):
    st, _, b = req(url, "POST", body)
    return st, json.loads(b or b"{}")


def test_paths_relative_to_home_on_every_computer():
    unc = r"\\NAS\Documents\Microscope Images\Jordan\scanresults"
    run = r"\\NAS\Documents\Microscope Images\Ludwig\exf\CL_run_1"
    rel = P.rel_of(run, unc, ntpath)
    assert rel == "../../Ludwig/exf/CL_run_1"
    assert P.resolve(dict(rel=rel), unc, ntpath) == run
    assert P.resolve(dict(rel=rel), "/Volumes/Documents/Microscope Images/Jordan/scanresults", posixpath) == \
        "/Volumes/Documents/Microscope Images/Ludwig/exf/CL_run_1"
    assert P.resolve(dict(rel="r1"), r"Z:\Jordan\scanresults", ntpath) == r"Z:\Jordan\scanresults\r1"
    assert P.rel_of(r"Y:\other\run", r"Z:\Jordan\scanresults", ntpath) is None      # (another drive: absolute path)
    d = P.set_run(P.new_doc("J"), r"Y:\other\run", True, r"Z:\Jordan\scanresults", 1, ntpath)
    assert P.run_paths(d, r"Z:\Jordan\scanresults", ntpath) == [r"Y:\other\run"]


def test_removed_run_not_brought_back_by_an_old_copy():
    home = "/s/J"
    old = P.set_run(P.set_run(P.new_doc("J"), "/s/J/a", True, home, 1), "/s/J/b", True, home, 2)
    new = P.set_list(json.loads(json.dumps(old)), ["/s/J/a"], home, 5)
    assert P.run_paths(P.merge(old, new), home) == P.run_paths(P.merge(new, old), home) == ["/s/J/a"]
    again = P.set_run(json.loads(json.dumps(new)), "/s/J/b", True, home, 9)      # (added again later: wins)
    assert P.run_paths(P.merge(new, again), home) == ["/s/J/a", "/s/J/b"]


def test_first_person_takes_the_list_second_starts_empty(lab):
    u, store, jl, ll = lab["url"], lab["store"], lab["jl"], lab["ll"]
    st, r = post(u + "/api/people/use", dict(name="Jordan"))
    assert st == 200 and r["person"] == "Jordan"
    wait_sync()
    f = jl / "_designer_data" / "people" / "Jordan.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    assert P.run_paths(doc, str(jl)) == lab["runs"] and doc["aliases"] == ["jpack@labpc"]
    assert {e["rel"] for e in doc["runs"].values()} == {"20261004_run_5", "20261005_run_1"}
    assert D.by_name() == "Jordan" and D.user_name().startswith("Jordan@")
    cfg = D.load_config()
    assert cfg["person"] == "jordan" and cfg["people"]["jordan"]["home"] == str(jl)

    st, r = post(u + "/api/people/use", dict(name="Ludwig"))       # (a second person, same directory)
    assert st == 200 and store.added == []                           # (empty, never seeded from the ticked runs)
    st, r = post(u + "/api/runs/add", dict(path=str(ll / "CL_run_1")))
    assert st == 200 and store.added == [str(ll / "CL_run_1")]
    wait_sync()
    lu = json.loads((jl / "_designer_data" / "people" / "Ludwig.json").read_text(encoding="utf-8"))
    assert P.run_paths(lu, str(jl)) == [str(ll / "CL_run_1")] and lu["aliases"] == []
    assert next(iter(lu["runs"].values()))["rel"] == "../../Ludwig/exf/CL_run_1"

    st, r = post(u + "/api/people/use", dict(name="jordan"))       # (names compared without case)
    assert st == 200 and r["person"] == "Jordan" and store.added == lab["runs"]
    st, _, b = req(u + "/api/people")
    info = json.loads(b)
    assert [p["name"] for p in info["people"]][:2] == ["Jordan", "Ludwig"] and info["aliases"]["jpack@labpc"] == "Jordan"


def test_other_computer_changes_come_back(lab):
    u, store, jl = lab["url"], lab["store"], lab["jl"]
    post(u + "/api/people/use", dict(name="Jordan"))
    wait_sync()
    f = jl / "_designer_data" / "people" / "Jordan.json"
    other = P.set_list(json.loads(f.read_text(encoding="utf-8")), [lab["runs"][0]], str(jl), P.now_ms() + 1000)
    f.write_text(json.dumps(other), encoding="utf-8")                # (the lab PC took a run off the list)
    D.person_sync_soon(store)
    wait_sync()
    assert store.added == [lab["runs"][0]]
    assert D.load_config()["added_runs"] == [lab["runs"][0]]


def test_loaded_runs_follow_the_person(lab):
    u, jl = lab["url"], lab["jl"]
    post(u + "/api/people/use", dict(name="Jordan"))
    st, r = post(u + "/api/people/ticked", dict(paths=[lab["runs"][1]]))
    assert st == 200 and r["ok"]
    wait_sync()
    doc = json.loads((jl / "_designer_data" / "people" / "Jordan.json").read_text(encoding="utf-8"))
    assert doc["ticked"]["keys"] == ["20261005_run_1"]
    assert D.people_info(D.load_config())["ticked"] == [lab["runs"][1]]


def test_share_down_keeps_the_computer_copy(lab, monkeypatch):
    u, store = lab["url"], lab["store"]
    post(u + "/api/people/use", dict(name="Jordan"))
    wait_sync()
    monkeypatch.setattr(P, "save", lambda f, doc, tries=5: None)    # (the share does not answer)
    st, r = post(u + "/api/runs/remove", dict(path=lab["runs"][0]))
    assert st == 200 and store.added == [lab["runs"][1]]
    wait_sync()
    cfg = D.load_config()
    assert P.run_paths(cfg["people"]["jordan"]["doc"], str(lab["jl"])) == [lab["runs"][1]]


def test_bad_names(lab):
    st, r = post(lab["url"] + "/api/people/use", dict(name="  ../  "))
    assert st == 400 and r["error"]


def test_new_computer_finds_the_person_in_the_directory(lab):
    u, store, jl = lab["url"], lab["store"], lab["jl"]
    post(u + "/api/people/use", dict(name="Jordan"))
    wait_sync()
    cfg = D.load_config()                                            # (another computer: nobody chosen, no list)
    cfg.pop("people"), cfg.pop("person")
    cfg["added_runs"] = []
    D.save_config(cfg)
    D.PERSON.update(name=None, home=None, data=None, doc=None)
    store.added = []
    st, _, b = req(u + "/api/people")
    assert [p["name"] for p in json.loads(b)["in_library"]] == ["Jordan"]
    st, r = post(u + "/api/people/use", dict(name="Jordan", home=str(jl)))
    assert st == 200 and store.added == lab["runs"]                  # (the list from the share)


def test_list_saved_on_another_computer_is_left_alone():
    """A list written on Windows (UNC-relative and other-drive runs) saved again on a Mac unchanged, and back."""
    unc, mac = r"\\NAS\D\Jordan\scanresults", "/Volumes/D/Jordan/scanresults"
    d = P.new_doc("J")
    for i, r in enumerate((unc + r"\r1", r"\\NAS\D\Ludwig\r2", r"Y:\other\r3")):
        P.set_run(d, r, True, unc, 10 + i, ntpath)
    P.set_run(d, unc + r"\r0", False, unc, 5, ntpath)
    P.set_ticked(d, [unc + r"\r1"], unc, 20, ntpath)
    before = json.dumps(d, sort_keys=True)
    on_mac = P.run_paths(d, mac, posixpath)
    assert on_mac == ["/Volumes/D/Jordan/scanresults/r1", "/Volumes/D/Ludwig/r2", r"Y:\other\r3"]
    P.set_list(d, on_mac, mac, 99, posixpath)
    assert json.dumps(d, sort_keys=True) == before
    P.set_list(d, on_mac[:2], mac, 100, posixpath)                  # (r3 taken off on the Mac: only that entry changes)
    assert [k for k, e in d["runs"].items() if e["t"] == 100] == [r"Y:/other/r3"]
    P.set_list(d, P.run_paths(d, unc, ntpath), unc, 101, ntpath)
    assert not any(e["t"] == 101 for e in d["runs"].values())
    P.set_ticked(d, ["/Volumes/D/Ludwig/r2"], mac, 102, posixpath)
    assert d["ticked"]["keys"] == ["../../Ludwig/r2"]
    P.set_run(d, mac + "/r0", True, mac, 1, posixpath)             # (a clock behind: still later than the last change)
    assert d["runs"]["r0"]["on"] and d["runs"]["r0"]["t"] == 6
