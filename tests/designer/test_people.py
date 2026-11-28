"""People (2026-11-23, people.py): each person's run list and loaded runs in their own file on the share
(2026-11-23e: Microscope Images/<Name>/flakeid_designer_user.json), switching person on a shared computer, merging
the lists of two computers and reading the file again as it changes.  python3 -m pytest tests/designer -q"""
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
    share = tmp_path / "Documents" / "Microscope Images"
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
    yield dict(url=f"http://127.0.0.1:{srv.server_address[1]}", store=store, jl=jl, ll=ll, runs=runs,
               jh=share / "Jordan", lh=share / "Ludwig")
    wait_sync()
    srv.shutdown()
    srv.server_close()
    D.PERSON.update(name=None, home=None, doc=None)


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


def profile(home):
    return json.loads((Path(home) / P.PROFILE).read_text(encoding="utf-8"))


def forget_computer(store):
    """Another computer: nobody chosen, no list."""
    cfg = D.load_config()
    cfg.pop("people", None), cfg.pop("person", None)
    cfg["added_runs"] = []
    D.save_config(cfg)
    D.PERSON.update(name=None, home=None, doc=None)
    store.added = []


def test_person_folder():
    assert P.person_folder("/Volumes/Documents/Microscope Images/Avram/Exfoliation/BN/BN_17_auto", mod=posixpath) == \
        "/Volumes/Documents/Microscope Images/Avram"
    assert P.person_folder(r"\\NAS\Documents\microscope images\Yifan\GR\0407-26", mod=ntpath) == \
        r"\\NAS\Documents\microscope images\Yifan"
    assert P.person_folder(r"Z:\Microscope Images\Yifan", mod=ntpath) == r"Z:\Microscope Images\Yifan"
    assert P.person_folder("/elsewhere/me", mod=posixpath) == "/elsewhere/me"


def test_first_person_takes_the_list_second_starts_empty(lab):
    u, store, jl, ll, jh, lh = lab["url"], lab["store"], lab["jl"], lab["ll"], lab["jh"], lab["lh"]
    st, r = post(u + "/api/people/new", dict(path=str(jl / "20261004_run_5"), ticked=[lab["runs"][1]]))
    assert st == 200 and r["person"] == "Jordan" and r["made"] and r["home"] == str(jh)
    doc = profile(jh)                                                # (written at once, in his own folder)
    assert P.run_paths(doc, str(jh)) == lab["runs"] and doc["aliases"] == ["jpack@labpc"]
    assert {e["rel"] for e in doc["runs"].values()} == {"scanresults/20261004_run_5", "scanresults/20261005_run_1"}
    assert doc["ticked"]["keys"] == ["scanresults/20261005_run_1"] and doc["library"]["rel"] == "scanresults"
    assert D.by_name() == "Jordan" and D.user_name().startswith("Jordan@")
    cfg = D.load_config()
    assert cfg["person"] == "jordan" and cfg["people"]["jordan"]["home"] == str(jh)
    assert D.people_info(cfg)["ticked"] == [lab["runs"][1]]

    st, r = post(u + "/api/people/new", dict(path=str(lh)))         # (a second person: empty)
    assert st == 200 and r["person"] == "Ludwig" and store.added == []
    st, r = post(u + "/api/runs/add", dict(path=str(ll / "CL_run_1")))
    assert st == 200 and store.added == [str(ll / "CL_run_1")]
    wait_sync()
    lu = profile(lh)
    assert P.run_paths(lu, str(lh)) == [str(ll / "CL_run_1")] and lu["aliases"] == []
    assert next(iter(lu["runs"].values()))["rel"] == "exf/CL_run_1"

    st, r = post(u + "/api/people/use", dict(name="jordan"))       # (one used here: from his file)
    assert st == 200 and r["person"] == "Jordan" and not r["made"] and store.added == lab["runs"]
    st, _, b = req(u + "/api/people")
    info = json.loads(b)
    assert [p["name"] for p in info["people"]][:2] == ["Jordan", "Ludwig"] and info["aliases"]["jpack@labpc"] == "Jordan"
    st, r = post(u + "/api/people/use", dict(name="Nobody"))
    assert st == 400 and "New user" in r["error"]


def test_other_computer_changes_come_back_at_once(lab):
    u, store, jh = lab["url"], lab["store"], lab["jh"]
    post(u + "/api/people/new", dict(path=str(jh)))
    wait_sync()
    rev0 = json.loads(req(u + "/api/people/state")[2])["person_rev"]
    assert D.person_check(store) is False                           # (nothing new)
    f = jh / P.PROFILE
    other = P.set_list(profile(jh), [lab["runs"][0]], str(jh), P.now_ms() + 1000)
    P.set_ticked(other, [lab["runs"][0]], str(jh), P.now_ms() + 1000)
    time.sleep(0.05)
    f.write_text(json.dumps(other), encoding="utf-8")                # (the lab PC took a run off and ticked one)
    assert D.person_check(store) is True
    assert store.added == [lab["runs"][0]] and D.load_config()["added_runs"] == [lab["runs"][0]]
    st = json.loads(req(u + "/api/people/state")[2])
    assert st["person_rev"] == rev0 + 1 and st["ticked"] == [lab["runs"][0]] and st["person"] == "Jordan"
    assert D.person_check(store) is False


def test_loaded_runs_follow_the_person(lab):
    u, jh = lab["url"], lab["jh"]
    post(u + "/api/people/new", dict(path=str(jh)))
    rev0 = json.loads(req(u + "/api/people/state")[2])["person_rev"]
    st, r = post(u + "/api/people/ticked", dict(paths=[lab["runs"][1]]))
    assert st == 200 and r["ok"]
    wait_sync()
    assert profile(jh)["ticked"]["keys"] == ["scanresults/20261005_run_1"]
    assert D.people_info(D.load_config())["ticked"] == [lab["runs"][1]]
    D.person_check(lab["store"])                                     # (its own change: not news for the page)
    assert json.loads(req(u + "/api/people/state")[2])["person_rev"] == rev0


def test_share_down_keeps_the_computer_copy(lab, monkeypatch):
    u, store = lab["url"], lab["store"]
    post(u + "/api/people/new", dict(path=str(lab["jh"])))
    wait_sync()
    monkeypatch.setattr(P, "save", lambda f, doc, tries=5: None)    # (the share does not answer)
    st, r = post(u + "/api/runs/remove", dict(path=lab["runs"][0]))
    assert st == 200 and store.added == [lab["runs"][1]]
    wait_sync()
    cfg = D.load_config()
    assert P.run_paths(cfg["people"]["jordan"]["doc"], str(lab["jh"])) == [lab["runs"][1]]


def test_bad_folders(lab, monkeypatch):
    st, r = post(lab["url"] + "/api/people/new", dict(path=str(lab["jh"].parent.parent / "gone")))
    assert st == 400 and "not reachable" in r["error"]
    st, r = post(lab["url"] + "/api/people/new", dict(path=str(lab["jh"] / "not_made_yet")))   # (in Microscope Images:
    assert st == 200 and r["home"] == str(lab["jh"])                                          # the person's folder)
    forget_computer(lab["store"])
    wait_sync()
    (lab["jh"] / P.PROFILE).write_text("{not json", encoding="utf-8")
    st, r = post(lab["url"] + "/api/people/new", dict(path=str(lab["jh"])))
    assert st == 400 and "could not be read" in r["error"]
    (lab["jh"] / P.PROFILE).unlink()
    monkeypatch.setattr(P, "save", lambda f, doc, tries=5: None)
    st, r = post(lab["url"] + "/api/people/new", dict(path=str(lab["jh"])))
    assert st == 400 and "could not be written" in r["error"] and D.PERSON["name"] is None


def test_new_computer_opens_the_persons_file_and_directory(lab):
    """A shared computer with Ludwig's directory open: New user…, a run of Jordan's chosen: his file, runs, loaded runs
    and directory."""
    u, store, jl, ll, jh = lab["url"], lab["store"], lab["jl"], lab["ll"], lab["jh"]
    post(u + "/api/people/new", dict(path=str(jh)))
    post(u + "/api/people/ticked", dict(paths=[lab["runs"][0]]))
    wait_sync()
    forget_computer(store)
    post(u + "/api/library/use", dict(path=str(ll)))
    assert store.library == str(ll)
    st, r = post(u + "/api/people/new", dict(path=str(jl / "20261005_run_1"), ticked=[str(ll / "CL_run_1")]))
    assert st == 200 and not r["made"] and r["home"] == str(jh)
    assert store.library == str(jl) and store.added == lab["runs"]
    assert D.people_info(D.load_config())["ticked"] == [lab["runs"][0]]   # (his, not the page's)


def test_settings_from_before_carried_over(lab):
    """A person chosen with the earlier version (file in a directory's _designer_data): not used as it is; New user…
    makes the file in their folder with that list."""
    u, store, jl, jh = lab["url"], lab["store"], lab["jl"], lab["jh"]
    old = P.set_list(P.new_doc("Jordan", ["jpack@labpc"]), lab["runs"], str(jl), 5)
    P.set_ticked(old, [lab["runs"][1]], str(jl), 6)
    cfg = D.load_config()
    cfg["people"] = {"jordan": dict(name="Jordan", home=str(jl), doc=old, used=1)}
    cfg["person"] = "jordan"
    D.save_config(cfg)
    assert D.apply_person(store, D.load_config(), "jordan") is False and D.PERSON["name"] is None
    assert D.people_info(D.load_config())["people"] == []
    st, r = post(u + "/api/people/new", dict(path=str(jh)))
    assert st == 200 and r["made"]
    doc = profile(jh)
    assert P.run_paths(doc, str(jh)) == lab["runs"] and P.ticked_paths(doc, str(jh)) == [lab["runs"][1]]
    assert doc["aliases"] == ["jpack@labpc"]


def test_library_follows_the_person(lab):
    u, store, jl, ll, jh = lab["url"], lab["store"], lab["jl"], lab["ll"], lab["jh"]
    post(u + "/api/people/new", dict(path=str(jh)))
    post(u + "/api/library/use", dict(path=str(ll)))
    wait_sync()
    assert profile(jh)["library"]["rel"] == "../Ludwig/exf"
    forget_computer(store)                                           # (another computer, Jordan's directory open)
    post(u + "/api/library/use", dict(path=str(jl)))
    st, r = post(u + "/api/people/new", dict(path=str(jh)))
    assert st == 200 and store.library == str(ll)                    # (the directory he had open)


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


def test_last_person_after_a_restart_and_log_out(lab):
    """(2026-11-27e) The person chosen comes back at the next start; Log out leaves nobody, here and at the next start,
    with an empty run list, and the person can come back with everything."""
    u, store, jl = lab["url"], lab["store"], lab["jl"]
    st, r = post(u + "/api/people/new", dict(path=str(jl / "20261004_run_5"), ticked=[]))
    assert st == 200 and r["person"] == "Jordan"
    wait_sync()
    cfg = D.load_config()                                       # (a new start: as main() does)
    s2 = D.Store(store.roots, store.data, store.library, added=cfg.get("added_runs"))
    assert D.apply_person(s2, cfg, cfg.get("person")) and D.PERSON["name"] == "Jordan" and s2.added == lab["runs"]

    st, r = post(u + "/api/people/logout", {})
    assert st == 200 and r["person"] is None and r["was"] == "Jordan" and r["reload"]
    assert D.PERSON["name"] is None and store.added == [] and D.by_name() != "Jordan"
    cfg = D.load_config()
    assert cfg["person"] is None and cfg["added_runs"] == [] and "jordan" in cfg["people"]
    s3 = D.Store(store.roots, store.data, store.library, added=cfg.get("added_runs"))
    assert not D.apply_person(s3, cfg, cfg.get("person")) and s3.added == []
    st, _, b = req(u + "/api/people")
    assert json.loads(b)["person"] is None

    st, r = post(u + "/api/people/use", dict(name="jordan"))   # (back: his list from his file)
    assert st == 200 and r["person"] == "Jordan" and store.added == lab["runs"]


def test_settings_saved_while_windows_holds_the_file(lab, monkeypatch):
    """(2026-11-27e) A rename refused a few times (a virus scanner holding the file) still saves the settings."""
    real, n = D.os.replace, dict(k=0)

    def flaky(a, b):
        n["k"] += 1
        if n["k"] < 3:
            raise PermissionError(13, "in use")
        return real(a, b)
    monkeypatch.setattr(D.os, "replace", flaky)
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    D.save_config(dict(D.load_config(), person="ludwig"))
    assert D.load_config()["person"] == "ludwig" and n["k"] >= 3
