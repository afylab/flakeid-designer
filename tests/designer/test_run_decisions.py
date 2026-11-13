"""(2026-11-10b) A run's decisions travel with the run: written to <run>/_designer_data/review too, copied there for
decisions made before, and read by another computer that adds the run from a different library.  New AFM lists are
made in <run>/_designer_data/afm."""
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flakeid_designer.__main__ as D  # noqa: E402
from flakeid_designer import kept as KP  # noqa: E402
from test_server import make_run, req  # noqa: E402

RUN = "BN7_run_1"


@pytest.fixture(autouse=True)
def quiet(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(D, "CONFIG", tmp_path / "config.json")
    for name in ("sync_run", "write_center"):                     # (no kept/ copies or AFM center files)
        monkeypatch.setattr(KP, name, lambda *a, **k: None)
    monkeypatch.setattr(KP.KEPT, "request", lambda *a, **k: None)
    monkeypatch.setattr(D.KEPT_COPY, "want", lambda *a, **k: None)


def share(tmp_path):
    """Paul's run somewhere on the share; Avram's library and Paul's library elsewhere."""
    run = tmp_path / "share" / "Paul" / "BN7"
    make_run(run, RUN)
    lib_a = tmp_path / "share" / "Avram" / "BN_15_auto"
    lib_p = tmp_path / "share" / "Paul" / "lib"
    lib_a.mkdir(parents=True)
    lib_p.mkdir(parents=True)
    return run / RUN, lib_a, lib_p


def store_for(lib, run_dir, user, monkeypatch):
    monkeypatch.setattr(D, "user_name", lambda: user)
    s = D.Store([lib], lib / "_designer_data", library=str(lib), added=[str(run_dir)])
    s.scan()
    D.RUN_DATA.wait()
    return s


def latest_of(store, tid):
    return ((store.review_state()[2].get(tid) or {}).get("body") or {})


def test_decisions_reach_another_library(tmp_path, monkeypatch):
    run_dir, lib_a, lib_p = share(tmp_path)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    fl = f"{RUN}:BN7_1-f0034"
    a.put_review(f"{fl}.afmdec", dict(afmdec="keep", flake=fl, run=RUN))
    a.put_review(f"{fl}.a", dict(decision="keep", terrace=f"{fl}.a"))
    D.RUN_DATA.wait()
    own = run_dir / "_designer_data" / "review" / "avram@mac.jsonl"
    assert own.is_file() and len(own.read_text().splitlines()) == 2
    assert (lib_a / "_designer_data" / "review" / "avram@mac.jsonl").is_file()       # (the library's copy, as before)
    # Paul's computer: another library, the same run added
    p = store_for(lib_p, run_dir, "paul@pc", monkeypatch)
    assert latest_of(p, f"{fl}.afmdec").get("afmdec") == "keep"
    assert latest_of(p, f"{fl}.a").get("decision") == "keep"
    # Paul changes it: Avram sees the change (the run's folder is read there too)
    p.put_review(f"{fl}.afmdec", dict(afmdec="discard", flake=fl, run=RUN))
    D.RUN_DATA.wait()
    assert latest_of(a, f"{fl}.afmdec").get("afmdec") == "discard"
    # Reset AFM screening on Avram's computer: cleared on Paul's too
    assert D._reset_review(a, RUN, afm=True) == 1
    D.RUN_DATA.wait()
    assert latest_of(p, f"{fl}.afmdec").get("afmdec") is None


def test_old_decisions_copied_to_the_run_once(tmp_path, monkeypatch):
    run_dir, lib_a, lib_p = share(tmp_path)
    fl = f"{RUN}:BN7_1-f0002"
    rv = lib_a / "_designer_data" / "review"
    rv.mkdir(parents=True)
    old = [dict(id=f"{fl}.afmdec", updated=10, by="avram@old", body=dict(afmdec="keep", flake=fl, run=RUN, updated=10)),
           dict(id=f"{fl}.afmdec", updated=20, by="avram@old", body=dict(afmdec="discard", flake=fl, run=RUN, updated=20)),
           dict(id=f"{fl}.stars", updated=15, by="avram@old", body=dict(stars=3, flake=fl, run=RUN, updated=15)),
           dict(id="other_run:x-f0001.a", updated=5, by="avram@old", body=dict(decision="keep", terrace="x"))]
    (rv / "avram@old.jsonl").write_text("".join(json.dumps(e) + "\n" for e in old))
    p0 = store_for(lib_p, run_dir, "paul@pc", monkeypatch)
    assert latest_of(p0, f"{fl}.afmdec") == {}                      # (before: not seen from another library)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)          # (the new version opens: copies them)
    f = run_dir / "_designer_data" / "review" / "avram@mac.jsonl"
    lines = [json.loads(q) for q in f.read_text().splitlines()]
    assert sorted(e["id"] for e in lines) == [f"{fl}.afmdec", f"{fl}.stars"]      # (latest only; not other runs')
    assert next(e for e in lines if e["id"].endswith("afmdec"))["body"]["afmdec"] == "discard"
    a.scan()
    D.RUN_DATA.wait()
    assert len(f.read_text().splitlines()) == 2                      # (once: nothing copied again)
    p = store_for(lib_p, run_dir, "paul@pc", monkeypatch)
    assert latest_of(p, f"{fl}.afmdec").get("afmdec") == "discard" and latest_of(p, f"{fl}.stars").get("stars") == 3


def test_run_inside_its_own_library_not_doubled(tmp_path, monkeypatch):
    """A library that is the run's own folder: its data is the run's data, written once."""
    run_dir, _, _ = share(tmp_path)
    a = store_for(run_dir, run_dir, "avram@mac", monkeypatch)
    a.put_review(f"{RUN}:BN7_1-f0001.a", dict(decision="keep"))
    D.RUN_DATA.wait()
    assert len((run_dir / "_designer_data" / "review" / "avram@mac.jsonl").read_text().splitlines()) == 1


def test_new_afm_list_in_the_run_folder(tmp_path, monkeypatch):
    run_dir, lib_a, lib_p = share(tmp_path)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    srv = D.make_server(0, a)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        s, _, b = req(url + "/api/afm/newlist", "POST", dict(run=RUN, scan="BN7_1", name="List 1", by="avram"),
                         headers={"X-Designer": "1"})
        assert s == 200, b
        pl = json.loads(b)
        assert Path(pl["folder"]).parent == run_dir / "_designer_data" / "afm"
        p = store_for(lib_p, run_dir, "paul@pc", monkeypatch)
        assert [q["id"] for q in p.afm_plans(drafts=True)] == [pl["id"]]
    finally:
        srv.shutdown()
        srv.server_close()
