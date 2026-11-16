"""(2026-11-16) A decision made on one computer is seen on another at once (a page reload, or its next poll): no walk of
the runs needed, not hidden by a network share's cached directory listing, and not held up behind the background
copy of old decisions or by a write to the run's folder that failed once."""
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flakeid_designer.__main__ as D  # noqa: E402
from test_run_decisions import RUN, latest_of, quiet, share  # noqa: E402,F401


def store_for(lib, run_dir, user, monkeypatch):
    monkeypatch.setattr(D, "user_name", lambda: user)
    s = D.Store([lib], lib / "_designer_data", library=str(lib), added=[str(run_dir)])
    s.scan()
    D.RUN_DATA.wait()
    return s


def as_user(user, monkeypatch):
    monkeypatch.setattr(D, "user_name", lambda: user)


def test_first_decision_on_a_run_seen_without_a_walk(tmp_path, monkeypatch):
    """Paul's computer walked the runs before anyone screened this run (no <run>/_designer_data yet)."""
    run_dir, lib_a, lib_p = share(tmp_path)
    p = store_for(lib_p, run_dir, "paul@pc", monkeypatch)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    assert not (run_dir / "_designer_data" / "review" / "avram@mac.jsonl").exists()
    fl = f"{RUN}:BN7_1-f0034"
    as_user("avram@mac", monkeypatch)
    a.put_review(f"{fl}.a", dict(decision="keep", terrace=f"{fl}.a"))
    D.RUN_DATA.wait()
    as_user("paul@pc", monkeypatch)
    assert latest_of(p, f"{fl}.a").get("decision") == "keep"          # (no p.scan())


def test_cached_listing_does_not_hide_a_decision(tmp_path, monkeypatch):
    """A network share's client caches directory listings (Windows: ~10 s): the file's size and time look unchanged."""
    run_dir, lib_a, lib_p = share(tmp_path)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    fl = f"{RUN}:BN7_1-f0034"
    a.put_review(f"{fl}.a", dict(decision="keep", terrace=f"{fl}.a"))
    D.RUN_DATA.wait()
    p = store_for(lib_p, run_dir, "paul@pc", monkeypatch)
    assert latest_of(p, f"{fl}.a").get("decision") == "keep"
    frozen = {}
    real = D.Store._entries

    def cached(d, ext, skip_under=False):                # (the listing as it was: sizes and times not updated)
        k = (str(d), ext, skip_under)
        if k not in frozen:
            frozen[k] = real(d, ext, skip_under)
        return frozen[k]
    monkeypatch.setattr(D.Store, "_entries", staticmethod(cached))
    p.review_state()
    as_user("avram@mac", monkeypatch)
    a.put_review(f"{fl}.a", dict(decision="discard", terrace=f"{fl}.a"))
    D.RUN_DATA.wait()
    as_user("paul@pc", monkeypatch)
    sig0 = p.review_state()[0]
    assert latest_of(p, f"{fl}.a").get("decision") == "discard"
    assert p.review_state()[0] == sig0                   # (unchanged since: the same signature, the page gets 304)


def test_copy_to_the_run_not_held_up_by_the_old_decisions_sync(tmp_path, monkeypatch):
    run_dir, lib_a, lib_p = share(tmp_path)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    gate = threading.Event()
    real_sync = D.RUN_DATA.sync
    monkeypatch.setattr(D.RUN_DATA, "sync", lambda store: (gate.wait(20), real_sync(store)))
    try:
        D.RUN_DATA.sync_soon(a)                          # (a slow share: this one takes long)
        time.sleep(0.2)
        fl = f"{RUN}:BN7_1-f0034"
        a.put_review(f"{fl}.a", dict(decision="keep", terrace=f"{fl}.a"))
        own = run_dir / "_designer_data" / "review" / "avram@mac.jsonl"
        t0 = time.time()
        while not own.is_file() and time.time() - t0 < 3:
            time.sleep(0.05)
        assert own.is_file(), "the decision waited for the sync"
    finally:
        gate.set()
        D.RUN_DATA.wait()


def test_failed_copy_to_the_run_tried_again_soon(tmp_path, monkeypatch):
    run_dir, lib_a, lib_p = share(tmp_path)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    monkeypatch.setattr(D.RunData, "RETRY_S", 0.3)
    real = D.RunData.run_dir
    fails = {"n": 2}

    def flaky(store, run, sub):
        if sub == "review" and fails["n"] > 0:
            fails["n"] -= 1
            raise OSError(60, "Operation timed out")
        return real(store, run, sub)
    monkeypatch.setattr(D.RunData, "run_dir", staticmethod(flaky))
    fl = f"{RUN}:BN7_1-f0034"
    a.put_review(f"{fl}.a", dict(decision="keep", terrace=f"{fl}.a"))
    own = run_dir / "_designer_data" / "review" / "avram@mac.jsonl"
    t0 = time.time()
    while not own.is_file() and time.time() - t0 < 5:
        time.sleep(0.05)
    assert own.is_file() and fails["n"] == 0


def test_decisions_in_another_library_of_the_same_person(tmp_path, monkeypatch):
    """(2026-11-16) A computer with a designer older than 2026-11-10b screened Paul's graphite run from his BN6 library:
    the decisions are only in Paul/exfoliations/BN/BN6/_designer_data.  Another computer adding the run sees them."""
    from test_server import make_run
    mi = tmp_path / "share" / "Microscope Images"
    gr = mi / "Paul" / "exfoliations" / "graphite" / "GR4 - Auto"
    make_run(gr, RUN)
    run_dir = gr / RUN
    bn6 = mi / "Paul" / "exfoliations" / "BN" / "BN6"
    (bn6 / "_designer_data" / "review").mkdir(parents=True)
    fl = f"{RUN}:BN7_1-f0034"
    (bn6 / "_designer_data" / "review" / "afyla@NPGS.jsonl").write_text(
        __import__("json").dumps(dict(id=f"{fl}.a", updated=1000, by="afyla@NPGS", body=dict(decision="keep", terrace=f"{fl}.a"))) + "\n")
    (bn6 / "_designer_data" / "screened_runs.json").write_text(__import__("json").dumps({RUN: dict(by="afyla@NPGS", at=1)}))
    lib = mi / "Avram" / "Exfoliation" / "BN" / "BN_15_auto"
    lib.mkdir(parents=True)
    other = mi / "Naz" / "lib"                            # (another person's library: not read)
    (other / "_designer_data" / "review").mkdir(parents=True)
    (other / "_designer_data" / "review" / "n@pc.jsonl").write_text(
        __import__("json").dumps(dict(id=f"{fl}.a", updated=5000, by="n@pc", body=dict(decision="discard", terrace=f"{fl}.a"))) + "\n")
    a = store_for(lib, run_dir, "avram@mac", monkeypatch)
    assert bn6 / "_designer_data" in a.person_data and other / "_designer_data" not in a.person_data
    assert latest_of(a, f"{fl}.a").get("decision") == "keep"
    assert RUN in [x["id"] for x in a.seen_state()[1]]
