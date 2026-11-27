"""(2026-11-27a) A share that hangs (the VPN: 30-60 s per stalled read) never holds up the page: decisions are saved on
this computer first and uploaded when the share answers, the lists and runs come from this computer's last copy,
pictures already seen are served from the local copies, and switching user uses this computer's copy of the person.
python3 -m pytest tests/designer -q"""
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
import flakeid_designer.sharegate as SG  # noqa: E402
from flakeid_designer import kept as KP  # noqa: E402
from test_server import make_run, req, server  # noqa: E402,F401
from test_people import lab, post, wait_sync  # noqa: E402,F401

TID = "20261016_run_1:a_1-f0001.a"


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    for name in ("sync_run", "write_center"):                     # (no kept/ copies or AFM center files)
        monkeypatch.setattr(KP, name, lambda *a, **k: None)
    monkeypatch.setattr(KP.KEPT, "request", lambda *a, **k: None)
    monkeypatch.setattr(D.KEPT_COPY, "want", lambda *a, **k: None)
    yield
    D.OUTBOX.flush(10)


def hang_until(monkeypatch, obj, name, ev):
    orig = getattr(obj, name)

    def slow(*a, **k):
        ev.wait(30)
        return orig(*a, **k)
    monkeypatch.setattr(obj, name, slow)


def review_list(url):
    st, _, b = req(url + "/api/db/review")
    assert st == 200
    return {x["id"]: x["data"] for x in json.loads(b)}


def test_gate_gives_up_and_says_slow():
    g = SG.Gate()
    ev = threading.Event()
    t0 = time.time()
    with pytest.raises(SG.ShareSlow):
        g.call(lambda: ev.wait(10), 0.3, "test")
    assert time.time() - t0 < 2 and g.state() == "slow" and g.brief()["stuck"] == 1
    ev.set()
    time.sleep(0.2)
    assert g.brief()["stuck"] == 0
    assert g.call(lambda: 5, 1.0) == 5
    with pytest.raises(ValueError):
        g.call(lambda: int("x"), 1.0)


def test_decision_saved_here_while_the_share_hangs(server, monkeypatch):
    url, store = server["url"], server["store"]
    ev = threading.Event()
    hang_until(monkeypatch, store, "write_review", ev)
    t0 = time.time()
    st, _, b = req(url + "/api/db/review/" + TID, "PUT", dict(decision="keep", terrace=TID))
    r = json.loads(b)
    assert st == 200 and r["ok"] and r["pending"] and time.time() - t0 < D.SAVE_WAIT_S + 2
    # the page's next poll shows it (before: the poll rebuilt the list from the share and the decision vanished)
    assert review_list(url)[TID]["decision"] == "keep"
    st, _, b = req(url + "/api/offline")
    assert json.loads(b)["outbox"]["pending"] == 1
    # a second decision on the same terrace, still not uploaded: stamped after the first
    req(url + "/api/db/review/" + TID, "PUT", dict(decision="discard", terrace=TID))
    assert review_list(url)[TID]["decision"] == "discard"
    ev.set()                                                        # (the share answers again)
    assert D.OUTBOX.flush(10)
    lines = [json.loads(x) for x in (store.data / "review").glob("*.jsonl").__next__().read_text().splitlines()]
    assert [x["body"]["decision"] for x in lines] == ["keep", "discard"] and lines[1]["updated"] > lines[0]["updated"]
    assert review_list(url)[TID]["decision"] == "discard"


def test_fast_share_saves_before_answering(server):
    url, store = server["url"], server["store"]
    st, _, b = req(url + "/api/db/review/" + TID, "PUT", dict(decision="keep", terrace=TID))
    assert st == 200 and json.loads(b)["uploaded"] is True
    assert "keep" in next((store.data / "review").glob("*.jsonl")).read_text()    # (on the share before the answer)
    st, _, b = req(url + "/api/db/seen/20261016_run_1", "PUT", dict(at=1))
    assert st == 200 and (store.data / "screened_runs.json").is_file()


def test_outbox_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_LOCAL_DIR", str(tmp_path / "local"))
    calls = []

    def fail(r):
        raise OSError(60, "Operation timed out")
    a = SG.Outbox(upload=fail)
    a.add("review", TID, "/share/lib/_designer_data", dict(decision="keep"), entry=dict(id=TID, updated=1, body={}))
    assert a.count() == 1 and not a.flush(0.5)                     # (the designer closed with it not uploaded)
    b = SG.Outbox(upload=calls.append)
    assert b.count() == 1
    b.start()
    assert b.flush(5) and calls[0]["id"] == TID and not (tmp_path / "local" / "outbox.jsonl").exists()


def test_lists_from_the_last_copy_when_the_share_hangs(server, monkeypatch):
    url, store = server["url"], server["store"]
    req(url + "/api/db/review/" + TID, "PUT", dict(decision="keep", terrace=TID))
    assert review_list(url)[TID]["decision"] == "keep"               # (read from the share once)
    ev = threading.Event()
    hang_until(monkeypatch, store, "review_state", ev)
    t0 = time.time()
    assert review_list(url)[TID]["decision"] == "keep"               # (the last copy)
    assert time.time() - t0 < D.LIST_WAIT_S + 1.5
    ev.set()


def test_runs_from_this_computers_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAKEID_DESIGNER_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "share" / "scanresults"
    make_run(root, "20261016_run_1")
    make_run(root, "20261016_run_2")
    added = [str(root / "20261016_run_1"), str(root / "20261016_run_2")]
    s = D.Store([root], root / "_designer_data", library=str(root), added=added)
    assert set(s.scan()) == {"20261016_run_1", "20261016_run_2"}
    # the next start: the runs at once, before any walk of the share
    s2 = D.Store([root], root / "_designer_data", library=str(root), added=added)
    assert s2.prime_runs() and [r["id"] for r in s2.runs_list(wait_s=0)[1]] == ["20261016_run_1", "20261016_run_2"]
    # a walk while one run folder does not answer: that run stays in the list
    real = D.Store._walk

    def walk(self, d, depth, todo, unbuilt):
        if d.name == "20261016_run_2":
            raise OSError(60, "Operation timed out")
        return real(self, d, depth, todo, unbuilt)
    monkeypatch.setattr(D.Store, "_walk", walk)
    assert set(s2.scan()) == {"20261016_run_1", "20261016_run_2"}


def test_pictures_from_the_local_copy(server, monkeypatch):
    url, store = server["url"], server["store"]
    st, _, b = req(url + "/api/db/runs")
    doc = json.loads(b)[0]["data"]
    pic = doc["files"]["x100_nat_a_0123456abc.jpg"]
    st, _, data = req(url + "/_blob/" + pic)
    assert st == 200
    # its name carries its content's hash: served without looking at the share (here: even with it gone)
    f = server["root"] / "20261016_run_1" / "designer" / "x100_nat_a_0123456abc.jpg"
    f.unlink()
    st, _, again = req(url + "/_blob/" + pic)
    assert st == 200 and again == data
    # bundle.json (no hash): from the share, or this computer's copy when the share does not answer in time
    st, _, bj = req(url + "/_blob/" + doc["bundle"])
    assert st == 200
    monkeypatch.setattr(D, "FILE_STAT_S", 0.3)
    ev = threading.Event()
    real = Path.stat

    def stat(self, *a, **k):
        if self.name == "bundle.json" and threading.current_thread().name.startswith("share"):
            ev.wait(10)
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "stat", stat)
    t0 = time.time()
    st, _, bj2 = req(url + "/_blob/" + doc["bundle"])
    assert st == 200 and bj2 == bj and time.time() - t0 < 3
    ev.set()


def test_switch_user_with_the_share_hanging(lab, monkeypatch):
    u, store = lab["url"], lab["store"]
    st, r = post(u + "/api/people/new", dict(path=str(lab["jh"])))
    assert st == 200
    post(u + "/api/people/new", dict(path=str(lab["lh"])))
    wait_sync()
    import flakeid_designer.people as P
    ev = threading.Event()
    real = P.load

    def load(f):
        ev.wait(20)
        return real(f)
    monkeypatch.setattr(P, "load", load)
    monkeypatch.setattr(D, "PERSON_READ_S", 0.5)
    t0 = time.time()
    st, r = post(u + "/api/people/use", dict(name="jordan"))
    assert st == 200 and r["person"] == "Jordan" and time.time() - t0 < 6
    assert store.added == lab["runs"]
    st, _, b = req(u + "/api/db/runs")
    assert {x["data"]["path"] for x in json.loads(b)} == set(lab["runs"])
    ev.set()
