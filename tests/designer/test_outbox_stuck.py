"""(2026-11-27c) The lab PC's 130 saves: a corrected outline's id has ':' (run:chip-flake), which Windows refuses in a
file name, and the outbox uploaded strictly in order, so everything waited behind it.  Now save files never have ':'
in their names, and one save that cannot be written no longer holds up the others.  python3 -m pytest tests/designer -q"""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flakeid_designer.__main__ as D  # noqa: E402
import flakeid_designer.sharegate as SG  # noqa: E402

WIN_BAD = set('<>:"/\\|?*')
TID = "bn_112126_run_5:bn_112126_5-f0016.outline"


def test_save_files_have_windows_safe_names(tmp_path):
    st = D.Store([], tmp_path / "_designer_data")
    st.put_terrace(TID, dict(kind="outline", flake="bn_112126_run_5:bn_112126_5-f0016", outline=[0, 0, 1, 0, 1, 1]))
    st.put_manual("m1:x", dict(name="a"))
    st.put_design("d1:x", dict(name="b"))
    names = [p.name for p in (tmp_path / "_designer_data").rglob("*")]
    assert names and not any(set(n) & WIN_BAD for n in names), names
    assert [x["id"] for x in st.collection("terraces")[1]] == [TID]
    assert [x["id"] for x in st.collection("manual")[1]] == ["m1:x"]


def test_mac_written_names_still_read_and_replaced(tmp_path):
    d = tmp_path / "_designer_data" / "terraces"
    d.mkdir(parents=True)
    (d / "r:c-f1.u1.json").write_text(json.dumps(dict(flake="r:c-f1", poly=[0, 0, 1, 0, 1, 1], t=5)), encoding="utf-8")
    st = D.Store([], tmp_path / "_designer_data")
    assert [x["id"] for x in st.collection("terraces")[1]] == ["r:c-f1.u1"]
    st.put_terrace("r:c-f1.u1", dict(flake="r:c-f1", poly=[0, 0, 2, 0, 2, 2], t=6))
    assert [p.name for p in d.glob("*.json")] == ["r%3Ac-f1.u1.json"]
    assert st.collection("terraces")[1][0]["data"]["t"] == 6
    st.delete_terrace("r:c-f1.u1")
    assert not list(d.glob("*.json"))


@pytest.fixture()
def fast(monkeypatch, tmp_path):
    monkeypatch.setenv("FLAKEID_DESIGNER_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.setattr(SG, "RETRY_S", (0.05, 0.05))
    monkeypatch.setattr(SG, "STUCK_RETRY_S", 60.0)
    share = tmp_path / "share" / "_designer_data"
    share.parent.mkdir(parents=True)
    return share


def wait_for(cond, t=5.0):
    t0 = time.time()
    while time.time() - t0 < t:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_one_bad_save_does_not_hold_up_the_others(fast):
    done = []

    def up(r):
        if r["id"] == "bad":
            raise FileNotFoundError(2, "No such file or directory", "x:bad.tmp")
        done.append(r["id"])
    ob = SG.Outbox(upload=up)
    ob.add("terraces", "bad", str(fast), {})
    for i in range(5):
        ob.add("review", f"good{i}", str(fast), {})
    assert wait_for(lambda: len(done) == 5)
    assert done == [f"good{i}" for i in range(5)]
    assert wait_for(lambda: ob.brief()["n_stuck"] == 1)
    b = ob.brief()
    assert b["pending"] == 1 and b["stuck"][0]["id"] == "bad" and "FileNotFoundError" in b["stuck"][0]["error"]
    assert ob.discard(b["stuck"][0]["n"]) and ob.count() == 0
    kept = (Path(ob.path()).with_name("outbox_discarded.jsonl")).read_text(encoding="utf-8")
    assert json.loads(kept)["id"] == "bad"


def test_share_not_reachable_is_not_the_saves_fault(fast, tmp_path):
    gone = tmp_path / "unplugged" / "_designer_data"            # (its folder not there: the share, not the save)
    calls = []

    def up(r):
        calls.append(r["id"])
        raise FileNotFoundError(2, "No such file or directory")
    ob = SG.Outbox(upload=up)
    ob.add("review", "a", str(gone), {})
    assert wait_for(lambda: len(calls) >= 4)
    assert ob.brief()["n_stuck"] == 0 and not ob.items[0].get("tries")


def test_older_version_of_a_document_never_lands_after_a_newer_one(fast):
    order, fail = [], {"first": True}

    def up(r):
        if r["body"].get("v") == 1 and fail["first"]:
            fail["first"] = False
            raise PermissionError(13, "locked")
        order.append((r["id"], r["body"]["v"]))
    ob = SG.Outbox(upload=up)
    ob.add("terraces", "x", str(fast), dict(v=1))
    ob.add("terraces", "x", str(fast), dict(v=2))
    ob.add("terraces", "y", str(fast), dict(v=1))
    assert wait_for(lambda: ob.count() == 0)
    assert [o for o in order if o[0] == "x"] == [("x", 1), ("x", 2)] and ("y", 1) in order
