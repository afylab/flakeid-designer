"""(2026-11-24) View › ＋ AFM scan (/api/afm/addscan, afm_add_scan) and the AFM list window's Manual AFM (a manual list,
afmplan.make_manual_plan): an AFM file chosen by hand linked to a flake.  A flake in no AFM list goes in the chip's
"Added by hand" list; a second file of the same flake is one more file of its point.  The first draft took the store's
lock and then asked for the run's folder, which takes the same (not re-entrant) lock: the designer froze."""
import json
import threading
from pathlib import Path

from flakeid_designer import __main__ as M
from flakeid_designer import afmplan as AP

HTML = (Path(M.__file__).parent / "index.html").read_text(encoding="utf-8")


class Store:
    """The parts of Store that afm_add_scan uses: lists in one folder, the lock, no runs."""

    def __init__(self, d):
        self.d, self.lock, self.runs, self.data = Path(d), threading.Lock(), {}, None

    def afm_plans(self, scan=None, drafts=False):
        out = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(self.d.glob("*/plan.json"))]
        return [p for p in out if scan is None or p.get("scan") == scan]

    def afm_plan(self, pid):
        f = self.d / str(pid) / "plan.json"
        return (json.loads(f.read_text(encoding="utf-8")), f) if f.is_file() else (None, None)


def add(store, d, **b):
    """afm_add_scan in a thread: a lock taken twice would hang, so a hang fails the test instead of stopping it."""
    out = {}
    t = threading.Thread(target=lambda: out.update(r=M.afm_add_scan(store, AP, b, d)), daemon=True)
    t.start()
    t.join(20)
    assert not t.is_alive(), "afm_add_scan hung (the store lock taken twice?)"
    return out["r"]


def test_add_scan_by_hand(tmp_path, monkeypatch):
    monkeypatch.setattr(AP, "analyse_file", lambda f, png: dict(file=Path(f).name, png=png.name, levels=[], scan_um=20.0))
    afm = tmp_path / "afm"
    afm.mkdir()
    s1, s2 = tmp_path / "a.spm", tmp_path / "b.spm"
    s1.write_bytes(b"1")
    s2.write_bytes(b"2")
    store = Store(afm)
    flake = dict(run="r1", scan="r1:chip_1", flake="r1:chip_1-f0007", terrace="t1", t_nm=4.2, px=[10.0, 20.0])
    r = add(store, afm, file=str(s1), **flake)
    pl = r["plan"]
    assert pl["manual"] and pl["byhand"] and pl["name"] == "Added by hand" and pl["run"] == "r1"
    assert [s["flake"] for s in pl["sites"]] == ["r1:chip_1-f0007"] and r["site"] == 1
    assert pl["results"]["1"]["source"] == str(s1)
    # a second file of the same flake (as View sends it: the list and point it is in): one more file of that point
    r2 = add(store, afm, file=str(s2), id=pl["id"], site=1, **flake)
    pl2, _ = store.afm_plan(pl["id"])
    assert r2["site"] == 1 and len(pl2["sites"]) == 1
    assert pl2["results"]["1"]["source"] == str(s1) and [x["source"] for x in pl2["more"]["1"]] == [str(s2)]
    # another flake of the chip: the same "Added by hand" list, a new point
    r3 = add(store, afm, file=str(s1), **dict(flake, flake="r1:chip_1-f0009"))
    assert r3["plan"]["id"] == pl["id"] and r3["site"] == 2
    assert len(store.afm_plans("r1:chip_1")) == 1
    # no file chosen / no such file
    monkeypatch.setattr(M, "pick_file", lambda *a, **k: None)
    assert add(store, afm, **flake) == dict(cancelled=True)
    assert "not found" in add(store, afm, file=str(tmp_path / "gone.spm"), **flake)["error"]


def test_manual_plan_has_no_fit(tmp_path):
    pl = AP.make_plan(dict(manual=True, scan="r1:chip_1", run="r1", sites=[dict(flake="f1"), dict(flake="f2")]), tmp_path)
    assert pl["manual"] and pl["transform"] is None and pl["pgm"] is None and pl["name"] == "Manual AFM"
    assert [s["n"] for s in pl["sites"]] == [1, 2]


def test_page_buttons():
    assert 'id="imgAfmAdd"' in HTML and '$("imgAfmAdd").onclick=()=>imgAfmAdd()' in HTML
    assert 'afmPost("/api/afm/addscan",body)' in HTML
    # Manual AFM only while a list is set up (a made list keeps its fit and point file); a manual list: no fit section
    assert 'else if(!P&&!done&&S.length)h+=`<div class="afmsec"><div class="row"><button id="afmManual"' in HTML
    assert "if(P&&!MAN){const T=P.transform;" in HTML and 'if(P&&$("afmSave")){' in HTML
