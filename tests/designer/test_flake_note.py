"""(2026-11-25b) A comment per flake, typed in the 100x window: saved as review entry "<flake>.note" with who wrote it;
cleared by an empty one; Reset screening leaves it."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flakeid_designer.__main__ as D  # noqa: E402
from test_run_decisions import RUN, quiet, share, store_for  # noqa: E402,F401

HTML = (ROOT / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def notes(store):
    return {r["id"]: r["data"] for r in store.list_review() if r["id"].endswith(".note")}


def test_note_saved_shared_cleared(tmp_path, monkeypatch):
    run_dir, lib_a, lib_p = share(tmp_path)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    fl = f"{RUN}:BN7_1-f0034"
    a.put_review(f"{fl}.note", dict(note="thin edge, check AFM", flake=fl, run=RUN))
    a.put_review(f"{fl}.a", dict(decision="keep", terrace=f"{fl}.a"))
    n = notes(a)[f"{fl}.note"]
    assert n["note"] == "thin edge, check AFM" and n["flake"] == fl and n["by"]
    p = store_for(lib_p, run_dir, "paul@pc", monkeypatch)          # (another library adding the run sees it)
    assert notes(p)[f"{fl}.note"]["note"] == "thin edge, check AFM"
    assert D._reset_review(a, RUN) == 1                             # (Reset screening: the decision only)
    assert notes(a)[f"{fl}.note"]["note"] == "thin edge, check AFM"
    a.put_review(f"{fl}.note", dict(note=None, flake=fl, run=RUN))
    assert f"{fl}.note" not in notes(a)


def test_page_field():
    assert '<span id="imgCapR"><input type="text" id="imgNote"' in HTML
    assert 'db.collection("review").doc(f.id+".note").set(body)' in HTML
    assert 'if(d.id.endsWith(".note")){if(x.note)NOTES[' in HTML and "STARS={};NOTES={};" in HTML
    assert "function imgStarsSync(){noteSync();" in HTML


def test_note_stamp():
    # (2026-11-25c) shown after the date and time it was written; the text alone while typing
    assert "const noteShown=n=>n&&n.note?" in HTML and "${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}" in HTML
    assert "at:v?at:null" in HTML and 'addEventListener("focus"' in HTML
